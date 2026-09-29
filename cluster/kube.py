"""kubectl subprocesses and verified streaming transfers; no Kubernetes SDK."""

import json
import shutil
import subprocess
import tarfile
import threading
import uuid
from pathlib import Path

from cluster.control import APP, run_label, verify_download


def safe_extract(archive, root):
    root = Path(root).resolve()
    for member in archive:
        target = (root / member.name).resolve()
        if not target.is_relative_to(root) or Path(member.name).is_absolute() or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe archive member: ' + member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)


class Kube:
    def __init__(self, cfg):
        self.cfg, self.helper = cfg, None
        self.helper_lock = threading.Lock()
        self.command = ['kubectl', '--context', cfg['context'], '-n', cfg['namespace'], '--request-timeout=30s']

    def run(self, *args, input=None, timeout=60):
        result = subprocess.run(self.command + list(args), input=input, text=True,
                                capture_output=True, timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout

    def create(self, manifest):
        try:
            return json.loads(self.run('create', '-f', '-', '-o', 'json', input=json.dumps(manifest)))
        except RuntimeError as error:
            if 'AlreadyExists' not in str(error):
                raise
            existing = json.loads(self.run('get', manifest['kind'].lower(), manifest['metadata']['name'], '-o', 'json'))
            expected = manifest['metadata']['labels']
            if any(existing['metadata'].get('labels', {}).get(k) != v for k, v in expected.items()):
                raise ValueError('Existing workload has another owner') from error
            return existing

    def snapshot(self, run_id):
        return json.loads(self.run('get', 'jobs,pods', '-l',
            f'app.kubernetes.io/name={APP},mayn-run={run_label(run_id)}', '-o', 'json'))['items']

    def choose_gpu(self, exp):
        from cluster.gpu import pools
        nodes = json.loads(self.run('get', 'nodes', '-o', 'json'))['items']
        try:
            pod_data = self.run('get', 'pods', '--all-namespaces', '-o', 'json', timeout=60)
        except RuntimeError as error:
            if 'Forbidden' not in str(error):
                raise
            # Namespace-only accounts cannot see other tenants' reservations;
            # availability is an estimate and Kubernetes makes the final placement.
            pod_data = self.run('get', 'pods', '-o', 'json', timeout=60)
        pods = json.loads(pod_data)['items']
        quotas = json.loads(self.run('get', 'resourcequota', '-o', 'json'))['items']
        options = pools(nodes, pods, quotas, self.cfg, exp['mode'])
        if not options:
            raise RuntimeError('No accessible GPU pool meets the architecture/VRAM/quota constraints')
        chosen = max(options, key=lambda p: (p['free'] > 0, p['free'] / (1 + p['backlog']),
                                            -p['backlog'] / p['capacity']))
        return {'resource': chosen['resource'], 'products': chosen['products'],
                'min_memory_mib': self.cfg['gpu_min_memory_mib'][exp['mode']]}

    def logs(self, job):
        try:
            return self.run('logs', 'job/' + job, '--all-containers=true', '--tail=250', timeout=40)
        except RuntimeError as error:
            return 'Logs unavailable: ' + str(error)

    def events(self, job):
        # Include pod scheduling/eviction events, which are not attached to the Job.
        pods = json.loads(self.run('get', 'pods', '-l', 'job-name=' + job, '-o', 'json'))['items']
        names = {job} | {p['metadata']['name'] for p in pods}
        events = json.loads(self.run('get', 'events', '-o', 'json'))['items']
        return [{k: e.get(k) for k in ('reason', 'message', 'type', 'lastTimestamp')}
                for e in events if e.get('involvedObject', {}).get('name') in names][-30:]

    def login(self):
        with self.helper_lock:
            return self._login()

    def _login(self):
        helper = self.helper
        if helper:
            value = self.run('get', 'pod', helper, '--ignore-not-found', '-o', 'json')
            pod = json.loads(value) if value else {}
            if pod.get('status', {}).get('phase') == 'Running':
                return helper
        name = 'mayn-login-' + uuid.uuid4().hex[:12]
        resources = {'cpu': '1', 'memory': '2Gi', 'ephemeral-storage': '1Gi'}
        manifest = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {
            'name': name, 'namespace': self.cfg['namespace'],
            'labels': {'app.kubernetes.io/name': APP, 'mayn-kind': 'login'}}, 'spec': {
            'restartPolicy': 'Never', 'activeDeadlineSeconds': 3600, 'automountServiceAccountToken': False,
            'containers': [{'name': 'login', 'image': self.cfg['login_image'], 'command': ['sleep', '3600'],
                'resources': {'requests': resources, 'limits': dict(resources)},
                'volumeMounts': [{'name': 'data', 'mountPath': '/mnt'}]}],
            'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': self.cfg['pvc']}}]}}
        self.create(manifest)
        self.run('wait', '--for=condition=Ready', 'pod/' + name, '--timeout=180s', timeout=200)
        self.helper = name
        return name

    def preflight(self):
        self.run('get', 'pvc', self.cfg['pvc'], '-o', 'name')
        if self.cfg['image_pull_secret']:
            self.run('get', 'secret', self.cfg['image_pull_secret'], '-o', 'name')
        pod = self.login()
        script = '''
import json, os, pathlib, sys
c=json.loads(sys.argv[1]); root=pathlib.Path(c['remote_root']); root.mkdir(parents=True,exist_ok=True)
p=root/('.probe-'+sys.argv[2]); p.write_text('ok'); p.unlink()
if 'map' in c['modes']:
    for path in (pathlib.Path(c['dino_weights']),):
        if not path.is_file(): raise SystemExit('Missing DINO asset: '+str(path))
    if pathlib.Path(c['dino_weights']).name != 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth':
        raise SystemExit('Expected the pinned DINOv3 ViT-L/16 checkpoint')
print(json.dumps({'pvc_write': True, 'available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize}))
'''
        return json.loads(self.run('exec', pod, '--', 'python', '-c', script,
                                  json.dumps(self.cfg), uuid.uuid4().hex, timeout=90))

    def fetch(self, remote, destination):
        destination = Path(destination)
        if destination.exists():
            verify_download(destination)
            return
        # The source must belong to this project's root, even when resuming saved state.
        if not Path(remote).is_relative_to(Path(self.cfg['remote_root'])) or '..' in Path(remote).parts:
            raise ValueError('Refusing transfer outside the project root')
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        archive_path = destination.with_name(destination.name + '.partial.tar')
        if partial.exists():
            shutil.rmtree(partial)  # Only this attempt's incomplete local transfer.
        partial.mkdir()
        pod = self.login()
        # Exec streams large artifacts to disk rather than buffering them in memory.
        with archive_path.open('wb') as stream:
            command = self.command[:-1] + ['--request-timeout=0']
            result = subprocess.run(command + ['exec', pod, '--', 'tar', '-C', remote, '-cf', '-', '.'],
                                    stdout=stream, stderr=subprocess.PIPE, timeout=3300, check=False)
        if result.returncode:
            with self.helper_lock:
                if self.helper == pod:
                    self.helper = None
            raise RuntimeError(result.stderr.decode(errors='replace'))
        try:
            with tarfile.open(archive_path, 'r:') as archive:
                safe_extract(archive, partial)
        except tarfile.TarError as error:
            raise RuntimeError('Incomplete/corrupt download: ' + str(error)) from error
        verify_download(partial)
        partial.rename(destination)
        archive_path.unlink()
