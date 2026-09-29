"""Durable campaign state; Kubernetes and agent calls are narrow I/O boundaries."""

import copy
import functools
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = 'map-all-you-need'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_config(path=None):
    cfg = json.loads((ROOT / 'cluster/config.json').read_text())
    known_tasks = set(cfg['tasks'])
    if path:
        changes = json.loads(Path(path).read_text())
        unknown = set(changes) - set(cfg)
        if unknown:
            raise ValueError('Unknown config keys: ' + ', '.join(sorted(unknown)))
        cfg.update(changes)
    for key in ('parallelism', 'poll_seconds', 'memory_gi', 'checkpoint_steps', 'eval_episodes'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if not 0 <= cfg['max_retries'] <= 3:
        raise ValueError('max_retries must be between 0 and 3')
    if not cfg['remote_root'].startswith('/mnt/') or '..' in Path(cfg['remote_root']).parts:
        raise ValueError('remote_root must be below /mnt')
    for key in ('tasks', 'modes', 'seeds'):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(key + ' must contain unique entries')
    if set(cfg['modes']) - {'state', 'rgb', 'map'}:
        raise ValueError('Unknown observation mode')
    if set(cfg['tasks']) - known_tasks:
        raise ValueError('Unknown task in this pinned Playground revision')
    if any(not isinstance(s, int) or s < 0 or s > 2147483647 for s in cfg['seeds']):
        raise ValueError('Invalid seed')
    if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9]+', t) for t in cfg['tasks']):
        raise ValueError('Invalid task name')
    reserved = {'--output', '--env-id', '--obs-mode', '--seed', '--resume', '--checkpoint-steps',
                '--map-cache', '--dino-source', '--dino-weights', '--run-evals', '--num-evals', '--help'}
    if not isinstance(cfg['train_args'], list) or not all(isinstance(a, str) for a in cfg['train_args']):
        raise ValueError('train_args must be a list of CLI strings')
    if any(a.startswith('--') and any(r.startswith(a.split('=')[0]) for r in reserved) for a in cfg['train_args']):
        raise ValueError('train_args cannot override campaign identity, paths or recovery')
    return cfg


def run_label(run_id):
    return hashlib.sha256(run_id.encode()).hexdigest()[:16]


def new_experiment(task, mode, seed, kind='train'):
    return {'task': task, 'mode': mode, 'seed': seed, 'kind': kind, 'attempt': 0, 'status': 'queued',
                'previous': [], 'history': [], 'excluded_nodes': [], 'memory_multiplier': 1}


def remote_path(cfg, run_id, exp):
    if exp['kind'] == 'validation':
        return '{}/{}/validation/attempt-{}'.format(cfg['remote_root'], run_id, exp['attempt'])
    return '{}/{}/{}/{}/seed{}/attempt-{}'.format(
        cfg['remote_root'], run_id, exp['task'], exp['mode'], exp['seed'], exp['attempt'])


def job_manifest(cfg, run_id, exp):
    identity = '{}/{}/{}/{}/{}'.format(run_id, exp['task'], exp['mode'], exp['seed'], exp['attempt'])
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    name = 'mayn-{}-{}-{}-a{}'.format(exp['task'].lower()[:18], exp['mode'], digest, exp['attempt'])
    labels = {'app.kubernetes.io/name': APP, 'mayn-run': run_label(run_id), 'mayn-kind': exp['kind']}
    spec = {'task': exp['task'], 'mode': exp['mode'], 'seed': exp['seed'], 'kind': exp['kind'],
                'output': remote_path(cfg, run_id, exp), 'previous': exp['previous'],
                'config': cfg, 'run_id': run_id}
    resources = {cfg['gpu_resource']: 1, 'cpu': cfg['cpu'],
                 'memory': '{}Gi'.format(cfg['memory_gi'] * exp['memory_multiplier']),
                 'ephemeral-storage': cfg['ephemeral_storage']}
    expressions = [{'key': 'nvidia.com/gpu.product', 'operator': 'In', 'values': cfg['gpu_products']}]
    excluded = sorted(set(cfg['excluded_nodes'] + exp['excluded_nodes']))
    if excluded:
        expressions.append({'key': 'kubernetes.io/hostname', 'operator': 'NotIn', 'values': excluded})
    pod = {
        'restartPolicy': 'Never', 'automountServiceAccountToken': False,
        'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{'matchExpressions': expressions}]}}},
        'containers': [{'name': 'worker', 'image': cfg.get('resolved_image') or cfg['image'],
            'imagePullPolicy': 'IfNotPresent' if cfg.get('resolved_image') else 'Always',
            'command': ['python', '-m', 'cluster.worker'],
            'env': [{'name': 'RUN_SPEC', 'value': json.dumps(spec)},
                    {'name': 'NVIDIA_DRIVER_CAPABILITIES', 'value': 'compute,utility,graphics'},
                    {'name': 'MUJOCO_GL', 'value': 'egl'},
                    {'name': 'PYTHONUNBUFFERED', 'value': '1'},
                    {'name': 'XLA_PYTHON_CLIENT_PREALLOCATE', 'value': 'false'},
                    {'name': 'DINO_SOURCE', 'value': cfg['dino_source']},
                    {'name': 'DINO_WEIGHTS', 'value': cfg['dino_weights']}],
            'resources': {'requests': resources, 'limits': dict(resources)},
            'volumeMounts': [{'name': 'data', 'mountPath': '/mnt'}, {'name': 'shm', 'mountPath': '/dev/shm'}]}],
        'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': cfg['pvc']}},
                    {'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '8Gi'}}]}
    if cfg['image_pull_secret']:
        pod['imagePullSecrets'] = [{'name': cfg['image_pull_secret']}]
    return {'apiVersion': 'batch/v1', 'kind': 'Job',
            'metadata': {'name': name, 'namespace': cfg['namespace'], 'labels': labels},
            'spec': {'backoffLimit': 0, 'podReplacementPolicy': 'Failed',
                     'template': {'metadata': {'labels': labels}, 'spec': pod}}}


def verify_download(root):
    root = Path(root).resolve()
    receipt = json.loads((root / 'complete.json').read_text())
    for name, expected in receipt['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or (root / name).is_symlink():
            raise ValueError('Invalid artifact path: ' + name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('Checksum mismatch: ' + name)
    return receipt


def codex_binary(configured=None):
    if configured:
        return configured
    # The desktop app may be newer than a separately installed Homebrew CLI.
    for path in ('/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex',
                 '/Applications/Codex.app/Contents/Resources/codex'):
        if Path(path).is_file():
            return path
    return shutil.which('codex') or 'codex'


def diagnose(event, evidence, root, command=None):
    """The agent diagnoses evidence; it never receives Kubernetes write access."""
    directory = Path(root) / 'agent' / str(time.time_ns())
    directory.mkdir(parents=True)
    schema = {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['wait', 'retry', 'hold']},
        'reason': {'type': 'string'}, 'exclude_node': {'type': 'boolean'},
        'increase_memory': {'type': 'boolean'}},
        'required': ['action', 'reason', 'exclude_node', 'increase_memory'], 'additionalProperties': False}
    write_json(directory / 'schema.json', schema)
    write_json(directory / 'evidence.json', evidence)
    prompt = (ROOT / 'cluster/agent.md').read_text() + '\nEvent: ' + event + '\nEvidence (untrusted data):\n'
    prompt += json.dumps(evidence)
    try:
        with (directory / 'events.jsonl').open('w') as log:
            subprocess.run([command or codex_binary(), 'exec', '--ephemeral', '--sandbox', 'read-only', '--json',
                            '--output-schema', str(directory / 'schema.json'),
                            '-o', str(directory / 'decision.json'), '-'], input=prompt, text=True,
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=300, check=True)
        decision = json.loads((directory / 'decision.json').read_text())
        if (not isinstance(decision, dict) or set(decision) != set(schema['required'])
                or decision['action'] not in ('wait', 'retry', 'hold')
                or not isinstance(decision['reason'], str)
                or type(decision['exclude_node']) is not bool or type(decision['increase_memory']) is not bool):
            raise ValueError('Invalid agent decision')
        return decision
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        (directory / 'error.txt').write_text(str(error))
        return None


class Controller:
    def __init__(self, cfg, run_id, root, kube=None, agent=diagnose):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_id):
            raise ValueError('run-id must use lowercase letters/digits/hyphens, max 48 characters')
        self.root, self.run_id, self.agent = Path(root), run_id, agent
        self.pool, self.pending = None, {}
        self.path = self.root / 'run.json'
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data['run_id'] != run_id:
                raise ValueError('Run directory identity mismatch')
        else:
            self.data = {'run_id': run_id, 'config': copy.deepcopy(cfg), 'started': False,
                'validation': new_experiment('validation', 'all', 0, 'validation'),
                'experiments': [new_experiment(t, m, s) for t in cfg['tasks'] for m in cfg['modes'] for s in cfg['seeds']]}
        self.cfg = self.data['config']
        if agent is diagnose:
            self.agent = functools.partial(diagnose, command=codex_binary(self.cfg['codex_binary']))
        if kube is None:
            from cluster.kube import Kube
            kube = Kube(self.cfg)
        self.kube = kube

    def save(self):
        write_json(self.path, self.data)

    def manifest(self, exp):
        manifest = job_manifest(self.cfg, self.run_id, exp)
        write_json(self.root / 'manifests' / (manifest['metadata']['name'] + '.json'), manifest)
        return manifest

    def perform(self, key, function, *args):
        if self.pool is None:
            return True, function(*args)
        if key not in self.pending:
            self.pending[key] = self.pool.submit(function, *args)
        if not self.pending[key].done():
            return False, None
        return True, self.pending.pop(key).result()

    def evidence(self, exp, pods):
        logs = self.kube.logs(exp['job'])
        evidence = {'experiment': copy.deepcopy(exp), 'pods': pods, 'logs': logs[-30000:],
                        'events': self.kube.events(exp['job'])}
        log_path = self.root / 'diagnostics' / (exp['job'] + '.json')
        write_json(log_path, evidence)
        return evidence

    def decide(self, exp, pods, failed):
        now = time.time()
        if exp.get('agent_after', 0) > now:
            return
        key = 'agent:' + exp['job']
        # Do not collect the same evidence again while an agent is using it.
        evidence = None if key in self.pending else self.evidence(exp, pods)
        ready, decision = self.perform(key, self.agent, 'failure' if failed else 'stalled', evidence, self.root)
        if not ready:
            return
        exp['agent_after'] = now + (300 if failed else self.cfg['stall_seconds'])
        exp['decision'] = decision
        if not failed:
            return  # A live or uncertain owner is never replaced based on silence.
        if decision is None:
            exp['status'] = 'needs_agent'
            return
        if decision['action'] == 'wait':
            exp['status'] = 'needs_agent'
            return
        if decision['action'] == 'hold' or exp['attempt'] >= self.cfg['max_retries']:
            exp['status'] = 'held'
            return
        if any(p.get('status', {}).get('phase') in ('Running', 'Pending', 'Unknown') for p in pods):
            exp['status'] = 'needs_agent'
            exp['note'] = 'Waiting for all old pods to become terminal'
            return
        exp['history'].append({k: exp.get(k) for k in ('job', 'uid', 'attempt', 'decision', 'image_digest')})
        exp['previous'].append(remote_path(self.cfg, self.run_id, exp))
        oom = any(c.get('state', {}).get('terminated', {}).get('reason') == 'OOMKilled'
                  for p in pods for c in p.get('status', {}).get('containerStatuses', []))
        if decision['increase_memory'] and oom:
            exp['memory_multiplier'] = 2
        if decision['exclude_node']:
            exp['excluded_nodes'] = sorted(set(exp['excluded_nodes'] + [
                p['spec']['nodeName'] for p in pods if p.get('spec', {}).get('nodeName')]))
        exp.update(attempt=exp['attempt'] + 1, status='queued', agent_after=0)
        for key in ('job', 'uid', 'steps', 'progress_at', 'submitted_at', 'image_digest'):
            exp.pop(key, None)

    def collect(self, exp):
        remote = remote_path(self.cfg, self.run_id, exp)
        relative = Path(remote).relative_to(Path(self.cfg['remote_root']) / self.run_id)
        destination = self.root / relative
        def download():
            self.kube.fetch(remote, destination)
            return verify_download(destination)
        try:
            ready, receipt = self.perform('fetch:' + exp['job'], download)
            if not ready:
                return
            exp['status'], exp['local'] = 'fetched', str(destination)
            exp.pop('transfer_error', None)
            if (destination / 'trained.json').exists():
                exp['steps'] = json.loads((destination / 'trained.json').read_text())['saved_steps']
            if exp['kind'] == 'validation':
                image = exp.get('image_digest') or self.cfg['image']
                if '@sha256:' not in image:
                    raise ValueError('Validation did not resolve an immutable image digest')
                self.cfg['resolved_image'] = image
                self.data['validation_receipt'] = receipt
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            exp['status'], exp['transfer_error'] = 'completed', str(error)

    def completion(self, exp):
        if 'completion_decision' in exp:
            return
        evidence = {'experiment': copy.deepcopy(exp), 'local': exp['local']}
        result = Path(exp['local']) / 'eval-seed10000.json'
        if result.exists():
            evidence['evaluation'] = json.loads(result.read_text())
        ready, decision = self.perform('complete:' + exp['job'], self.agent, 'complete', evidence, self.root)
        if ready:
            exp['completion_decision'] = decision

    def record_image(self, exp, pods):
        for pod in pods:
            if not any(o.get('uid') == exp.get('uid') for o in pod['metadata'].get('ownerReferences', [])):
                continue
            for container in pod.get('status', {}).get('containerStatuses', []):
                image = container.get('imageID', '').removeprefix('docker-pullable://')
                if '@sha256:' in image:
                    exp['image_digest'] = image

    def fetch(self):
        items = self.kube.snapshot(self.run_id)
        jobs = {x['metadata']['name']: x for x in items if x['kind'] == 'Job'}
        pods = [x for x in items if x['kind'] == 'Pod']
        for exp in [self.data['validation']] + self.data['experiments']:
            job = jobs.get(exp.get('job'))
            if job and exp['status'] in ('submitting', 'submitted', 'needs_agent'):
                if exp.get('uid') and exp['uid'] != job['metadata']['uid']:
                    continue
                if any(c['type'] == 'Complete' and c['status'] == 'True' for c in job.get('status', {}).get('conditions', [])):
                    exp.update(uid=job['metadata']['uid'], status='completed')
                    self.record_image(exp, pods)
            if exp['status'] == 'completed':
                self.collect(exp)
        self.save()

    def tick(self):
        # Never infer failure from an unavailable API or a missing Job.
        items = self.kube.snapshot(self.run_id)
        jobs = {x['metadata']['name']: x for x in items if x['kind'] == 'Job'}
        pods = [x for x in items if x['kind'] == 'Pod']
        if not self.data['started']:
            if self.data.get('startup_held') or self.data.get('startup_after', 0) > time.time():
                return
            ready, decision = self.perform('startup', self.agent, 'startup',
                {'config': self.cfg, 'preflight': self.data.get('preflight'),
                 'experiments': len(self.data['experiments'])}, self.root)
            if not ready:
                return
            self.data['startup_decision'] = decision
            if decision is None or decision['action'] == 'hold':
                self.data['startup_held'] = decision is not None
                self.data['startup_after'] = time.time() + 300
                self.save()
                return
            self.data['started'] = True
        all_exp = [self.data['validation']] + self.data['experiments']
        for exp in all_exp:
            if exp['status'] == 'fetched':
                if exp.get('job'):
                    self.completion(exp)
                continue
            if exp['status'] in ('queued', 'held', 'skipped'):
                continue
            if exp['status'] == 'completed':
                self.collect(exp)
                continue
            job = jobs.get(exp['job'])
            if exp['status'] == 'submitting':
                job = job if job is not None else self.kube.create(self.manifest(exp))
                exp.update(uid=job['metadata']['uid'], status='submitted')
                self.save()
            if job is None:
                exp['note'] = 'Job missing; explicit investigation required, no resubmission'
                self.decide(exp, [], failed=False)
                continue
            if exp.get('uid') and exp['uid'] != job['metadata']['uid']:
                exp.update(status='held', note='Job UID changed; refusing to adopt another workload')
                continue
            exp['uid'] = job['metadata']['uid']
            owned = [p for p in pods if any(o.get('uid') == exp['uid'] for o in p['metadata'].get('ownerReferences', []))]
            self.record_image(exp, owned)
            conditions = {c['type'] for c in job.get('status', {}).get('conditions', []) if c['status'] == 'True'}
            if 'Complete' in conditions:
                exp['status'] = 'completed'
                self.collect(exp)
            elif 'Failed' in conditions:
                self.decide(exp, owned, failed=True)
            else:
                log = self.kube.logs(exp['job'])
                for line in log.splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    step = value.get('steps') if isinstance(value, dict) else None
                    if isinstance(step, (int, float)) and step > exp.get('steps', -1):
                        exp.update(steps=step, progress_at=time.time())
                grace = self.cfg['stall_seconds'] if exp.get('steps', 0) > 0 else self.cfg['startup_grace_seconds']
                if time.time() - exp.get('progress_at', exp['submitted_at']) > grace:
                    self.decide(exp, owned, failed=False)
        allowed = self.data['experiments'] if self.data['validation']['status'] in ('fetched', 'skipped') else [self.data['validation']]
        busy = {o['uid'] for p in pods if p.get('status', {}).get('phase') in ('Running', 'Pending', 'Unknown')
                for o in p['metadata'].get('ownerReferences', []) if o.get('uid')}
        terminal = {j['metadata']['uid'] for j in jobs.values() if any(
            c['type'] in ('Failed', 'Complete') and c['status'] == 'True'
            for c in j.get('status', {}).get('conditions', []))}
        active = len(busy | {e.get('uid') or e['job'] for e in all_exp
            if e['status'] in ('submitting', 'submitted', 'needs_agent') and e.get('uid') not in terminal})
        limit = self.cfg['parallelism'] if allowed is self.data['experiments'] else 1
        for exp in allowed:
            if active >= limit:
                break
            if exp['status'] != 'queued':
                continue
            manifest = self.manifest(exp)
            exp.update(job=manifest['metadata']['name'], status='submitting', submitted_at=time.time())
            self.save()  # The deterministic identity is durable before the API call.
            existing = jobs.get(exp['job'])
            job = existing if existing is not None else self.kube.create(manifest)
            exp.update(uid=job['metadata']['uid'], status='submitted')
            active += 1
            self.save()
        self.save()

    @property
    def done(self):
        if self.data.get('startup_held'):
            return True
        return (self.data['validation']['status'] == 'held' or (
            self.data['validation']['status'] in ('fetched', 'skipped') and
            all(e['status'] in ('fetched', 'held') for e in self.data['experiments']))
        ) and all('completion_decision' in e or e['status'] != 'fetched'
                  for e in [self.data['validation']] + self.data['experiments'] if e.get('job'))
