"""Mac entry point. Only Python's standard library, kubectl and Codex are needed."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cluster.control import (
    ROOT,
    Controller,
    codex_binary,
    job_manifest,
    load_config,
    write_json,
)
from cluster.kube import Kube


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This run already has an active controller') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def render(ctrl, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for exp in ctrl.data['experiments']:
        manifest = job_manifest(ctrl.cfg, ctrl.run_id, exp)
        write_json(directory / (manifest['metadata']['name'] + '.json'), manifest)


def status(ctrl):
    print('Run: {}  Updated: {}'.format(ctrl.run_id, ctrl.data.get('updated_at', 'not yet watched')))
    print('{:<28} {:<6} {:<5} {:<12} {:>12} {:>7}  {}'.format(
        'Task', 'Policy', 'Seed', 'Status', 'Steps', 'Attempt', 'Fetched'))
    for exp in [ctrl.data['validation']] + ctrl.data['experiments']:
        print('{:<28} {:<6} {:<5} {:<12} {:>12} {:>7}  {}'.format(
            exp['task'], exp['mode'], exp['seed'], exp['status'], exp.get('steps', '-'),
            exp['attempt'], exp.get('local', '-')))
        note = exp.get('transfer_error') or exp.get('note') or (exp.get('decision') or {}).get('reason')
        if note:
            print('  ' + note)
        if 'completion_decision' in exp:
            decision = exp['completion_decision']
            print('  Completion: ' + (decision['action'] + ': ' + decision['reason'] if decision
                  else 'Agent unavailable; artifacts were fetched. See agent/ logs.'))
    if ctrl.data.get('observer_error'):
        print('Monitor: ' + ctrl.data['observer_error'])
    if 'startup_decision' in ctrl.data:
        decision = ctrl.data['startup_decision']
        print('Startup: ' + (decision['reason'] if decision else 'Agent unavailable; submission waits. See agent/ logs.'))


def watch(ctrl):
    # Transfers and agent decisions may take minutes; continue polling while they run.
    with locked(ctrl.root), ThreadPoolExecutor(max_workers=6) as pool:
        ctrl.pool = pool
        previous = {}
        write_json(ctrl.root / 'watcher.json', {'pid': os.getpid(), 'started_at': time.time()})
        awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        try:
            while True:
                started = time.monotonic()
                try:
                    ctrl.tick()
                    ctrl.data.pop('observer_error', None)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    ctrl.data['observer_error'] = str(error)
                    print('Monitor will retry: ' + str(error), flush=True)
                ctrl.data['updated_at'] = datetime.now(timezone.utc).isoformat()
                ctrl.save()
                if 'startup_decision' in ctrl.data and ('startup' not in previous or
                        previous['startup'] != ctrl.data['startup_decision']):
                    decision = ctrl.data['startup_decision']
                    print('Startup agent: ' + (json.dumps(decision) if decision
                          else 'unavailable; submission waits while the monitor retries. See agent/ logs.'), flush=True)
                    previous['startup'] = decision
                for exp in [ctrl.data['validation']] + ctrl.data['experiments']:
                    identity = (exp['task'], exp['mode'], exp['seed'])
                    event = {k: exp.get(k) for k in ('task', 'mode', 'seed', 'status', 'attempt',
                             'decision', 'completion_decision', 'note', 'transfer_error', 'local')}
                    if event != previous.get(identity):
                        print(json.dumps(event), flush=True)
                        previous[identity] = event
                if ctrl.done:
                    break
                time.sleep(max(1, ctrl.cfg['poll_seconds'] - (time.monotonic() - started)))
        except KeyboardInterrupt:
            print('Monitor stopped. Cluster jobs continue; use watch to resume.', flush=True)
        finally:
            awake.terminate()
            awake.wait()
        status(ctrl)


def prerequisites(cfg):
    command = codex_binary(cfg['codex_binary'])
    for executable in ('kubectl', command, 'caffeinate'):
        if not shutil.which(executable):
            raise RuntimeError('Missing Mac command: ' + executable)
    subprocess.run([command, 'login', 'status'], check=True, timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'render', 'login'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, help='JSON overrides for cluster/config.json')
        if name != 'login':
            command.add_argument('--image', help='Registry image tag or digest')
            command.add_argument('--run-id')
        if name == 'run':
            command.add_argument('--skip-validation', action='store_true',
                                 help='Explicitly waive GPU validation; requires an immutable image digest')
        if name == 'render':
            command.add_argument('--output', type=Path, default=ROOT / 'runs/cluster/render')
    for name in ('status', 'watch', 'fetch'):
        command = commands.add_parser(name)
        command.add_argument('run_id')
    args = parser.parse_args(argv)
    if args.command in ('run', 'render', 'login'):
        cfg = load_config(args.config)
        if args.command == 'login':
            kube = Kube(cfg)
            print(json.dumps(kube.preflight(), indent=2))
            print('kubectl --context {} -n {} exec -it {} -- bash'.format(
                cfg['context'], cfg['namespace'], kube.helper))
            return
        cfg['image'] = args.image or cfg['image']
        run_id = args.run_id or ('preview' if args.command == 'render' else
            datetime.now(timezone.utc).strftime('rl-%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
        root = ROOT / 'runs/cluster' / run_id
        if args.command == 'render':
            cfg['image'] = cfg['image'] or 'IMAGE_REQUIRED'
            ctrl = Controller(cfg, run_id, Path(args.output) / '.preview')
            render(ctrl, args.output)
            print('{} manifests: {}'.format(len(ctrl.data['experiments']), args.output.resolve()))
            if cfg['image'] == 'IMAGE_REQUIRED':
                print('Preview only. Set --image before submitting.')
            return
        prerequisites(cfg)
        if not cfg['image']:
            raise ValueError('Set --image to a built and pushed cluster/Dockerfile image')
        if args.skip_validation and '@sha256:' not in cfg['image']:
            raise ValueError('--skip-validation requires an immutable image digest')
        with locked(root):
            if (root / 'run.json').exists():
                raise ValueError('Run already exists; use watch ' + run_id)
            ctrl = Controller(cfg, run_id, root)
            if args.skip_validation:
                ctrl.cfg['resolved_image'] = cfg['image']
                ctrl.data['validation'].update(status='skipped', note='GPU validation explicitly waived by user')
            ctrl.data['preflight'] = ctrl.kube.preflight()
            ctrl.data['created_at'] = datetime.now(timezone.utc).isoformat()
            ctrl.data['source_revision'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            ctrl.save()
            render(ctrl, root / 'manifests')
        with (root / 'monitor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'cluster', 'watch', run_id],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        write_json(root / 'launcher.json', {'pid': process.pid})
        print('Run: {}\nMonitor: {}\nStatus: python3 -m cluster status {}'.format(
            run_id, root / 'monitor.log', run_id))
        return
    root = ROOT / 'runs/cluster' / args.run_id
    if not (root / 'run.json').is_file():
        raise ValueError('Unknown run: ' + args.run_id)
    ctrl = Controller(None, args.run_id, root)
    if args.command == 'status':
        status(ctrl)
    elif args.command == 'watch':
        watch(ctrl)
    else:
        with locked(root):
            ctrl.fetch()
        status(ctrl)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
