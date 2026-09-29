"""One Job: validate the image, or train/resume, evaluate, and publish artifacts."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from benchmark.common.geometry import validate_geometry
from cluster.control import write_json


def select_resume(paths, spec, latest=None):
    if latest is None:
        from benchmark.common.policy import latest_checkpoint
        latest = latest_checkpoint
    candidates = []
    for value in paths:
        root = Path(value)
        if not (root / 'config.json').is_file() or not (root / 'checkpoints').is_dir():
            continue  # A failure during imports/map creation has no parameters yet.
        meta = json.loads((root / 'config.json').read_text())
        if (meta['env_config']['env_id'], meta['env_config']['obs_mode'], meta['seed']) != (
                spec['task'], spec['mode'], spec['seed']):
            raise ValueError('Resume attempt belongs to a different experiment')
        validate_geometry(meta['env_config'])
        try:
            checkpoint, _ = latest(root)
        except ValueError:
            continue
        candidates.append((meta.get('resume_base_steps', 0) + int(checkpoint.name), root))
    return max(candidates, key=lambda c: c[0])[1] if candidates else None


def training_command(spec, resume):
    cfg = spec['config']
    command = [sys.executable, '-m', 'benchmark.rl.train', '--env-id', spec['task'],
               '--obs-mode', spec['mode'], '--seed', str(spec['seed']), '--output', spec['output'],
               '--checkpoint-steps', str(cfg['checkpoint_steps']), '--no-run-evals']
    if resume:
        command += ['--resume', str(resume)]
    elif spec['mode'] == 'map':
        command += ['--map-cache', str(Path(cfg['remote_root']) / spec['run_id'] / 'cache' /
                                      spec['task'] / ('seed' + str(spec['seed'])))]
    return command + cfg['train_args']


def copy_file(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoints/cache are immutable; hard links avoid copying them on the same PVC.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def bundle(output, spec):
    output = Path(output)
    if (output / 'config.json').exists():
        meta = json.loads((output / 'config.json').read_text())
        for path in meta.get('map_cache_paths', []):
            destination = output / 'map-cache' / Path(path).name
            if not destination.exists():
                copy_file(path, destination)
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'complete.json':
            if path.is_symlink():
                raise ValueError('Artifact must be a regular file: ' + str(path))
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            files[str(path.relative_to(output))] = digest.hexdigest()
    write_json(output / 'complete.json', {'kind': spec['kind'], 'finished_at': time.time(), 'files': files})


def run_command(command, output, stage, log):
    status = {'event': 'heartbeat', 'phase': stage, 'steps': 0}
    status_path = output.parent / (output.name + '.status.json')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)

    def consume():
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
            try:
                value = json.loads(line)
                if isinstance(value, dict) and isinstance(value.get('steps'), (int, float)):
                    status['steps'] = max(status['steps'], value['steps'])
            except ValueError:
                pass

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    while True:
        status.update(at=time.time(), pid=process.pid)
        write_json(status_path, status)
        print(json.dumps(status), flush=True)
        try:
            code = process.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            pass
    reader.join()
    process.stdout.close()
    if code:
        raise RuntimeError(f'{stage} failed with exit code {code}')


def execute(spec):
    cfg, output = spec['config'], Path(spec['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').is_file():
        from cluster.control import verify_download
        verify_download(output)
        return
    log_path = output.parent / (output.name + '.log')
    with log_path.open('a', buffering=1) as log:
        print(json.dumps({'event': 'start', 'kind': spec['kind'], 'output': str(output)}), flush=True)
        if spec['kind'] == 'evaluate':
            output.mkdir(exist_ok=True)
            run_command([sys.executable, '-m', 'benchmark.eval', '--checkpoint', spec['checkpoint'],
                         '--output', str(output / 'eval-seed10000'), '--episodes', str(cfg['eval_episodes']),
                         '--num-envs', '8', '--seed', '10000'], output, 'evaluate', log)
        elif spec['kind'] == 'validation':
            output.mkdir(exist_ok=True)
            os.environ['RGB_VALIDATION_OUTPUT'] = str(output / 'rgb-views')
            command = [sys.executable, '-m', 'pytest', '-q', '-m', 'integration']
            collected = subprocess.run(command + ['--collect-only', 'tests/test_integration.py'], capture_output=True,
                                      text=True, timeout=120, check=False)
            log.write(collected.stdout + collected.stderr)
            if collected.returncode:
                print(collected.stdout + collected.stderr, flush=True)
                collected.check_returncode()
            cases = [line for line in collected.stdout.splitlines()
                     if line.startswith('tests/test_integration.py::')]
            if not cases:
                raise RuntimeError('No GPU validation cases were collected')
            # CUDA allocations from Warp/JAX/PyTorch accumulate across a long
            # pytest session. A fresh process per case releases them completely.
            for number, case in enumerate(cases, 1):
                run_command(command + [case, '--junitxml=' + str(output / f'tests-{number:03}.xml')],
                            output, f'validation {number}/{len(cases)}: {case}', log)
                print(json.dumps({'event': 'validated', 'steps': number, 'cases': len(cases)}), flush=True)
        else:
            from benchmark.common.policy import latest_checkpoint
            from benchmark.rl.resume import resume_budget
            source = select_resume(spec['previous'], spec)
            trained = False
            if source:
                meta = json.loads((source / 'config.json').read_text())
                checkpoint, _ = latest_checkpoint(source)
                _, _, remaining = resume_budget(meta, checkpoint)
                trained = remaining == 0
            if trained:
                # Evaluation/bundling failed after durable training completion.
                output.mkdir(exist_ok=False)
                shutil.copy2(source / 'config.json', output / 'config.json')
                if (source / 'train.csv').exists():
                    shutil.copy2(source / 'train.csv', output / 'train.csv')
                shutil.copytree(checkpoint, output / 'checkpoints' / checkpoint.name, copy_function=copy_file)
            else:
                run_command(training_command(spec, source), output, 'train', log)
            checkpoint, _ = latest_checkpoint(output)
            meta = json.loads((output / 'config.json').read_text())
            target, steps, remaining = resume_budget(meta, checkpoint)
            if remaining:
                raise RuntimeError('Training exited before the target was durably checkpointed')
            write_json(output / 'trained.json', {'target_timesteps': target, 'saved_steps': steps})
            print(json.dumps({'event': 'trained', 'steps': steps}), flush=True)
            run_command([sys.executable, '-m', 'benchmark.eval', '--checkpoint', str(output),
                         '--episodes', str(cfg['eval_episodes']), '--num-envs', '8', '--seed', '10000'],
                        output, 'evaluate', log)
        write_json(output / 'cluster.json', spec)
        revision = Path('/app/code-revision.txt')
        if revision.exists():
            shutil.copy2(revision, output / revision.name)
    shutil.copy2(log_path, output / 'worker.log')
    bundle(output, spec)
    print(json.dumps({'event': 'complete', 'output': str(output)}), flush=True)


def main():
    spec = json.loads(os.environ['RUN_SPEC'])
    try:
        execute(spec)
    except Exception:
        output = Path(spec['output'])
        write_json(output.parent / (output.name + '.failure.json'), {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
