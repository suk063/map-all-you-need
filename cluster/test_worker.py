import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster.worker import bundle, execute, select_resume, training_command


class WorkerTests(unittest.TestCase):
    def test_validation_isolates_cases_in_fresh_processes(self):
        cases = ['tests/test_integration.py::test_native_rgb[PandaPickCube]',
                 'tests/test_integration.py::test_real_checkpoint_continuation[map]']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'validation'
            spec = {'kind': 'validation', 'output': str(output), 'config': {}}
            collected = subprocess.CompletedProcess([], 0, '\n'.join(cases) + '\n2 tests collected', '')
            with patch.dict('os.environ'), patch('cluster.worker.subprocess.run', return_value=collected), patch('cluster.worker.run_command') as run:
                execute(spec)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            for command, case in zip(commands, cases):
                self.assertIn(case, command)
                self.assertIn('integration', command)
                self.assertNotIn('--collect-only', command)
            self.assertTrue((output / 'complete.json').is_file())

    def test_empty_validation_cannot_unlock_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'validation'
            collected = subprocess.CompletedProcess([], 0, 'no tests collected', '')
            with patch.dict('os.environ'), patch('cluster.worker.subprocess.run', return_value=collected), self.assertRaisesRegex(RuntimeError, 'No GPU validation'):
                execute({'kind': 'validation', 'output': str(output), 'config': {}})
            self.assertFalse((output / 'complete.json').exists())

    def test_eviction_during_metadata_write_has_nothing_to_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"incomplete":')
            self.assertIsNone(select_resume([str(root)], {'task': 'T', 'mode': 'state', 'seed': 0},
                latest=lambda _: self.fail('No checkpoint was created')))

    def test_map_cache_isolated_across_runs_and_seeds(self):
        paths = set()
        for run_id, seed in [('run-a', 0), ('run-a', 1), ('run-b', 0)]:
            args = training_command({'task': 'PandaPickCube', 'mode': 'map', 'seed': seed,
                'run_id': run_id, 'output': '/mnt/new', 'config': {'checkpoint_steps': 1000000,
                'train_args': [], 'remote_root': '/mnt/project'}}, None)
            paths.add(args[args.index('--map-cache') + 1])
        self.assertEqual(len(paths), 3)

    def test_resume_command_does_not_reapply_map_defaults(self):
        spec = {'task': 'PandaPickCube', 'mode': 'map', 'seed': 0, 'output': '/mnt/new',
                'config': {'checkpoint_steps': 1000000, 'train_args': [], 'remote_root': '/mnt/project'}}
        args = training_command(spec, Path('/mnt/old'))
        self.assertIn('--resume', args)
        self.assertNotIn('--map-cache', args)
        self.assertNotIn('--total-timesteps', args)

    def test_selects_most_advanced_durable_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, offset in [('attempt-0', 0), ('attempt-1', 100)]:
                path = root / name
                path.mkdir()
                (path / 'config.json').write_text(json.dumps({'seed': 0, 'resume_base_steps': offset,
                    'env_config': {'env_id': 'T', 'obs_mode': 'state'}}))
                (path / 'checkpoints/50').mkdir(parents=True)
            latest = lambda path: (path / 'checkpoints/50', [1, 2, 3])
            selected = select_resume([str(root / 'attempt-1'), str(root / 'attempt-0')],
                                     {'task': 'T', 'mode': 'state', 'seed': 0}, latest)
            self.assertEqual(selected, root / 'attempt-1')

    def test_bundles_map_cache_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'external/hash.h5'
            cache.parent.mkdir()
            cache.write_bytes(b'features')
            run = root / 'run'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps({'map_cache_paths': [str(cache)]}))
            bundle(run, {'kind': 'train'})
            self.assertEqual((run / 'map-cache/hash.h5').read_bytes(), b'features')
            receipt = json.loads((run / 'complete.json').read_text())
            self.assertEqual(receipt['files']['map-cache/hash.h5'], hashlib.sha256(b'features').hexdigest())
            self.assertNotIn('complete.json', receipt['files'])

    def test_completed_checkpoint_retries_only_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / 'attempt-0', root / 'attempt-1'
            checkpoint = source / 'checkpoints/100'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'weights').write_bytes(b'parameters')
            (source / 'config.json').write_text(json.dumps({'seed': 0, 'target_timesteps': 100,
                'ppo': {'num_timesteps': 100}, 'env_config': {'env_id': 'T', 'obs_mode': 'state'}}))
            spec = {'task': 'T', 'mode': 'state', 'seed': 0, 'kind': 'train', 'output': str(destination),
                    'previous': [str(source)], 'config': {'eval_episodes': 100}}
            with patch('benchmark.common.policy.latest_checkpoint', side_effect=lambda p: (Path(p) / 'checkpoints/100', [1, 2, 3])), patch('cluster.worker.run_command') as run:
                execute(spec)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[2], 'evaluate')
            self.assertEqual((destination / 'checkpoints/100/weights').read_bytes(), b'parameters')
            self.assertTrue((destination / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
