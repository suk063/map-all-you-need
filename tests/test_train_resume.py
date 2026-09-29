"""Exercise the real train orchestration without importing GPU libraries."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from benchmark.common.policy import latest_checkpoint


@contextmanager
def trainer():
    env = SimpleNamespace(action_size=3, observation_size=4)
    config = {'env_id': 'PandaPickCubeCartesian', 'obs_mode': 'state', 'environment': {'impl': 'warp'}}
    params = {'num_timesteps': 1000, 'num_envs': 2, 'num_eval_envs': 1, 'batch_size': 2,
              'num_minibatches': 1, 'unroll_length': 2, 'action_repeat': 1, 'episode_length': 4,
              'num_evals': 2, 'num_resets_per_eval': 1, 'normalize_observations': True,
              'network_factory': {'policy_hidden_layer_sizes': [32]}}
    envs = SimpleNamespace(DEFAULT_TASK=config['env_id'], OBS_MODES=('state', 'rgb', 'map'),
                           TASKS=(config['env_id'], 'PandaPickCube'), env_config=lambda *a: config.copy(),
                           make_env=Mock(return_value=env), ppo_config=lambda _: params.copy())
    ppo = SimpleNamespace(train=Mock())
    modules = {name: ModuleType(name) for name in ('brax', 'brax.training', 'brax.training.agents',
                                                  'brax.training.agents.ppo', 'mujoco_playground',
                                                  'mujoco_playground._src')}
    modules['brax.training.agents.ppo'].train = ppo
    modules['mujoco_playground._src'].wrapper = SimpleNamespace(wrap_for_brax_training=object())
    modules.update({'jax': SimpleNamespace(devices=lambda: ['test-device']), 'benchmark.common.envs': envs})
    path = Path(__file__).parents[1] / 'benchmark/rl/train.py'
    spec = importlib.util.spec_from_file_location('_benchmark_train_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    module.version = lambda _: 'test'
    module.distribution = lambda _: SimpleNamespace(read_text=lambda _: '{}')
    module.network_factory = lambda *a: 'test-network'
    module.latest_checkpoint = lambda root: latest_checkpoint(root, load=lambda path: ('normalizer', 'actor', 'critic'))
    yield module, envs, ppo


class TrainResumeTest(unittest.TestCase):
    def test_chain_uses_checkpoints_and_restores_actor_critic_normalizer(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, _envs, ppo):
            root = Path(directory).resolve()
            first = module.train(module.parser().parse_args(['--seed', '3', '--no-run-evals', '--output', str(root / 'first')]))
            (first / 'checkpoints' / '300').mkdir(parents=True)
            (first / 'train.csv').write_text('steps,metric,value\n999999,training/loss,1\n')
            second = module.train(module.parser().parse_args(['--resume', str(first), '--no-run-evals', '--output', str(root / 'second')]))
            saved = json.loads((second / 'config.json').read_text())
            self.assertEqual((saved['target_timesteps'], saved['resume_base_steps'], saved['ppo']['num_timesteps']), (1000, 300, 700))
            self.assertFalse(saved['resume']['optimizer_restored'])
            self.assertFalse(saved['resume']['rng_restored'])
            self.assertEqual(ppo.train.call_args.kwargs['restore_params'], ('normalizer', 'actor', 'critic'))
            self.assertTrue(ppo.train.call_args.kwargs['restore_value_fn'])
            self.assertEqual(ppo.train.call_args.kwargs['seed'], 3)
            (second / 'checkpoints' / '200').mkdir(parents=True)
            third = module.train(module.parser().parse_args(['--resume', str(second), '--no-run-evals', '--output', str(root / 'third')]))
            saved = json.loads((third / 'config.json').read_text())
            self.assertEqual((saved['resume_base_steps'], saved['ppo']['num_timesteps']), (500, 500))
            self.assertEqual(saved['ppo']['network_factory'], {'policy_hidden_layer_sizes': [32]})
            ppo.train.call_args.kwargs['progress_fn'](20, {'loss': 1.0})
            self.assertIn('520,loss,1.0', (third / 'train.csv').read_text())

    def test_completed_run_returns_source_without_creating_output_or_env(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, envs, ppo):
            root = Path(directory).resolve()
            first = module.train(module.parser().parse_args(['--no-run-evals', '--output', str(root / 'first')]))
            (first / 'checkpoints' / '1024').mkdir(parents=True)
            envs.make_env.reset_mock()
            ppo.train.reset_mock()
            output = root / 'not-created'
            result = module.train(module.parser().parse_args(['--resume', str(first), '--output', str(output)]))
            self.assertEqual(result, first)
            self.assertFalse(output.exists())
            envs.make_env.assert_not_called()
            ppo.train.assert_not_called()

    def test_incompatible_explicit_override_fails_before_new_run(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, _envs, ppo):
            root = Path(directory).resolve()
            first = module.train(module.parser().parse_args(['--no-run-evals', '--output', str(root / 'first')]))
            (first / 'checkpoints' / '300').mkdir(parents=True)
            ppo.train.reset_mock()
            for option in (['--seed', '2'], ['--obs-mode', 'rgb'], ['--env-id', 'PandaPickCube'], ['--total-timesteps', '2000']):
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'incompatible'):
                    module.train(module.parser().parse_args(['--resume', str(first), *option]))
            ppo.train.assert_not_called()

    def test_checkpoint_epochs_can_save_without_running_evaluations(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, _envs, ppo):
            result = module.train(module.parser().parse_args(['--no-run-evals', '--checkpoint-steps', '100', '--output', directory + '/run']))
            saved = json.loads((result / 'config.json').read_text())
            self.assertEqual(saved['checkpoint_steps'], 100)
            self.assertGreater(saved['ppo']['num_evals'], 1)
            self.assertFalse(ppo.train.call_args.kwargs['run_evals'])
            self.assertIsNone(ppo.train.call_args.kwargs['eval_env'])
            self.assertFalse(ppo.train.call_args.kwargs['vision'])

    def test_disabled_rgb_evaluations_still_get_matching_renderer_batch(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, envs, ppo):
            module.train(module.parser().parse_args(['--obs-mode', 'rgb', '--no-run-evals', '--output', directory + '/run']))
            self.assertEqual([call.args[1] for call in envs.make_env.call_args_list], [2, 1])
            self.assertFalse(ppo.train.call_args.kwargs['run_evals'])
            self.assertIsNotNone(ppo.train.call_args.kwargs['eval_env'])

    def test_normalization_override_is_recorded_and_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, trainer() as (module, _envs, ppo):
            first = module.train(module.parser().parse_args([
                '--no-normalize-observations', '--no-run-evals', '--output', directory + '/first']))
            saved = json.loads((first / 'config.json').read_text())
            self.assertFalse(saved['ppo']['normalize_observations'])
            self.assertFalse(ppo.train.call_args.kwargs['normalize_observations'])
            (first / 'checkpoints' / '300').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                module.train(module.parser().parse_args(['--resume', str(first), '--normalize-observations']))
            with self.assertRaisesRegex(ValueError, 'Map observations cannot be normalized'):
                module.train(module.parser().parse_args(['--obs-mode', 'map', '--normalize-observations']))


if __name__ == '__main__':
    unittest.main()
