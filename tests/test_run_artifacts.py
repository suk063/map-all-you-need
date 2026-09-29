"""Portable run bookkeeping; intentionally needs only the Python standard library."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class RunArtifactsTest(unittest.TestCase):
    def test_checkpoint_selection_skips_temporary_and_broken_directories(self):
        from benchmark.common.policy import latest_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('9', '100', '200', '300.orbax-checkpoint-tmp', '.400'):
                (root / 'checkpoints' / name).mkdir(parents=True)
            seen = []
            def load(path):
                seen.append(path.name)
                if path.name == '200':
                    raise ValueError('interrupted checkpoint')
                return ('normalizer', 'actor', 'critic')
            with self.assertWarnsRegex(RuntimeWarning, 'Skipped invalid checkpoints'):
                path, params = latest_checkpoint(root, load=load)
            self.assertEqual(path.name, '100')
            self.assertEqual(params, ('normalizer', 'actor', 'critic'))
            self.assertEqual(seen, ['200', '100'])

    def test_no_valid_checkpoint_fails_with_useful_error(self):
        from benchmark.common.policy import latest_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No valid Brax checkpoint'):
                latest_checkpoint(directory, load=lambda p: None)
            (Path(directory) / 'checkpoints' / '100').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'No valid Brax checkpoint'):
                latest_checkpoint(directory, load=lambda p: ('actor-only',))

    def test_map_policy_keeps_checkpoint_path_after_loading_feature_caches(self):
        from benchmark.common import policy
        saved = {'format_version': 2, 'env_config': {'obs_mode': 'map', 'map': {'cache': '/old'}},
                 'map_cache_paths': ['/old/a.h5', '/old/b.h5'], 'observation_size': {'xyz': [6], 'feature_ids': [2]},
                 'action_size': 3, 'ppo': {'network_factory': {}, 'normalize_observations': False}}
        inference = object()
        modules = {'jax': SimpleNamespace(jit=lambda fn: fn),
                   'brax.training': SimpleNamespace(types=SimpleNamespace(identity_observation_preprocessor=object())),
                   'brax.training.acme': SimpleNamespace(running_statistics=SimpleNamespace(normalize=object())),
                   'brax.training.agents.ppo': SimpleNamespace(networks=SimpleNamespace(
                       make_inference_fn=lambda _: lambda *a, **kw: inference)),
                   'benchmark.common.mapping': SimpleNamespace(FeatureBank=lambda: SimpleNamespace(add=lambda _: None))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'config.json').write_text(json.dumps(saved))
            checkpoint = root / 'checkpoints' / '100'
            with patch.dict('sys.modules', modules), patch.object(policy, 'latest_checkpoint', return_value=(checkpoint, (1, 2, 3))), \
                    patch.object(policy, 'network_factory', return_value=lambda *a, **kw: object()):
                restored = policy.load_policy(root)
            self.assertEqual(restored.checkpoint, checkpoint)
            self.assertIs(restored.inference, inference)

    def test_portable_bundle_requires_every_cache_and_does_not_mutate_saved_metadata(self):
        from benchmark.common.policy import resolve_map_caches
        saved = {'env_config': {'obs_mode': 'map', 'map': {'cache': '/old/cache', 'dino': {'weights': '/missing'}}},
                 'map_cache_paths': ['/old/cache/a.h5', '/old/cache/b.h5']}
        original = copy.deepcopy(saved)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.assertEqual(resolve_map_caches(root, saved), saved)
            bundle = root / 'map-cache'
            bundle.mkdir()
            (bundle / 'a.h5').touch()
            with self.assertRaisesRegex(ValueError, 'Incomplete bundled map cache'):
                resolve_map_caches(root, saved)
            (bundle / 'b.h5').touch()
            resolved = resolve_map_caches(root, saved)
            self.assertEqual(resolved['map_cache_paths'], [str(bundle / 'a.h5'), str(bundle / 'b.h5')])
            self.assertEqual(resolved['env_config']['map']['cache'], str(bundle))
            self.assertTrue(resolved['env_config']['map']['cache_read_only'])
            self.assertEqual(saved, original)

    def test_resume_budget_counts_saved_steps_across_attempts_only(self):
        from benchmark.rl.resume import resume_budget
        self.assertEqual(resume_budget({'ppo': {'num_timesteps': 1000}}, Path('300')), (1000, 300, 700))
        second = {'target_timesteps': 1000, 'resume_base_steps': 300, 'ppo': {'num_timesteps': 700}}
        self.assertEqual(resume_budget(second, Path('200')), (1000, 500, 500))
        self.assertEqual(resume_budget(second, Path('750')), (1000, 1050, 0))

    def test_resume_rejects_explicit_identity_and_map_overrides(self):
        from argparse import Namespace

        from benchmark.rl.resume import validate_resume_overrides
        saved = {'seed': 3, 'target_timesteps': 1000,
                 'env_config': {'env_id': 'PandaPickCube', 'obs_mode': 'map', 'environment': {'impl': 'warp'},
                                'map': {'robot': 'full', 'background': False, 'views': 96, 'extra_views': 512,
                                        'cache': '/old/cache', 'dino': {'source': '/old/dino', 'weights': '/old/weights'}}}}
        validate_resume_overrides(Namespace(), saved)
        validate_resume_overrides(Namespace(env_id='PandaPickCube', seed=3, map_robot='full'), saved)
        for values in ({'seed': 2}, {'env_id': 'AlohaHandOver'}, {'obs_mode': 'rgb'}, {'impl': 'jax'},
                       {'map_robot': 'gripper'}, {'map_views': 4}, {'total_timesteps': 1200}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'incompatible'):
                validate_resume_overrides(Namespace(**values), saved)

    def test_checkpoint_cadence_does_not_multiply_large_native_batches(self):
        from benchmark.rl.resume import checkpoint_schedule
        params = {'num_timesteps': 1800000000, 'batch_size': 512, 'unroll_length': 100,
                  'num_minibatches': 32, 'action_repeat': 4, 'num_resets_per_eval': 1}
        epochs, interval = checkpoint_schedule(params, 1000000)
        quantum = 512 * 100 * 32 * 4
        self.assertGreater(epochs, 1)
        self.assertLess((epochs - 1) * interval - params['num_timesteps'], quantum)
        self.assertEqual(interval % quantum, 0)
        with self.assertRaisesRegex(ValueError, 'positive'):
            checkpoint_schedule(params, 0)

    def test_prime_number_of_rgb_batches_stays_near_requested_checkpoint_interval(self):
        from benchmark.rl.resume import checkpoint_schedule
        params = {'num_timesteps': 5000000, 'batch_size': 16, 'unroll_length': 10,
                  'num_minibatches': 8, 'action_repeat': 1, 'num_resets_per_eval': 1}
        epochs, interval = checkpoint_schedule(params, 1000000)
        self.assertLessEqual(epochs, 7)
        self.assertGreaterEqual(interval, 800000)
        self.assertLessEqual(interval, 1200000)
        self.assertLess((epochs - 1) * interval - params['num_timesteps'], interval)

    def test_checkpoint_rounding_bound_matches_brax_epoch_formula(self):
        from benchmark.rl.resume import checkpoint_schedule
        for target in range(1, 500, 7):
            for quantum in (1, 2, 5, 13, 64, 1000):
                for requested in (1, 7, 50, 1000000):
                    params = {'num_timesteps': target, 'batch_size': quantum, 'unroll_length': 1,
                              'num_minibatches': 1, 'action_repeat': 1, 'num_resets_per_eval': 1}
                    num_evals, interval = checkpoint_schedule(params, requested)
                    epochs = num_evals - 1
                    self.assertEqual(interval, math.ceil(target / (epochs * quantum)) * quantum)
                    self.assertGreaterEqual(epochs * interval, target)
                    self.assertLess(epochs * interval - target, interval)


if __name__ == '__main__':
    unittest.main()
