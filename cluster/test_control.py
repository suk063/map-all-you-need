"""Controller contracts run on a Mac without any learning dependencies."""

import copy
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from cluster import control as c


class FakeKube:
    def __init__(self):
        self.items, self.created = [], []
        self.fail_snapshot = False
        self.lose_create_reply = False

    def snapshot(self, run_id):
        if self.fail_snapshot:
            raise RuntimeError('API unavailable')
        return copy.deepcopy(self.items)

    def create(self, manifest):
        job = copy.deepcopy(manifest)
        job['metadata']['uid'] = 'uid-' + job['metadata']['name']
        job['status'] = {'active': 1}
        self.items.append(job)
        self.created.append(job)
        if self.lose_create_reply:
            self.lose_create_reply = False
            raise RuntimeError('lost reply after server accepted job')
        return job

    def logs(self, name):
        return ''

    def events(self, name):
        return []

    def fetch(self, remote, local):
        local.mkdir(parents=True, exist_ok=True)
        (local / 'complete.json').write_text(json.dumps({'files': {}}))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = c.load_config()
        self.cfg.update(gpu_auto=False, parallelism=4)
        self.cfg['image'] = 'example/rl@sha256:' + 'a' * 64
        self.kube = FakeKube()
        self.agent = lambda *args: {'action': 'retry', 'reason': 'node lost',
                                    'exclude_node': False, 'increase_memory': False}

    def tearDown(self):
        self.tmp.cleanup()

    def controller(self):
        return c.Controller(self.cfg, 'test-run', self.root, self.kube, self.agent)

    def validated(self):
        ctrl = self.controller()
        ctrl.data['validation']['status'] = 'fetched'
        ctrl.data['started'] = True
        ctrl.save()
        return ctrl

    def test_thirty_distinct_manifests(self):
        ctrl = self.controller()
        manifests = [c.job_manifest(self.cfg, 'test-run', e) for e in ctrl.data['experiments']]
        self.assertEqual(len(manifests), 30)
        self.assertEqual(len({m['metadata']['name'] for m in manifests}), 30)
        for manifest in manifests:
            self.assertEqual(manifest['spec']['backoffLimit'], 0)
            self.assertEqual(manifest['spec']['podReplacementPolicy'], 'Failed')
            pod = manifest['spec']['template']['spec']
            self.assertEqual(pod['restartPolicy'], 'Never')
            resources = pod['containers'][0]['resources']
            self.assertEqual(resources['requests'], resources['limits'])
            self.assertEqual(resources['limits']['nvidia.com/rtxa6000'], 1)

    def test_validation_gates_training(self):
        ctrl = self.controller()
        ctrl.tick()
        self.assertEqual(len(self.kube.created), 1)
        self.assertEqual(self.kube.created[0]['metadata']['labels']['mayn-kind'], 'validation')

    def test_startup_hold_stops_without_repeated_agent_calls(self):
        ctrl = self.controller()
        calls = []
        def hold(*args):
            calls.append(args)
            return {'action': 'hold', 'reason': 'configuration needs correction',
                    'exclude_node': False, 'increase_memory': False}
        ctrl.agent = hold
        ctrl.tick()
        ctrl.tick()
        self.assertTrue(ctrl.done)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.kube.created)

    def test_concurrency_and_restart_do_not_duplicate_jobs(self):
        ctrl = self.validated()
        ctrl.tick()
        self.assertEqual(len(self.kube.created), 4)
        self.controller().tick()
        self.assertEqual(len(self.kube.created), 4)

    def test_lost_submission_reply_adopts_existing_job(self):
        ctrl = self.validated()
        self.kube.lose_create_reply = True
        with self.assertRaises(RuntimeError):
            ctrl.tick()
        self.controller().tick()
        self.assertEqual(len(self.kube.created), 4)
        self.assertEqual(len({m['metadata']['name'] for m in self.kube.created}), 4)

    def test_api_failure_never_resubmits(self):
        ctrl = self.validated()
        ctrl.tick()
        before = copy.deepcopy(ctrl.data)
        self.kube.fail_snapshot = True
        with self.assertRaises(RuntimeError):
            ctrl.tick()
        self.assertEqual(ctrl.data, before)
        self.assertEqual(len(self.kube.created), 4)

    def test_submission_failure_before_acceptance_retries_same_identity(self):
        ctrl = self.validated()
        create = self.kube.create
        self.kube.create = lambda manifest: (_ for _ in ()).throw(RuntimeError('offline'))
        with self.assertRaises(RuntimeError):
            ctrl.tick()
        name = ctrl.data['experiments'][0]['job']
        self.kube.create = create
        self.controller().tick()
        self.assertEqual(len(self.kube.created), 4)
        self.assertEqual(self.kube.created[0]['metadata']['name'], name)

    def test_live_or_missing_pod_does_not_authorize_duplicate(self):
        ctrl = self.validated()
        ctrl.tick()
        self.kube.items[0]['status'] = {'failed': 1}  # No terminal Job condition.
        ctrl.tick()
        self.assertEqual(len(self.kube.created), 4)

    def test_failed_job_gets_fresh_bounded_attempt(self):
        ctrl = self.validated()
        ctrl.tick()
        exp = ctrl.data['experiments'][0]
        old_path = c.remote_path(self.cfg, 'test-run', exp)
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        ctrl.tick()
        self.assertEqual(exp['attempt'], 1)
        self.assertIn(old_path, exp['previous'])
        self.assertNotEqual(exp['job'], self.kube.created[0]['metadata']['name'])
        exp['attempt'] = self.cfg['max_retries']
        active = next(j for j in self.kube.items if j['metadata']['name'] == exp['job'])
        active['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        ctrl.tick()
        self.assertEqual(exp['status'], 'held')

    def test_oom_recovery_caps_ram_and_excludes_only_failed_node(self):
        ctrl = self.validated()
        ctrl.tick()
        exp = ctrl.data['experiments'][0]
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        self.kube.items.append({'kind': 'Pod', 'metadata': {'name': 'old-pod',
            'ownerReferences': [{'uid': exp['uid']}]}, 'spec': {'nodeName': 'failed-node'},
            'status': {'phase': 'Failed', 'containerStatuses': [
                {'state': {'terminated': {'reason': 'OOMKilled'}}}]}})
        ctrl.agent = lambda *args: {'action': 'retry', 'reason': 'host RAM OOM',
                                   'exclude_node': True, 'increase_memory': True}
        ctrl.tick()
        self.assertEqual(exp['memory_multiplier'], 2)
        self.assertEqual(exp['excluded_nodes'], ['failed-node'])
        memory = self.kube.created[-1]['spec']['template']['spec']['containers'][0]['resources']['limits']['memory']
        self.assertEqual(memory, '64Gi')

    def test_terminal_job_with_live_pod_is_not_replaced(self):
        ctrl = self.validated()
        ctrl.tick()
        exp = ctrl.data['experiments'][0]
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        self.kube.items.append({'kind': 'Pod', 'metadata': {'name': 'old-pod',
            'ownerReferences': [{'uid': exp['uid']}]}, 'status': {'phase': 'Running'}})
        ctrl.tick()
        self.assertEqual(exp['attempt'], 0)
        self.assertEqual(len(self.kube.created), 4)

    def test_fetch_failure_does_not_restart_training(self):
        ctrl = self.validated()
        ctrl.tick()
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Complete', 'status': 'True'}]}
        def fail(*args):
            raise RuntimeError('transfer interrupted')
        self.kube.fetch = fail
        ctrl.tick()
        exp = ctrl.data['experiments'][0]
        self.assertEqual(exp['status'], 'completed')
        self.assertEqual(exp['attempt'], 0)
        self.assertEqual(len(self.kube.created), 5)  # One newly available slot.

    def test_fetch_discovers_newly_completed_jobs_without_submitting(self):
        ctrl = self.validated()
        ctrl.tick()
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Complete', 'status': 'True'}]}
        ctrl.agent = lambda *args: self.fail('fetch must not invoke an agent')
        ctrl.fetch()
        self.assertEqual(ctrl.data['experiments'][0]['status'], 'fetched')
        self.assertEqual(len(self.kube.created), 4)

    def test_agent_unavailable_holds_recovery(self):
        ctrl = self.validated()
        ctrl.tick()
        ctrl.agent = lambda *args: None
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Failed', 'status': 'True'}]}
        ctrl.tick()
        self.assertEqual(ctrl.data['experiments'][0]['attempt'], 0)
        self.assertEqual(ctrl.data['experiments'][0]['status'], 'needs_agent')
        self.assertEqual(len(self.kube.created), 5)  # Recovery waits; unused GPU slot serves queued work.

    def test_malformed_agent_output_does_not_stop_monitoring(self):
        def invalid_output(command, **kwargs):
            Path(command[command.index('-o') + 1]).write_text('null')
        with patch('cluster.control.subprocess.run', side_effect=invalid_output):
            self.assertIsNone(c.diagnose('failure', {}, self.root, command='codex'))
        errors = list((self.root / 'agent').glob('*/error.txt'))
        self.assertEqual(len(errors), 1)
        self.assertIn('Invalid agent decision', errors[0].read_text())

    def test_slow_download_keeps_polling_and_fills_free_slot(self):
        ctrl = self.validated()
        ctrl.tick()
        future = Future()
        class Pool:
            def submit(self, *args):
                return future
        ctrl.pool = Pool()
        self.kube.items[0]['status'] = {'conditions': [{'type': 'Complete', 'status': 'True'}]}
        ctrl.tick()
        self.assertEqual(ctrl.data['experiments'][0]['status'], 'completed')
        self.assertEqual(len(self.kube.created), 5)
        ctrl.tick()
        self.assertEqual(len(self.kube.created), 5)
        future.set_exception(RuntimeError('interrupted transfer'))
        ctrl.tick()
        self.assertIn('interrupted', ctrl.data['experiments'][0]['transfer_error'])

    def test_receipt_verifies_contents_and_rejects_escape(self):
        (self.root / 'weights').write_bytes(b'abc')
        receipt = {'files': {'weights': hashlib.sha256(b'abc').hexdigest()}}
        (self.root / 'complete.json').write_text(json.dumps(receipt))
        c.verify_download(self.root)
        (self.root / 'weights').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            c.verify_download(self.root)
        receipt['files'] = {'../outside': 'bad'}
        (self.root / 'complete.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            c.verify_download(self.root)


if __name__ == '__main__':
    unittest.main()
