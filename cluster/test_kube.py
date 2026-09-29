import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster.control import load_config
from cluster.kube import Kube, safe_extract


class ArchiveTests(unittest.TestCase):
    def archive(self, name, symlink=False):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as archive:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type, member.linkname = tarfile.SYMTYPE, '/tmp/outside'
            else:
                member.size = 3
            archive.addfile(member, None if symlink else io.BytesIO(b'abc'))
        data.seek(0)
        return data

    def test_extract_regular_artifact(self):
        with tempfile.TemporaryDirectory() as path:
            with tarfile.open(fileobj=self.archive('./checkpoints/100/weights')) as archive:
                safe_extract(archive, Path(path))
            self.assertEqual((Path(path) / 'checkpoints/100/weights').read_bytes(), b'abc')

    def test_rejects_escape_and_links(self):
        for name, link in [('../escape', False), ('/tmp/escape', False), ('weights', True)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as path, tarfile.open(fileobj=self.archive(name, link)) as archive, self.assertRaises(ValueError):
                safe_extract(archive, Path(path))

    def test_interrupted_and_corrupt_transfers_retry_atomically(self):
        payload = io.BytesIO()
        receipt = json.dumps({'files': {'weights': hashlib.sha256(b'abc').hexdigest()}}).encode()
        with tarfile.open(fileobj=payload, mode='w') as archive:
            for name, content in [('weights', b'abc'), ('complete.json', receipt)]:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        kube = Kube(load_config())
        kube.login = lambda: 'test-login'
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'attempt-0'
            for data, code in [(b'partial', 1), (b'corrupt', 0), (payload.getvalue(), 0)]:
                def fake(command, data=data, code=code, **kwargs):
                    self.assertIn('kubectl', command)
                    kwargs['stdout'].write(data)
                    return subprocess.CompletedProcess(command, code, stderr=b'interrupted')
                with patch('subprocess.run', side_effect=fake):
                    if data != payload.getvalue():
                        with self.assertRaises(RuntimeError):
                            kube.fetch('/mnt/map-all-you-need/test/run', destination)
                        self.assertFalse(destination.exists())
                    else:
                        kube.fetch('/mnt/map-all-you-need/test/run', destination)
                        self.assertEqual((destination / 'weights').read_bytes(), b'abc')


if __name__ == '__main__':
    unittest.main()
