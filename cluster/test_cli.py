import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster.__main__ import locked, main


class CliTests(unittest.TestCase):
    def test_lock_excludes_second_controller(self):
        with tempfile.TemporaryDirectory() as directory, locked(Path(directory)), self.assertRaises(RuntimeError), locked(Path(directory)):
            pass

    def test_render_is_offline_and_writes_thirty_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('subprocess.run', side_effect=AssertionError('must be offline')):
                main(['render', '--output', directory])
            files = list(Path(directory).glob('*.json'))
            self.assertEqual(len(files), 30)
            self.assertTrue(all(json.loads(f.read_text())['kind'] == 'Job' for f in files))


if __name__ == '__main__':
    unittest.main()
