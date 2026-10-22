import json
from pathlib import Path
import tempfile
import unittest
from mesh.cached_evidence import Evidence


class EvidenceTests(unittest.TestCase):
    def test_existing_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):Evidence(directory)

    def test_failure_and_interrupt_retained(self):
        for count in (0,1):
            for error in (AssertionError('false prediction'),KeyboardInterrupt()):
                with tempfile.TemporaryDirectory() as directory:
                    evidence=Evidence(Path(directory)/'exclusive')
                    try:
                        for i in range(count):evidence.append({'case':i,'prediction':False})
                        raise error
                    except BaseException as caught:
                        evidence.finish('failed',error_type=type(caught).__name__)
                    self.assertEqual(len((evidence.path/'rows.jsonl').read_text().splitlines()),count)
                    self.assertEqual(json.loads((evidence.path/'terminal.json').read_text())['status'],'failed')
                    self.assertEqual(json.loads((evidence.path/'status.json').read_text())['status'],'running')

if __name__=='__main__':unittest.main()
