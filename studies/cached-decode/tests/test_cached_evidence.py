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

class SubprocessRetentionTests(unittest.TestCase):
    def test_actual_interrupt_before_and_after_row(self):
        import subprocess
        import sys
        import signal
        code='''
from mesh.cached_evidence import Evidence
import signal,sys
e=Evidence(sys.argv[1])
try:
    if sys.argv[2]=='1': e.append({'case':0,'prediction':False})
    print('ready',flush=True)
    signal.pause()
except BaseException as error:
    e.finish('failed',error_type=type(error).__name__)
    raise
'''
        for count in (0,1):
            with tempfile.TemporaryDirectory() as directory:
                output=Path(directory)/'run'
                process=subprocess.Popen([sys.executable,'-c',code,str(output),str(count)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                try:
                    self.assertEqual(process.stdout.readline().strip(),'ready')
                    process.send_signal(signal.SIGINT)
                    process.communicate(timeout=5)
                    self.assertNotEqual(process.returncode,0)
                    self.assertEqual(len((output/'rows.jsonl').read_text().splitlines()),count)
                    self.assertEqual(json.loads((output/'terminal.json').read_text())['error_type'],'KeyboardInterrupt')
                finally:
                    if process.poll() is None:process.kill();process.communicate()
