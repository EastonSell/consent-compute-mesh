import os
import subprocess
import sys
import unittest
from mesh.cached_experiment import process_rss


class ResourceTests(unittest.TestCase):
    def test_controller_and_spawned_child_accounted(self):
        self.assertGreater(process_rss()[os.getpid()],0)
        child=subprocess.Popen([sys.executable,'-c',"import time; print('ready',flush=True); time.sleep(10)"],stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'ready')
            usage=process_rss()
            self.assertGreater(usage[child.pid],0)
            self.assertGreater(usage[os.getpid()],0)
        finally:
            child.terminate();child.wait(timeout=3);child.stdout.close()
