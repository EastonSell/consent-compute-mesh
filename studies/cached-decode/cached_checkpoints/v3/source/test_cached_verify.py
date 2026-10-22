import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from mesh.cached_verify import verify
from mesh.cached_experiment import PREFIXES


class VerifyTests(unittest.TestCase):
    def test_complete_and_corrupt_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'terminal.json').write_text(json.dumps({'status':'complete','rows':36}))
            data=b'\0'*(49152*4); digest=hashlib.sha256(data).hexdigest();rows=[]
            for case,prefix in enumerate(PREFIXES+[PREFIXES[0]]):
                for step in range(9):
                    names=[f'case-{case}-step-{step}-{label}.f32' for label in ('reference','split')]
                    for name in names:(path/name).write_bytes(data)
                    rows.append(dict(case=case,step=step,files=names,sha256=[digest,digest],
                                     reference_token=0,split_token=0,exact=True,reset_replay_exact=True,
                                     input_tokens=prefix if step==0 else [0]))
            (path/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            self.assertEqual(len(verify(path)),72)
            (path/rows[-1]['files'][0]).write_bytes(b'bad')
            with self.assertRaises(ValueError):verify(path)
