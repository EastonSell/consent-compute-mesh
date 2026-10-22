"""Independent retained-artifact checks, no inference or heavyweight imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from .cached_experiment import PREFIXES


def verify(directory):
    directory=Path(directory)
    terminal=json.loads((directory/'terminal.json').read_text())
    if terminal['status'] != 'complete' or terminal['rows'] != 36:
        raise ValueError('run not complete')
    rows=[json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
    if [(r['case'],r['step']) for r in rows] != [(c,s) for c in range(4) for s in range(9)]:
        raise ValueError('coverage mismatch')
    deterministic={}
    for row in rows:
        case,step=row['case'],row['step']
        if len(row['files']) != 2 or len(row['sha256']) != 2:raise ValueError('artifact count mismatch')
        arrays=[]
        for index,label in enumerate(('reference','split')):
            expected_name=f'case-{case}-step-{step}-{label}.f32'
            if row['files'][index] != expected_name:raise ValueError('artifact name mismatch')
            data=(directory/expected_name).read_bytes()
            if len(data) != 49152*4 or hashlib.sha256(data).hexdigest() != row['sha256'][index]:
                raise ValueError('artifact size/hash mismatch')
            values=[x[0] for x in struct.iter_unpack('<f',data)]
            if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite logits')
            token=max(range(len(values)),key=values.__getitem__)
            if token != row[label+'_token']:raise ValueError('greedy token mismatch')
            arrays.append(data)
            deterministic[expected_name]=hashlib.sha256(data).hexdigest()
        if arrays[0] != arrays[1] or not row['exact'] or not row['reset_replay_exact']:
            raise ValueError('nonexact retained result')
        expected_tokens=(PREFIXES+[PREFIXES[0]])[case] if step==0 else [rows[case*9+step-1]['reference_token']]
        if row['input_tokens'] != expected_tokens:raise ValueError('autoregressive input mismatch')
        if case==3 and row['sha256'] != rows[step]['sha256']:raise ValueError('reset replay mismatch')
    return deterministic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second')
    args=parser.parse_args()
    first,second=verify(args.first),verify(args.second)
    if first != second:raise ValueError('cross-run deterministic output mismatch')
    if (Path(args.first)/'source_hashes.json').read_bytes() != (Path(args.second)/'source_hashes.json').read_bytes():
        raise ValueError('cross-run source mismatch')
    print(json.dumps({'status':'verified','rows_per_run':36,'logit_files_per_run':72,'exact_reproduction':True},sort_keys=True))

if __name__=='__main__':main()
