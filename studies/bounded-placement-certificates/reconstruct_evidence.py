"""Restore exact scientific JSON evidence into a fresh directory; never run experiments."""
import argparse,gzip,hashlib,io,json,os
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent
RECORDS=[{'payload': 'evidence/all_cases.json.gz', 'restore_as': 'results/all_cases.json', 'compressed_sha256': 'fb3253e66fe27632f658c472953874347ccd4170ed0e28fd109bcfd9e686f1c6', 'compressed_bytes': 214264, 'restored_sha256': '6064b2564df31cccb2fbebd1fcd7e576a0b687385b7b8f47d3a2ce0652d1c5fb', 'restored_bytes': 6888985}, {'payload': 'evidence/first-grid-all-cases.json.gz', 'restore_as': 'historical/first-grid-all-cases.json', 'compressed_sha256': 'cccedb6d8b329426d6a5bfcbd9b7d1a2a7b3ef9897bf20a8240c5800588e9b02', 'compressed_bytes': 213974, 'restored_sha256': '9e73f27b7c22ce799c547ccc8430344a30edf59332ed265d771f06c160c88367', 'restored_bytes': 6887125}]
def reconstruct(out):
    out=Path(out)
    if os.path.lexists(out):raise FileExistsError('Output must be a fresh path')
    if not out.parent.is_dir():raise ValueError('Output parent must exist')
    ready=[]
    for row in RECORDS:
        payload=(PACKAGE/row['payload']).read_bytes()
        if len(payload)!=row['compressed_bytes'] or hashlib.sha256(payload).hexdigest()!=row['compressed_sha256']:raise ValueError('Compressed payload integrity failure')
        with gzip.GzipFile(fileobj=io.BytesIO(payload),mode='rb') as stream:data=stream.read(row['restored_bytes']+1)
        if len(data)!=row['restored_bytes'] or hashlib.sha256(data).hexdigest()!=row['restored_sha256']:raise ValueError('Restored evidence integrity failure')
        ready.append((row['restore_as'],data))
    out.mkdir()
    for relative,data in ready:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write(data)
    return [relative for relative,data in ready]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args();print(json.dumps({'restored':reconstruct(args.out),'scientific_computation_performed':False},indent=2))
