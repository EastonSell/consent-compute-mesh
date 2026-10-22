"""Exclusive, incrementally durable experiment evidence; no model dependency."""
import hashlib
import json
import os
from pathlib import Path


class Evidence:
    def __init__(self, output):
        self.path = Path(output)
        self.path.mkdir(parents=False, exist_ok=False)
        self.write('status.json', {'status':'running'})
        self.rows = (self.path/'rows.jsonl').open('x')

    def write(self, name, value):
        with (self.path/name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

    def append(self, row):
        self.rows.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        self.rows.flush(); os.fsync(self.rows.fileno())

    def finish(self, status, **details):
        self.rows.close()
        self.write('terminal.json', dict(status=status, **details))


def source_hashes(root):
    root=Path(root)
    paths=sorted((root/'mesh').glob('*.py'))+sorted((root/'tests').glob('*.py'))
    paths += [root/'CACHED_EXPERIMENT.md']
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
