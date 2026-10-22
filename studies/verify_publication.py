"""Verify the integrated publication inventory without model dependencies."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PUBLICATION_SHA256.json').read_text())
for name, item in manifest['files'].items():
    path = root / name
    data = path.read_bytes()
    assert len(data) == item['bytes'], f'Length mismatch: {name}'
    assert hashlib.sha256(data).hexdigest() == item['sha256'], f'Hash mismatch: {name}'
print(f"Verified {len(manifest['files'])} files")
