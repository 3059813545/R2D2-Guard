"""Verify SHA-256 transport integrity, using Python's standard library only."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
failures = []
for relative, expected in manifest['files'].items():
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(relative)
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        failures.append(relative)
if failures:
    raise SystemExit('Integrity check failed for: ' + ', '.join(failures))
print('SHA-256 integrity passed for', len(manifest['files']), 'files.')
