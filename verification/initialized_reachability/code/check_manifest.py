from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent.parent
manifest=json.loads((p/'SHA256_MANIFEST.json').read_text())
for name,expected in manifest.items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print(f'PASS: {len(manifest)} SHA-256 hashes')
