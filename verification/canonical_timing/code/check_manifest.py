from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent.parent
m=json.loads((p/'SHA256_MANIFEST.json').read_text())
for n,h in m.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
print(f'PASS: {len(m)} SHA-256 hashes')
