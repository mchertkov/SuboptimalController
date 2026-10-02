from pathlib import Path
import hashlib,sys,json
root=Path(__file__).resolve().parent
ok=True
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
 h,rel=line.split('  ',1);p=root/rel;got=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'MISSING'
 if got!=h: print('FAIL',rel,got,h);ok=False
j=json.loads((root/'results/STAGE4C_COMPARISON.json').read_text())
w=j['stage4c_combined_all_independent_paths']['Wplus_pooled_normalizer']
if not (0.0248 < w < 0.0251): print('FAIL W range',w);ok=False
print('PASS' if ok else 'FAIL');sys.exit(0 if ok else 1)
