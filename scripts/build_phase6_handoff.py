#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'WAF_PHASE6_FINAL_HANDOFF.zip'; EXCLUDE={'.git','.pytest_cache','__pycache__','node_modules'}
def files():
    for p in ROOT.rglob('*'):
        if p.is_file():
            r=p.relative_to(ROOT)
            if not any(x in EXCLUDE for x in r.parts) and r.name!=OUT.name: yield p,r
result=json.loads((ROOT/'phase6_master_exam_result.json').read_text())
if result.get('status')!='PASS' or result.get('score',0)<9.9 or result.get('critical_defects'): raise SystemExit('handoff refused: Phase 6 is not a clean PASS')
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p,r in sorted(files(),key=lambda x:str(x[1])): z.write(p,r.as_posix())
print(json.dumps({'archive':OUT.name,'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'files':sum(1 for _ in files())},indent=2))
