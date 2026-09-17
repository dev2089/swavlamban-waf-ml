#!/usr/bin/env python3
"""Build a portable Phase 8 future-chat handoff after a clean CI gate."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'WAF_PHASE8_FINAL_HANDOFF.zip'; EXCLUDE={'.git','.pytest_cache','__pycache__','node_modules'}
def files():
    for p in ROOT.rglob('*'):
        if p.is_file():
            r=p.relative_to(ROOT)
            if not any(x in EXCLUDE for x in r.parts) and r.name not in {'WAF_PHASE8_FINAL_HANDOFF.zip','WAF_PHASE7_FINAL_HANDOFF.zip','WAF_PHASE6_FINAL_HANDOFF.zip'}: yield p,r
def main():
    result=json.loads((ROOT/'phase8_master_exam_result.json').read_text(encoding='utf-8'))
    if result.get('status')!='PASS' or float(result.get('score',0))<9.9 or result.get('critical_defects'): raise SystemExit('Phase 8 handoff refused: clean gate required')
    if not (ROOT/'state/project_ledger.db').exists() or not (ROOT/'state/phase8_ledger.sql').exists(): raise SystemExit('Phase 8 handoff requires generated SQLite and SQL ledger')
    required=['waf/security/production_security.py','tests/test_phase8_security.py','supabase/migrations/20260917120000_phase8_security.sql','scripts/phase8_master_exam.py','docs/PHASE8_COMPLETE.md','handoff/PHASE8_FINAL_STATUS.md','handoff/WAF_PHASE8_LOG.md']
    missing=[x for x in required if not (ROOT/x).exists()]
    if missing: raise SystemExit(f'missing required evidence: {missing}')
    with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
        for p,r in sorted(files(),key=lambda x:str(x[1])): z.write(p,r.as_posix())
    print(json.dumps({'archive':OUT.name,'files':sum(1 for _ in files()),'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'phase8_score':result['score'],'critical_defects':result['critical_defects'],'ledger_included':True},indent=2))
if __name__=='__main__': main()
