#!/usr/bin/env python3
"""Build a portable Phase 9 handoff archive from the verified local checkout."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT.parent/'WAF_PHASE9_FINAL_HANDOFF.zip'; EXCLUDE_DIRS={'.git','__pycache__','.pytest_cache','node_modules'}; EXCLUDE_NAMES={'WAF_PHASE5_FINAL_HANDOFF.zip','WAF_PHASE6_FINAL_HANDOFF.zip','WAF_PHASE7_FINAL_HANDOFF.zip','WAF_PHASE8_FINAL_HANDOFF.zip','WAF_PHASE9_FINAL_HANDOFF.zip'}
def main():
 result=json.loads((ROOT/'phase9_master_exam_result.json').read_text(encoding='utf-8'))
 if result['status']!='PASS' or result['score']<9.9 or result['critical_defects']: raise SystemExit('refusing to package a non-PASS Phase 9 checkout')
 if not (ROOT/'state/project_ledger.db').exists() or not (ROOT/'state/phase9_ledger.sql').exists(): raise SystemExit('portable ledger is missing')
 if OUT.exists(): OUT.unlink()
 count=0
 with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
  for path in sorted(ROOT.rglob('*')):
   if path.is_dir() or any(part in EXCLUDE_DIRS for part in path.parts) or path.name in EXCLUDE_NAMES: continue
   zf.write(path,path.relative_to(ROOT).as_posix()); count+=1
 print(json.dumps({'archive':str(OUT),'files':count,'bytes':OUT.stat().st_size,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
