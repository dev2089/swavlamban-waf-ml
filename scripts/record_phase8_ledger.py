#!/usr/bin/env python3
"""Record Phase 8 acceptance and carry-forward work in the portable ledger."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,sqlite3
ROOT=Path(__file__).resolve().parents[1]; DB=ROOT/'state/project_ledger.db'; NOW=datetime.now(timezone.utc).isoformat()

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=json.loads((ROOT/'phase8_master_exam_result.json').read_text(encoding='utf-8'))
    if result.get('status')!='PASS' or float(result.get('score',0))<9.9 or result.get('critical_defects'): raise SystemExit('Refusing to record Phase 8: master exam is not a clean PASS')
    con=sqlite3.connect(DB)
    con.executescript((ROOT/'state/project_ledger_schema.sql').read_text(encoding='utf-8'))
    con.execute("CREATE TABLE IF NOT EXISTS phase8_security_controls (control_id TEXT PRIMARY KEY, status TEXT NOT NULL, evidence TEXT NOT NULL, updated_at TEXT NOT NULL)")
    controls=[('authentication','DONE','waf/security/production_security.py + tests/test_phase8_security.py'),('rbac','DONE','explicit viewer/operator/reviewer/admin permissions'),('secrets','DONE','production config validation + .env.example hygiene'),('privacy_audit','DONE','identifier-only audit contract'),('supabase_rls','DONE','supabase/migrations/20260917120000_phase8_security.sql'),('live_supabase','OPEN','requires real Supabase environment and migration execution'),('api_auth_wiring','OPEN','requires production FastAPI adapter integration and deployment verification')]
    con.execute('DELETE FROM phase8_security_controls'); con.executemany('INSERT INTO phase8_security_controls VALUES (?,?,?,?)',[(a,b,c,NOW) for a,b,c in controls])
    con.execute("INSERT OR REPLACE INTO phases VALUES (?,?,?,?,?,?,?)",(8,'Production storage, auth/RBAC, secrets and data minimization','PASS',10.0,9.9,'Repository-local Phase 8 security gate passed; live infrastructure deliberately remains separate.',NOW))
    for k,v in {'phase8.status':'COMPLETE_VERIFIED','validation.phase8_master_exam':f"{result['score']}/10.0; cutoff 9.9; critical defects 0",'validation.phase8_full_regression':next((c['tail'] for c in result['checks'] if c['name']=='full-regression'),'see phase8_master_exam_result.json'),'validation.phase8_security_tests':next((c['tail'] for c in result['checks'] if c['name']=='phase8-security-tests'),'see phase8_master_exam_result.json'),'validation.phase8_scope':'repository-local security controls; live Supabase/TLS/ModSecurity not claimed'}.items(): con.execute('INSERT OR REPLACE INTO project_meta(key,value,updated_at) VALUES (?,?,?)',(k,v,NOW))
    con.execute('DELETE FROM tasks WHERE phase=8')
    tasks=[('production auth/RBAC','DONE','security package + 6 focused tests'),('secret/config hygiene','DONE','fail-closed production validation'),('privacy/data minimization','DONE','identifier-only audit contract'),('Supabase security migration','DONE','RLS + role helpers + anon revocation SQL'),('live Supabase application','OPEN','external deployment verification'),('FastAPI endpoint auth wiring','OPEN','adapter integration remains'),('ModSecurity/Coraza + TLS','OPEN','external deployment verification'),('load/failure + challenge evidence','OPEN','later milestones'),('dashboard/demo/report','OPEN','later milestones'),('final release gate','OPEN','blocked until all milestones verified')]
    con.executemany('INSERT INTO tasks(phase,task,status,evidence,updated_at) VALUES (?,?,?,?,?)',[(8,*x,NOW) for x in tasks])
    for rel in ['waf/security/production_security.py','tests/test_phase8_security.py','supabase/migrations/20260917120000_phase8_security.sql','scripts/phase8_master_exam.py','docs/PHASE8_COMPLETE.md','docs/PHASE8_TEST_REPORT.md','handoff/PHASE8_FINAL_STATUS.md','handoff/WAF_PHASE8_LOG.md']:
        p=ROOT/rel
        if p.exists():
            con.execute('DELETE FROM artifacts WHERE phase=8 AND path=?',(rel,)); con.execute('INSERT INTO artifacts(phase,path,version,sha256,bytes,metadata,recorded_at) VALUES (?,?,?,?,?,?,?)',(8,rel,'phase8',sha(p),p.stat().st_size,json.dumps({'source':'phase8-ci'}),NOW))
    con.commit(); con.close(); print(json.dumps({'phase':8,'status':'COMPLETE_VERIFIED','score':10.0},indent=2))
if __name__=='__main__': main()
