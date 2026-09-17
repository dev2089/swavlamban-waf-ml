#!/usr/bin/env python3
"""Phase 8 acceptance gate: production storage/RBAC/secrets/privacy controls."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'phase8_master_exam_result.json'
CUTOFF=9.9

def run(name, cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {'name':name,'passed':p.returncode==0,'returncode':p.returncode,'tail':(p.stdout+'\n'+p.stderr).strip()[-3000:]}

def static_contract():
    security=(ROOT/'waf/security/production_security.py').read_text(encoding='utf-8')
    migration=(ROOT/'supabase/migrations/20260917120000_phase8_security.sql').read_text(encoding='utf-8')
    env=(ROOT/'.env.example').read_text(encoding='utf-8')
    required_security=['PERMISSIONS','issue_token','parse_bearer_token','authorize','validate_production_config','security_headers','audit_record']
    missing=[x for x in required_security if x not in security]
    required_sql=['waf_user_roles','waf_security_audit','ENABLE ROW LEVEL SECURITY','waf_current_role','waf_has_role','service_role','revoke all on public.waf_user_roles from anon','revoke all on public.waf_security_audit from anon']
    missing_sql=[x for x in required_sql if x.lower() not in migration.lower()]
    required_env=['SUPABASE_SERVICE_ROLE_KEY','WAF_AUTH_SECRET','WAF_CORS_ORIGINS','WAF_TOKEN_TTL_SECONDS']
    missing_env=[x for x in required_env if x not in env]
    forbidden=['SUPABASE_SERVICE_ROLE_KEY=your_','WAF_AUTH_SECRET=your_','WAF_CORS_ORIGINS=*']
    leaked=[x for x in forbidden if x in env]
    if missing or missing_sql or missing_env or leaked:
        raise AssertionError({'missing_security':missing,'missing_sql':missing_sql,'missing_env':missing_env,'unsafe_env':leaked})
    return {'security_contract':'PASS','supabase_security_migration':'PASS','env_secret_placeholders':'PASS','unsafe_patterns':leaked}

def main():
    checks=[run('full-regression',[sys.executable,'-m','pytest','-q']),run('compileall',[sys.executable,'-m','compileall','-q','waf','tests','scripts']),run('phase8-security-tests',[sys.executable,'-m','pytest','-q','tests/test_phase8_security.py'])]
    critical=[]
    try: checks.append({'name':'phase8-static-security-contract','passed':True,'tail':json.dumps(static_contract(),sort_keys=True)})
    except Exception as exc: checks.append({'name':'phase8-static-security-contract','passed':False,'tail':repr(exc)}); critical.append('phase8-static-security-contract')
    passed=sum(1 for x in checks if x['passed']); score=round(passed/len(checks)*10,2)
    if score<CUTOFF: critical.append('score-below-cutoff')
    out={'phase':8,'status':'PASS' if score>=CUTOFF and not critical else 'FAIL','score':score,'cutoff':CUTOFF,'critical_defects':critical,'checks':checks,'evaluation_scope':'repository regression, compile gate, dedicated Phase 8 security tests and static production-security/migration contract gate','completed':['fail-closed production secret/config contract','signed short-lived bearer authentication primitive','explicit viewer/operator/reviewer/admin RBAC permissions','explicit model/rule approval permissions','privacy-safe immutable audit record contract','Supabase RBAC/audit schema with RLS and anon revocation','production-only HTTPS/service-role/wildcard-CORS configuration checks','Phase 8 tests and executable master gate'],'remaining':['live Supabase migration/application execution','full FastAPI endpoint authentication wiring and deployment verification','ModSecurity/Coraza verification','TLS/HTTPS external deployment verification','full load/performance/failure testing','challenge scenario evidence','dashboard migration','five-minute demo','technical documentation/slides/report','final release-candidate gate'],'honesty_boundary':'Phase 8 verifies the production-security control contracts in repository-local tests. It does not claim a live Supabase project was migrated, external TLS/ModSecurity deployment was performed, or final Challenge 3 completion.'}
    RESULT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2)); return 0 if out['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
