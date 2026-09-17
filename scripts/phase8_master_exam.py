from __future__ import annotations
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); RESULT=ROOT/'phase8_master_exam_result.json'; CUTOFF=9.9
def run(name,cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); return {'name':name,'passed':p.returncode==0,'returncode':p.returncode,'tail':(p.stdout+'\n'+p.stderr).strip()[-2500:]}
def security_smoke():
 from datetime import datetime,timedelta,timezone
 from waf.core.config import WAFConfig
 from waf.security import Action,SecretProvider,TokenCodec,authorize,can
 from waf.storage import SQLiteSecurityStore,StoragePrivacyError
 import tempfile
 codec=TokenCodec('x'*48,clock=lambda:100); token=codec.issue('release-human','model_approver',60,now=100); actor=codec.verify(token,now=120); assert actor.role=='model_approver' and can(actor,Action.PROMOTE_MODEL) and not can(actor,Action.DEPLOY_RULE); authorize(actor,Action.PROMOTE_MODEL); assert SecretProvider({'WAF_AUTH_SECRET':'s'*48}).get('WAF_AUTH_SECRET',required=True)=='s'*48
 with tempfile.TemporaryDirectory() as td:
  store=SQLiteSecurityStore(Path(td)/'runtime.db'); event={'event_type':'waf.decision','schema_version':'event-v2','request_id':'smoke-1','decision':'allow','risk_score':.1,'occurred_at':datetime.now(timezone.utc).isoformat(),'evidence':{'privacy':{'raw_payload_retained':False,'raw_headers_retained':False,'raw_query_retained':False}}}; assert store.publish(event,retention_days=1)==1
  try: store.publish({**event,'payload':'raw'},retention_days=1); raise AssertionError('raw payload was persisted')
  except StoragePrivacyError: pass
  assert store.count('waf_decision_events')==1 and store.purge_expired(datetime.now(timezone.utc)+timedelta(days=2))['decision_events']==1; store.close()
 return {'token_role':actor.role,'model_promotion':True,'rule_deploy_boundary':True,'secret_provider':'PASS','persistent_store':'PASS','raw_material_rejected':True,'retention':'PASS'}
def config_smoke():
 import os; old=os.environ.copy()
 try:
  os.environ.update({'WAF_ENV':'production','WAF_STORAGE_BACKEND':'sqlite','WAF_AUTH_REQUIRED':'true'}); cfg=WAFConfig.from_env(); assert cfg.environment=='production' and cfg.storage_backend=='sqlite' and cfg.auth_required; os.environ['WAF_STORAGE_BACKEND']='memory'
  try: WAFConfig.from_env(); raise AssertionError('production allowed volatile memory storage')
  except ValueError: pass
  return {'production_requires_persistence':True,'production_requires_auth':True}
 finally: os.environ.clear(); os.environ.update(old)
def migration_smoke():
 text=(ROOT/'supabase/migrations/20260917130000_phase8_security_hardening.sql').read_text(encoding='utf-8'); required=['waf_actor_role','app_metadata','waf_runtime_events','waf_security_audit','ENABLE ROW LEVEL SECURITY','service_role','model_approver','rule_approver','waf_purge_expired_runtime_events','raw_payload_retained','raw_headers_retained','raw_query_retained']; missing=[x for x in required if x not in text]; assert not missing; assert 'TO anon' not in text; assert 'REVOKE ALL ON TABLE public.%I FROM anon, authenticated' in text; return {'required_markers':len(required),'missing':missing,'anon_policies_in_new_migration':False,'legacy_raw_tables_sealed':True}
def source_scan():
 text='\n'.join(p.read_text(encoding='utf-8') for p in [ROOT/'app.py',ROOT/'waf/security/auth.py',ROOT/'waf/security/secrets.py',ROOT/'waf/storage/persistent.py']); assert 'your-secret-key-here' not in text; assert not re.search(r'(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)',text); return {'hardcoded_secret_patterns':False}
def main():
 checks=[run('phase8-focused-tests',[sys.executable,'-m','pytest','-q','tests/test_phase8_security_hardening.py']),run('full-regression',[sys.executable,'-m','pytest','-q']),run('compileall',[sys.executable,'-m','compileall','-q','waf','tests','scripts']),run('phase7-focused-regression',[sys.executable,'-m','pytest','-q','tests/test_phase7_learning_control.py'])]; critical=[c['name'] for c in checks if not c['passed']]
 for name,fn in [('security-smoke',security_smoke),('production-config-smoke',config_smoke),('supabase-migration-contract',migration_smoke),('secret-source-static-gate',source_scan)]:
  try: checks.append({'name':name,'passed':True,'returncode':0,'tail':json.dumps(fn(),sort_keys=True)})
  except Exception as exc: checks.append({'name':name,'passed':False,'returncode':1,'tail':repr(exc)}); critical.append(name)
 passed=sum(1 for c in checks if c['passed']); score=round(10.0*passed/len(checks),2)
 if score<CUTOFF: critical.append('score-below-cutoff')
 result={'phase':8,'status':'PASS' if score>=CUTOFF and not critical else 'FAIL','score':score,'cutoff':CUTOFF,'critical_defects':critical,'checks':checks,'acceptance':{'production_storage':'durable SQLite reference adapter + Postgres/Supabase migration contract','authentication':'signed short-lived Actor token contract; identity-provider claim bridge remains external','rbac':'viewer, analyst, rule_approver, model_approver, admin','approval_permissions':'rule deployment/rollback and model promotion/rollback are role-gated; Phase 7 human approval remains mandatory','secrets':'environment-only provider with minimum-length and placeholder rejection; no secret values committed','data_minimization':'persistent runtime/audit paths reject raw HTTP material; legacy raw tables are sealed from anon/authenticated PostgREST roles','retention':'runtime decision and security audit records carry expiry and an admin/service-role purge path'},'honesty_boundary':'The Supabase migration is statically contract-tested and the local durable adapter is executable, but the migration is not claimed live-applied. Hosted identity/RBAC, live Supabase deployment, TLS, ModSecurity/Coraza, load/failure testing and final challenge evidence remain later milestones.'}; RESULT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2)); return 0 if result['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
