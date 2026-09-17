from __future__ import annotations
from datetime import datetime,timedelta,timezone
import re
from pathlib import Path
import pytest
from waf.core.config import WAFConfig
from waf.security import Action,Actor,AuthError,SecretError,SecretProvider,TokenCodec,authorize,can
from waf.storage import SQLiteSecurityStore,StoragePrivacyError
ROOT=Path(__file__).resolve().parents[1]; MIGRATION=ROOT/'supabase/migrations/20260917130000_phase8_security_hardening.sql'
def test_token_round_trip_tamper_and_expiry():
 codec=TokenCodec('x'*48,clock=lambda:100); token=codec.issue('human-1','model_approver',ttl_seconds=60); actor=codec.verify(token,now=120); assert actor.actor_id=='human-1' and actor.role=='model_approver'
 with pytest.raises(AuthError): codec.verify(token[:-1]+('a' if token[-1]!='a' else 'b'),now=120)
 with pytest.raises(AuthError): codec.verify(token,now=160)
def test_rbac_explicit_approval_boundaries():
 viewer=Actor('v','viewer',1,2,'t'); model=Actor('m','model_approver',1,2,'t'); rule=Actor('r','rule_approver',1,2,'t'); admin=Actor('a','admin',1,2,'t'); assert not can(viewer,Action.PROMOTE_MODEL); assert can(model,Action.PROMOTE_MODEL); assert can(rule,Action.DEPLOY_RULE); assert not can(model,Action.DEPLOY_RULE); assert authorize(admin,Action.MANAGE_SECURITY) is admin
 with pytest.raises(PermissionError): authorize(viewer,Action.ROLLBACK_MODEL)
def test_production_config_requires_persistent_store_and_auth(monkeypatch):
 monkeypatch.setenv('WAF_ENV','production'); monkeypatch.setenv('WAF_STORAGE_BACKEND','sqlite'); monkeypatch.setenv('WAF_AUTH_REQUIRED','true'); cfg=WAFConfig.from_env(); assert cfg.environment=='production' and cfg.storage_backend=='sqlite' and cfg.auth_required is True; monkeypatch.setenv('WAF_STORAGE_BACKEND','memory')
 with pytest.raises(ValueError): WAFConfig.from_env()
def test_secret_provider_rejects_missing_weak_and_placeholder():
 with pytest.raises(SecretError): SecretProvider({}).get('WAF_AUTH_SECRET',required=True)
 with pytest.raises(SecretError): SecretProvider({'WAF_AUTH_SECRET':'short'}).get('WAF_AUTH_SECRET',required=True)
 with pytest.raises(SecretError): SecretProvider({'WAF_AUTH_SECRET':'change-me'}).get('WAF_AUTH_SECRET',required=True)
 assert SecretProvider({'WAF_AUTH_SECRET':'s'*48}).get('WAF_AUTH_SECRET',required=True)=='s'*48
def test_persistent_store_minimizes_data_and_expires_records(tmp_path):
 store=SQLiteSecurityStore(tmp_path/'runtime.db'); event={'event_type':'waf.decision','schema_version':'event-v2','request_id':'req-1','decision':'block','risk_score':1.0,'occurred_at':datetime.now(timezone.utc).isoformat(),'evidence':{'privacy':{'raw_payload_retained':False,'raw_headers_retained':False,'raw_query_retained':False},'feature_snapshot':{'path_length':0.4}}}; assert store.publish(event,retention_days=1)==1
 with pytest.raises(StoragePrivacyError): store.publish({**event,'query':'secret=1'},retention_days=1)
 with pytest.raises(StoragePrivacyError): store.audit('a','admin','x','y','z','ok','reason',{'source_ip':'10.0.0.1'})
 assert store.audit('a','admin','manage_retention','runtime','req-1','success','retention test',retention_days=1)==1; future=datetime.now(timezone.utc)+timedelta(days=2); assert store.purge_expired(future)=={'decision_events':1,'audit_records':1}; assert store.count('waf_decision_events')==0 and store.count('waf_security_audit')==0; store.close()
def test_phase8_migration_has_rls_roles_secrets_and_legacy_seal():
 text=MIGRATION.read_text(encoding='utf-8'); required=['waf_actor_role','app_metadata','waf_runtime_events','waf_security_audit','ENABLE ROW LEVEL SECURITY','REVOKE ALL ON TABLE public.%I FROM anon, authenticated','waf_purge_expired_runtime_events','service_role','raw_payload_retained','raw_headers_retained','raw_query_retained','phase7_model_events_insert','model_approver','rule_approver']; [pytest.fail(marker) for marker in required if marker not in text]; assert 'TO anon' not in text
def test_no_hardcoded_runtime_secrets_in_authoritative_security_surfaces():
 text='\n'.join(p.read_text(encoding='utf-8') for p in [ROOT/'app.py',ROOT/'waf/security/auth.py',ROOT/'waf/security/secrets.py',ROOT/'.env.example']); assert 'your-secret-key-here' not in text; assert not re.search(r'(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)',text); assert 'SUPABASE_SERVICE_ROLE_KEY=<runtime secret' in (ROOT/'.env.example').read_text(encoding='utf-8')
