from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib, json, re
from threading import RLock
from typing import Any, Mapping

RULE_SCHEMA_VERSION='rule-v1'
DEPLOYMENT_SCHEMA_VERSION='deployment-v1'

class RuleStatus(str,Enum):
    GENERATED='generated'; VALIDATED='validated'; PENDING_APPROVAL='pending_approval'; APPROVED='approved'; REJECTED='rejected'; DEPLOYED='deployed'; ROLLED_BACK='rolled_back'

@dataclass(frozen=True,slots=True)
class RuleValidation:
    valid: bool; errors: tuple[str,...]=(); warnings: tuple[str,...]=()

@dataclass(slots=True)
class RuleRecord:
    rule_id:str; name:str; description:str; matcher_type:str; matcher:Mapping[str,Any]; action:str; confidence:float; source:str; source_request_id:str; source_detector:str; source_rule_ids:tuple[str,...]=(); schema_version:str=RULE_SCHEMA_VERSION; version:int=1; status:RuleStatus=RuleStatus.GENERATED; validation:RuleValidation|None=None; approved_by:str|None=None; approved_at:str|None=None; deployed_at:str|None=None; created_at:str=''; updated_at:str=''
    def __post_init__(self):
        n=_now(); self.created_at=self.created_at or n; self.updated_at=self.updated_at or n; self.source_rule_ids=tuple(self.source_rule_ids)
        if not 0<=self.confidence<=1: raise ValueError('confidence must be in [0, 1]')
        if self.schema_version!=RULE_SCHEMA_VERSION: raise ValueError('unsupported rule schema')
    def canonical(self):
        return {'rule_id':self.rule_id,'name':self.name,'description':self.description,'matcher_type':self.matcher_type,'matcher':dict(sorted(self.matcher.items())),'action':self.action,'confidence':round(float(self.confidence),6),'source':self.source,'source_request_id':self.source_request_id,'source_detector':self.source_detector,'source_rule_ids':sorted(self.source_rule_ids),'schema_version':self.schema_version,'version':self.version}
    def to_dict(self):
        p=asdict(self); p['status']=self.status.value
        if self.validation: p['validation']=asdict(self.validation)
        return p

@dataclass(frozen=True,slots=True)
class _Deployment:
    deployment_id:str; revision:int; active_rule_ids:tuple[str,...]; ruleset_sha256:str; deployed_at:str; status:str; rollback_of:str|None=None

FEATURE_RULES={'has_sql_keyword':('SQL-like signature','WAF-ML-SQL'),'has_xss_token':('XSS-like signature','WAF-ML-XSS'),'has_traversal':('path traversal-like signature','WAF-ML-TRAV'),'has_command_token':('command-injection-like signature','WAF-ML-CMD'),'double_encoded_flag':('double-encoding obfuscation','WAF-ML-DOUBLE-ENCODE'),'null_byte_flag':('null-byte obfuscation','WAF-ML-NULL-BYTE'),'malformed_percent_flag':('malformed percent encoding','WAF-ML-MALFORMED-PERCENT')}
_SAFE_ACTIONS = {"block"}
_RULE_ID=re.compile(r'^WAF-ML-[A-Z0-9-]+-[0-9A-F]{8}$')

def _now(): return datetime.now(timezone.utc).isoformat()
def _fingerprint(m,a): return hashlib.sha256(json.dumps({'matcher':dict(sorted(m.items())),'action':a},sort_keys=True).encode()).hexdigest()[:8].upper()
def _ruleset_hash(rules): return hashlib.sha256(json.dumps([r.canonical() for r in sorted(rules,key=lambda x:x.rule_id)],sort_keys=True,separators=(',',':')).encode()).hexdigest()

class RuleValidator:
    def validate(self,rule):
        e=[]; m=rule.matcher; f=m.get('feature'); op=m.get('operator'); t=m.get('threshold')
        if not _RULE_ID.fullmatch(rule.rule_id): e.append('rule_id does not match the Phase 6 managed rule format')
        if rule.matcher_type!='feature_threshold': e.append('only feature_threshold matcher_type is deployable')
        if f not in FEATURE_RULES: e.append('feature outside the allowlist')
        if op!='>=': e.append('only >= operator is supported')
        if t!=1.0: e.append('threshold must equal 1.0')
        if rule.action not in _SAFE_ACTIONS: e.append('action outside safe deployable action set')
        if rule.confidence<0.80: e.append('confidence below deployable floor')
        if rule.source!='phase5-decision-evidence': e.append('unsupported rule source')
        if not rule.name.strip() or not rule.description.strip(): e.append('name and description are required')
        if rule.version<1: e.append('rule version must be positive')
        if rule.source_request_id and any(c in rule.source_request_id for c in '\r\n'): e.append('source_request_id contains a prohibited control character')
        if rule.source.startswith('raw:'): e.append('raw request material cannot be a rule source')
        return RuleValidation(not e,tuple(e),())

class RuleLifecycleManager:
    def __init__(self,min_confidence=.70):
        if not 0<=min_confidence<=1: raise ValueError('min_confidence must be in [0, 1]')
        self.min_confidence=min_confidence; self._rules={}; self._deployments=[]; self._audit=[]; self._current_revision=0; self._lock=RLock(); self.validator=RuleValidator()
    def _require(self,rid):
        if rid not in self._rules: raise KeyError(f'unknown rule: {rid}')
        return self._rules[rid]
    def generate_from_evidence(self,evidence):
        if evidence.schema_version!='evidence-v1': raise ValueError('unsupported evidence schema')
        out=[]
        with self._lock:
            if evidence.decision=='allow': return []
            for row in evidence.detector_contributions:
                detector=str(row.get('detector','')); contribution=float(row.get('risk_contribution',0)); reasons=tuple(str(x).lower() for x in row.get('reasons',()))
                if detector not in {'supervised-v1','unsupervised-oneclasssvm-v1'} or (detector=='unsupervised-oneclasssvm-v1' and contribution<0): continue
                attribution=float(evidence.feature_attribution.get(detector,{}).get('attack_signatures',0))
                for feature,(label,prefix) in FEATURE_RULES.items():
                    if float(evidence.feature_snapshot.get(feature,0))<.5: continue
                    named=any(feature in r for r in reasons)
                    if detector=='supervised-v1' and not named: continue
                    if detector=='unsupervised-oneclasssvm-v1' and not named and attribution<.25: continue
                    conf=min(.99,max(self.min_confidence,.70+contribution*.30+float(row.get('confidence',0))*.20+attribution*.20))
                    matcher={'feature':feature,'operator':'>=','threshold':1.0}; rid=f'{prefix}-{_fingerprint(matcher,"block")}'
                    if rid in self._rules: continue
                    rule=RuleRecord(rid,f'ML-derived {label} rule',f'Generated from Phase 5 decision evidence when {feature} was active; deployment remains human-approved.','feature_threshold',matcher,'block',round(conf,6),'phase5-decision-evidence',evidence.request_id,detector,tuple(sorted(evidence.rule_ids)))
                    self._rules[rid]=rule; out.append(rule); self._audit.append({'event':'rule_generated','rule_id':rid,'status':rule.status.value,'source':rule.source,'source_request_id':rule.source_request_id,'at':rule.created_at})
        return out
    def validate(self,rid):
        with self._lock:
            r=self._require(rid); v=self.validator.validate(r); r.validation=v; r.status=RuleStatus.PENDING_APPROVAL if v.valid else RuleStatus.REJECTED; r.updated_at=_now(); self._audit.append({'event':'rule_validated','rule_id':rid,'status':r.status.value,'valid':v.valid,'errors':list(v.errors),'at':r.updated_at}); return v
    def approve(self,rid,approver):
        approver=approver.strip()
        if not approver: raise ValueError('approver is required')
        with self._lock:
            r=self._require(rid)
            if r.status!=RuleStatus.PENDING_APPROVAL or not r.validation or not r.validation.valid: raise ValueError('rule must pass validation before approval')
            r.status=RuleStatus.APPROVED; r.approved_by=approver; r.approved_at=_now(); r.updated_at=r.approved_at; self._audit.append({'event':'rule_approved','rule_id':rid,'status':r.status.value,'approved_by':approver,'at':r.approved_at}); return r
    def reject(self,rid,reason=''):
        with self._lock:
            r=self._require(rid); r.status=RuleStatus.REJECTED; r.updated_at=_now(); self._audit.append({'event':'rule_rejected','rule_id':rid,'status':r.status.value,'reason':reason[:160],'at':r.updated_at}); return r
    def _active_map(self): return {rid:self._rules[rid] for rid in self._current_active_ids()}
    def deploy_approved(self):
        with self._lock:
            approved=[r for r in self._rules.values() if r.status==RuleStatus.APPROVED]
            if not approved: raise ValueError('no approved rules ready for deployment')
            active=self._active_map(); active.update({r.rule_id:r for r in approved}); rules=tuple(sorted(active.values(),key=lambda r:r.rule_id)); rev=self._current_revision+1; ts=_now()
            for r in approved: r.status=RuleStatus.DEPLOYED; r.deployed_at=ts; r.updated_at=ts
            did=f'DEPLOY-{rev:04d}-{hashlib.sha256((ts+"".join(r.rule_id for r in rules)).encode()).hexdigest()[:8].upper()}'; snap=_Deployment(did,rev,tuple(r.rule_id for r in rules),_ruleset_hash(rules),ts,'deployed'); self._deployments.append(snap); self._current_revision=rev; self._audit.append({'event':'rules_deployed','deployment_id':did,'revision':rev,'active_rule_ids':list(snap.active_rule_ids),'ruleset_sha256':snap.ruleset_sha256,'at':ts}); return {'schema_version':DEPLOYMENT_SCHEMA_VERSION,'deployment_id':did,'revision':rev,'active_rule_ids':list(snap.active_rule_ids),'ruleset_sha256':snap.ruleset_sha256,'deployed_at':ts,'status':'deployed'}
    def rollback(self,did):
        with self._lock:
            target=next((x for x in self._deployments if x.deployment_id==did),None)
            if target is None: raise KeyError(f'unknown deployment: {did}')
            idx=self._deployments.index(target); previous=self._deployments[idx-1] if idx else None
            restore=(previous.active_rule_ids if previous else ()) if idx==len(self._deployments)-1 else target.active_rule_ids
            active=set(self._current_active_ids()); restore_set=set(restore); rev=self._current_revision+1; ts=_now(); rid=f'ROLLBACK-{rev:04d}-{hashlib.sha256((ts+did).encode()).hexdigest()[:8].upper()}'
            for r in self._rules.values():
                if r.rule_id in restore_set:
                    if r.status==RuleStatus.ROLLED_BACK: r.status=RuleStatus.DEPLOYED
                elif r.rule_id in active: r.status=RuleStatus.ROLLED_BACK; r.updated_at=ts
            rules=[self._rules[x] for x in restore]; snap=_Deployment(rid,rev,tuple(restore),_ruleset_hash(rules),ts,'rollback',target.deployment_id); self._deployments.append(snap); self._current_revision=rev; self._audit.append({'event':'rules_rolled_back','deployment_id':rid,'rollback_of':did,'revision':rev,'active_rule_ids':list(restore),'ruleset_sha256':snap.ruleset_sha256,'at':ts}); return {'schema_version':DEPLOYMENT_SCHEMA_VERSION,'deployment_id':rid,'revision':rev,'active_rule_ids':list(restore),'ruleset_sha256':snap.ruleset_sha256,'deployed_at':ts,'status':'rollback','rollback_of':did}
    def active_rule_ids(self): return self._current_active_ids()
    def _current_active_ids(self): return tuple(self._deployments[-1].active_rule_ids) if self._deployments else ()
    def active_rules(self): return tuple(self._rules[x] for x in self._current_active_ids())
    def match(self,features):
        out=[]
        for r in self.active_rules():
            m=r.matcher; f=m.get('feature'); v=float(features.values.get(f,0))
            if m.get('operator')=='>=' and v>=float(m.get('threshold',1)): out.append(r.rule_id)
        return tuple(out)
    def get(self,rid): return self._require(rid)
    def audit_log(self): return tuple(self._audit)
    def export_snapshot(self): return {'schema_version':DEPLOYMENT_SCHEMA_VERSION,'manager':'phase6-rule-lifecycle-v1','revision':self._current_revision,'rules':[r.to_dict() for r in self._rules.values()],'deployments':[asdict(x) for x in self._deployments],'audit':list(self._audit)}
    def ruleset_metadata(self):
        active=self.active_rules(); return {'schema_version':RULE_SCHEMA_VERSION,'ruleset_sha256':_ruleset_hash(active),'active_rule_count':len(active),'deployment_revision':self._current_revision,'manager':'phase6-rule-lifecycle-v1'}
