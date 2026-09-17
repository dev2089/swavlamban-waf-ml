from __future__ import annotations
import json
from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.explainability import build_decision_evidence
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.ensemble import Phase4MLEnsemble
from waf.rules.lifecycle import RuleLifecycleManager, RuleStatus, RuleValidator

def evidence():
    req=RequestEnvelope('p6-test','GET','https','example.test','/','q=1 union select x'); ex=ProductionHTTPFeatureExtractor(); f=ex.extract(req); ml=Phase4MLEnsemble.train_default(); sig=OpenSourceWAFRuleEngine().detect(req,f)
    from waf.edge.policy import EdgeDecisionPolicy
    result=EdgeDecisionPolicy().decide(req,(sig,*ml.detect(req,f)),'phase6'); return build_decision_evidence(req,f,result,ml),result

def test_generation_validation_and_privacy():
    ev,result=evidence(); assert result.decision is Decision.BLOCK; m=RuleLifecycleManager(); rules=m.generate_from_evidence(ev); assert rules
    blob=json.dumps([r.to_dict() for r in rules]).lower(); assert 'union select x' not in blob and 'example.test' not in blob
    assert all(r.matcher_type=='feature_threshold' and r.status is RuleStatus.GENERATED for r in rules); assert m.validate(rules[0].rule_id).valid

def test_approval_is_required_and_unsafe_action_rejected():
    ev,_=evidence(); m=RuleLifecycleManager(); r=m.generate_from_evidence(ev)[0]
    try: m.approve(r.rule_id,'reviewer')
    except ValueError: pass
    else: raise AssertionError('unvalidated rule was approved')
    r.action='alert'; assert not RuleValidator().validate(r).valid

def test_deploy_match_and_provenance():
    waf=EdgeWAF(WAFConfig(pipeline_version='phase6')); req=RequestEnvelope('p6-live','GET','https','example.test','/','q=1 union select x'); result=waf.analyze(req); c=waf.recommend_rules(result)[0]
    assert waf.validate_rule(c.rule_id).valid; waf.approve_rule(c.rule_id,'security-reviewer'); d=waf.deploy_approved_rules(); assert c.rule_id in d['active_rule_ids']
    post=waf.analyze(req); assert post.decision is Decision.BLOCK and c.rule_id in post.rule_ids; assert post.evidence.versions['managed_rule_count']==1; assert post.evidence.versions['managed_deployment_revision']==1; assert post.evidence.versions['managed_ruleset_sha256']

def test_rollback_restores_previous_snapshot():
    waf=EdgeWAF(WAFConfig(pipeline_version='phase6')); req=RequestEnvelope('p6-rb','GET','https','example.test','/','q=1 union select x'); c=waf.recommend_rules(waf.analyze(req))[0]; waf.validate_rule(c.rule_id); waf.approve_rule(c.rule_id,'reviewer'); d=waf.deploy_approved_rules(); rb=waf.rollback_rules(d['deployment_id']); assert rb['status']=='rollback'; assert waf.rule_lifecycle.active_rule_ids()==()

def test_unknown_matcher_is_rejected():
    ev,_=evidence(); m=RuleLifecycleManager(); r=m.generate_from_evidence(ev)[0]; r.matcher={'feature':'unknown','operator':'>=','threshold':1.0}; assert not RuleValidator().validate(r).valid
