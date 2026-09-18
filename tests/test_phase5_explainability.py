import asyncio
import json
import pytest

from waf.core.config import WAFConfig
from waf.core.models import Decision, DecisionResult, DetectionSignal, FeatureVector, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.edge.policy import EdgeDecisionPolicy
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.explainability import build_decision_evidence, evidence_to_dict
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.ensemble import Phase4MLEnsemble
from waf.telemetry.events import decision_event


def build(req, ml):
    f = ProductionHTTPFeatureExtractor().extract(req)
    s = OpenSourceWAFRuleEngine().detect(req, f)
    r = EdgeDecisionPolicy().decide(req, (s, *ml.detect(req, f)), 'phase5')
    return r, f


def test_known_attack_rule_and_privacy():
    ml = Phase4MLEnsemble.train_default(); req = RequestEnvelope('attack','GET','https','example.test','/','id=1 union select password from users')
    result, features = build(req, ml); e = build_decision_evidence(req, features, result, ml)
    assert result.decision is Decision.BLOCK and 'WAF-SQL-001' in e.rule_ids
    assert e.privacy['raw_payload_retained'] is False and e.privacy['raw_query_retained'] is False and e.privacy['raw_headers_retained'] is False


def test_evidence_has_complete_bounded_features_and_no_raw_data():
    ml = Phase4MLEnsemble.train_default(); req = RequestEnvelope('private','POST','https','private.example','/api','q=secret-query',{'X-Private':'secret-header'},b'secret-body')
    result, features = build(req, ml); e = build_decision_evidence(req, features, result, ml); encoded = json.dumps(evidence_to_dict(e), sort_keys=True)
    assert len(features.values) == 40 and len(e.feature_snapshot) == 40
    assert all(0.0 <= v <= 1.0 for v in features.values.values())
    for bad in ('private.example','secret-query','secret-header','secret-body'): assert bad not in encoded


def test_group_attribution_and_provenance():
    ml = Phase4MLEnsemble.train_default(); req = RequestEnvelope('anom','GET','https','strange.example','/'+'A'*4000,'q='+'Z'*8000)
    result, features = build(req, ml); e = build_decision_evidence(req, features, result, ml)
    assert set(e.feature_attribution) >= {'supervised-v1','unsupervised-oneclasssvm-v1','behaviour-v1','open-source-waf-rules'}
    assert e.versions['feature_schema'] == 'http-v2' and e.versions['evidence_schema'] == 'evidence-v1'


def test_explanation_is_deterministic():
    ml = Phase4MLEnsemble.train_default(); req = RequestEnvelope('same','GET','https','example.test','/health','page=1',timestamp=100.0)
    result, features = build(req, ml); a = build_decision_evidence(req, features, result, ml); b = build_decision_evidence(req, features, result, ml)
    assert a.feature_snapshot == b.feature_snapshot and a.feature_attribution == b.feature_attribution and a.explanation == b.explanation


def test_live_edge_attaches_evidence_and_event_v2():
    waf = EdgeWAF(WAFConfig(pipeline_version='phase5')); result = waf.analyze(RequestEnvelope('live','GET','https','example.test','/health'))
    event = decision_event(result)
    assert result.evidence is not None and result.evidence.request_id == 'live'
    assert event['schema_version'] == 'event-v2' and event['evidence']['schema_version'] == 'evidence-v1'


def test_ml_failure_stays_fail_closed_and_explainable():
    waf = EdgeWAF(WAFConfig(pipeline_version='phase5'))
    class Broken:
        def detect(self, request, features): raise RuntimeError('simulated')
    waf.ml = Broken(); result = waf.analyze(RequestEnvelope('fail','GET','https','example.test','/health'))
    assert result.decision is Decision.BLOCK and result.evidence is not None and 'fail-closed' in result.evidence.explanation


def test_privacy_contract_rejects_true_raw_retention():
    ml = Phase4MLEnsemble.train_default(); req = RequestEnvelope('p','GET','https','example.test','/'); result, features = build(req, ml); e = build_decision_evidence(req, features, result, ml)
    with pytest.raises(ValueError):
        type(e)(schema_version=e.schema_version,decision=e.decision,risk_score=e.risk_score,detector_contributions=e.detector_contributions,feature_groups=e.feature_groups,feature_attribution=e.feature_attribution,reasons=e.reasons,rule_ids=e.rule_ids,versions=e.versions,explanation=e.explanation,privacy={**e.privacy,'raw_payload_retained':True},request_id=e.request_id,feature_snapshot=e.feature_snapshot)


def test_legacy_result_remains_constructible_without_evidence():
    result = DecisionResult(Decision.ALLOW,0.0,(),(),(DetectionSignal('fixed',0.0,1.0),),'legacy','phase4')
    assert result.evidence is None


def test_proxy_event_has_evidence():
    async def go():
        from aiohttp import ClientSession, web
        from waf.edge.reverse_proxy import WAFReverseProxy
        async def upstream(_): return web.Response(text='ok')
        app=web.Application(); app.router.add_route('*','/{tail:.*}',upstream); ur=web.AppRunner(app); await ur.setup(); await web.TCPSite(ur,'127.0.0.1',19410).start()
        proxy=WAFReverseProxy(WAFConfig(upstream_url='http://127.0.0.1:19410',listen_port=18410,pipeline_version='phase5')); pr=web.AppRunner(proxy.app); await pr.setup(); await web.TCPSite(pr,'127.0.0.1',18410).start()
        try:
            async with ClientSession() as c:
                r=await c.get('http://127.0.0.1:18410/ok?q=private-value',headers={'X-Request-ID':'proxy-evidence'}); assert r.status==200
            e=proxy.events[-1]; assert e['request_id']=='proxy-evidence'; assert e['evidence']['request_id']=='proxy-evidence'; assert 'private-value' not in json.dumps(e)
        finally: await proxy.close(); await pr.cleanup(); await ur.cleanup()
    asyncio.run(go())


def test_wrong_feature_count_is_rejected():
    ml=Phase4MLEnsemble.train_default(); req=RequestEnvelope('bad','GET','https','example.test','/'); f=ProductionHTTPFeatureExtractor().extract(req); vals=dict(f.values); vals.pop(next(iter(vals)))
    with pytest.raises(ValueError,match='complete 40-feature'): build_decision_evidence(req,FeatureVector('http-v2',vals),DecisionResult(Decision.ALLOW,0.0,(),(),(),'bad','phase5'),ml)
