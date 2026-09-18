from __future__ import annotations
import unittest
from datetime import datetime, timezone
from waf.config import Settings
from waf.core.models import Decision, DecisionAction, DetectionSignal, RequestContext
from waf.core.pipeline import FastPathPipeline

class FeatureStage:
    name='features'
    def run(self, request, state): return state.with_features({'path_length': len(request.path)})
class DetectionStage:
    name='detector'
    def run(self, request, state): return state.with_signals((DetectionSignal('test-detector',0.25,('test signal',)),))
class DecisionStage:
    name='decision'
    def run(self, request, state): return state.with_decision(Decision(DecisionAction.ALLOW,max((s.score for s in state.signals),default=0.0),state.signals,('test signal',)))
class ArchitectureTests(unittest.TestCase):
    def request(self): return RequestContext('req-1',datetime.now(timezone.utc),'get','https','example.test','/api/items',headers={'X-Test':'1'})
    def test_request_normalization_and_header_immutability(self):
        r=self.request(); self.assertEqual(r.method,'GET')
        with self.assertRaises(TypeError): r.headers['X-New']='2'
    def test_pipeline_order_and_evidence(self):
        s=FastPathPipeline((FeatureStage(),DetectionStage(),DecisionStage())).execute(self.request())
        self.assertEqual(s.features['path_length'],10); self.assertEqual(len(s.signals),1); self.assertEqual(s.decision.action,DecisionAction.ALLOW); self.assertIn('test signal',s.decision.reasons)
    def test_duplicate_stage_names_fail_fast(self):
        with self.assertRaises(ValueError): FastPathPipeline((FeatureStage(),FeatureStage()))
    def test_settings_defaults_are_bounded(self):
        s=Settings(); self.assertTrue(1<=s.port<=65535); self.assertGreater(s.max_body_bytes,0); self.assertGreater(s.telemetry_queue_max,0)
    def test_detection_scores_are_bounded(self):
        with self.assertRaises(ValueError): DetectionSignal('bad',1.1)
if __name__=='__main__': unittest.main()