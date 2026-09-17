from __future__ import annotations

from dataclasses import replace
import os

from waf.core.config import WAFConfig
from waf.core.models import DecisionResult, RequestEnvelope
from waf.edge.policy import EdgeDecisionPolicy
from waf.edge.rules import OpenSourceWAFRuleEngine
from waf.explainability import build_decision_evidence
from waf.features.http_v2 import ProductionHTTPFeatureExtractor
from waf.ml.ensemble import load_runtime
from waf.rules.lifecycle import RuleLifecycleManager


class EdgeWAF:
    """Phase 5 live path: signatures + ML + behavioural detection + evidence."""

    def __init__(self, config: WAFConfig) -> None:
        self.config = config
        self.features = ProductionHTTPFeatureExtractor()
        self.rule_lifecycle = RuleLifecycleManager()
        self.signature_detector = OpenSourceWAFRuleEngine(self.rule_lifecycle)
        artifact = os.getenv("WAF_MODEL_ARTIFACT", "models/phase4_models.joblib")
        self.ml = load_runtime(artifact)
        self.policy = EdgeDecisionPolicy(config.block_threshold, config.alert_threshold)

    def analyze(self, request: RequestEnvelope) -> DecisionResult:
        if len(request.body) > self.config.max_body_bytes:
            request = replace(request, body=request.body[: self.config.max_body_bytes])
        features = self.features.extract(request)
        if features.schema_version != self.config.feature_schema_version:
            raise ValueError(
                f"feature schema mismatch: expected {self.config.feature_schema_version}, got {features.schema_version}"
            )
        signature = self.signature_detector.detect(request, features)
        ml_signals = self.ml.detect(request, features)
        result = self.policy.decide(request, (signature, *ml_signals), self.config.pipeline_version)
        evidence = build_decision_evidence(request, features, result, self.ml)
        # Phase 6 provenance is attached at the live seam so direct Phase 5
        # evidence construction remains backward-compatible.
        versions = dict(evidence.versions)
        runtime_meta = self.ml.metadata()
        versions.update({
            "managed_ruleset_schema": self.rule_lifecycle.ruleset_metadata()["schema_version"],
            "managed_ruleset_sha256": self.rule_lifecycle.ruleset_metadata()["ruleset_sha256"],
            "managed_rule_count": self.rule_lifecycle.ruleset_metadata()["active_rule_count"],
            "managed_deployment_revision": self.rule_lifecycle.ruleset_metadata()["deployment_revision"],
            "learning_control_schema": "phase7-learning-control-v1",
            "learning_model_version": runtime_meta["model_version"],
            "learning_baseline_version": runtime_meta["baseline_version"],
        })
        evidence = replace(evidence, versions=versions)
        return replace(result, evidence=evidence)

    def recommend_rules(self, result: DecisionResult):
        if result.evidence is None:
            raise ValueError("decision evidence is required before generating managed rules")
        return self.rule_lifecycle.generate_from_evidence(result.evidence)

    def validate_rule(self, rule_id: str):
        return self.rule_lifecycle.validate(rule_id)

    def approve_rule(self, rule_id: str, approver: str):
        return self.rule_lifecycle.approve(rule_id, approver)

    def deploy_approved_rules(self):
        return self.rule_lifecycle.deploy_approved()

    def rollback_rules(self, deployment_id: str):
        return self.rule_lifecycle.rollback(deployment_id)
