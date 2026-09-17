from waf.core.models import Decision, DecisionResult, DetectionSignal, RequestEnvelope


class EdgeDecisionPolicy:
    """Phase 4 deterministic multi-signal edge policy."""

    def __init__(self, block_threshold: float = 0.80, alert_threshold: float = 0.50) -> None:
        if not 0.0 <= alert_threshold <= block_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= alert <= block <= 1")
        self.block_threshold = block_threshold
        self.alert_threshold = alert_threshold

    def decide(self, request: RequestEnvelope, signals: tuple[DetectionSignal, ...], pipeline_version: str) -> DecisionResult:
        signature = next((s for s in signals if s.detector == "open-source-waf-rules"), None)
        if signature is not None and signature.rule_ids:
            risk = 1.0
        else:
            weights = {
                "supervised-v1": 0.55,
                "unsupervised-oneclasssvm-v1": 0.30,
                "behaviour-v1": 0.15,
            }
            weighted = sum(weights.get(s.detector, 0.0) * s.score for s in signals)
            risk = max((s.score for s in signals), default=0.0) * 0.35 + weighted * 0.65
        if risk >= self.block_threshold:
            decision = Decision.BLOCK
        elif risk >= self.alert_threshold:
            decision = Decision.ALERT
        else:
            decision = Decision.ALLOW
        reasons = tuple(dict.fromkeys(reason for signal in signals for reason in signal.reasons))
        rule_ids = tuple(dict.fromkeys(rule_id for signal in signals for rule_id in signal.rule_ids))
        return DecisionResult(
            decision=decision,
            risk_score=round(min(1.0, max(0.0, risk)), 6),
            reasons=reasons,
            rule_ids=rule_ids,
            signals=signals,
            request_id=request.request_id,
            pipeline_version=pipeline_version,
        )
