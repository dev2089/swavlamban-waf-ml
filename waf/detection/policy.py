from __future__ import annotations

from waf.core.models import Decision, DecisionResult, DetectionSignal, RequestEnvelope


class ThresholdDecisionPolicy:
    """Deterministic decision policy over detector signals."""

    def __init__(self, block_threshold: float = 0.80, alert_threshold: float = 0.50) -> None:
        if not 0.0 <= alert_threshold <= block_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= alert <= block <= 1")
        self.block_threshold = block_threshold
        self.alert_threshold = alert_threshold

    def decide(self, request: RequestEnvelope, signals: tuple[DetectionSignal, ...], pipeline_version: str) -> DecisionResult:
        if not signals:
            risk = 0.0
        else:
            weighted = [s.score * (0.5 + 0.5 * s.confidence) for s in signals]
            risk = max(weighted)
            independent_high = sum(1 for s in signals if s.score >= self.alert_threshold and s.confidence >= 0.5)
            if independent_high >= 2:
                risk = max(risk, min(1.0, max(s.score for s in signals) + 0.10))
        if risk >= self.block_threshold:
            decision = Decision.BLOCK
        elif risk >= self.alert_threshold:
            decision = Decision.ALERT
        else:
            decision = Decision.ALLOW
        reasons: list[str] = []
        rule_ids: list[str] = []
        for signal in signals:
            reasons.extend(signal.reasons)
            rule_ids.extend(signal.rule_ids)
        return DecisionResult(
            decision=decision,
            risk_score=round(risk, 6),
            reasons=tuple(dict.fromkeys(reasons)),
            rule_ids=tuple(dict.fromkeys(rule_ids)),
            signals=signals,
            request_id=request.request_id,
            pipeline_version=pipeline_version,
        )
