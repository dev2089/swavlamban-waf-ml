from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from waf.core.models import DecisionResult


def decision_event(result: DecisionResult) -> dict[str, Any]:
    return {
        "event_type": "waf.decision",
        "schema_version": "event-v1",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "request_id": result.request_id,
        "decision": result.decision.value,
        "risk_score": result.risk_score,
        "reasons": list(result.reasons),
        "rule_ids": list(result.rule_ids),
        "signals": [asdict(s) for s in result.signals],
        "pipeline_version": result.pipeline_version,
    }
