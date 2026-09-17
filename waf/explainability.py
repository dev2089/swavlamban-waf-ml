from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from waf.core.models import DecisionEvidence, DecisionResult, FeatureVector, RequestEnvelope


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "request_shape": (
        "method_code", "method_known", "scheme_https", "host_length", "path_length",
        "normalized_path_length", "query_length", "body_length", "header_count", "header_bytes",
    ),
    "query_structure": (
        "query_param_count", "unique_query_key_count", "duplicate_query_key_count", "query_entropy", "query_key_entropy",
    ),
    "payload_structure": (
        "has_json_body", "has_form_body", "has_xml_body", "has_multipart_body", "has_content_length",
        "content_length_mismatch", "body_entropy", "body_utf8_replacement_ratio",
    ),
    "encoding_obfuscation": (
        "percent_encoded_ratio", "malformed_percent_flag", "double_encoded_flag", "null_byte_flag", "control_char_ratio",
    ),
    "attack_signatures": (
        "has_sql_keyword", "has_xss_token", "has_traversal", "has_command_token",
    ),
    "character_statistics": (
        "path_entropy", "target_entropy", "special_char_ratio", "digit_ratio", "alpha_ratio",
    ),
}


def _feature_snapshot(features: FeatureVector) -> dict[str, float]:
    return {name: round(float(value), 6) for name, value in sorted(features.values.items())}


def _group_snapshot(features: FeatureVector) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group, names in FEATURE_GROUPS.items():
        values = {name: round(float(features.values.get(name, 0.0)), 6) for name in names if name in features.values}
        out[group] = {
            "feature_count": len(values),
            "mean": round(float(np.mean(list(values.values()))) if values else 0.0, 6),
            "max": round(float(max(values.values())) if values else 0.0, 6),
            "active_features": [name for name, value in values.items() if value >= 0.5],
        }
    return out


def _supervised_group_deltas(detector, features: FeatureVector) -> dict[str, float]:
    names = tuple(detector.feature_names)
    base = np.asarray([[features.values.get(name, 0.0) for name in names]], dtype=float)
    original = float(detector.model.predict_proba(base)[0, 1])
    deltas: dict[str, float] = {}
    for group, group_names in FEATURE_GROUPS.items():
        perturbed = base.copy()
        indices = [i for i, name in enumerate(names) if name in group_names]
        if indices:
            perturbed[0, indices] = 0.0
        altered = float(detector.model.predict_proba(perturbed)[0, 1])
        deltas[group] = round(original - altered, 6)
    return deltas


def _anomaly_group_deltas(detector, features: FeatureVector) -> dict[str, float]:
    names = tuple(detector.feature_names)
    base = np.asarray([[features.values.get(name, 0.0) for name in names]], dtype=float)
    original_decision = float(detector.model.decision_function(base)[0])

    def risk(decision: float) -> float:
        value = 1.0 / (1.0 + np.exp(8.0 * (decision - detector.decision_threshold)))
        return float(max(0.0, min(1.0, value)))

    original = risk(original_decision)
    deltas: dict[str, float] = {}
    for group, group_names in FEATURE_GROUPS.items():
        perturbed = base.copy()
        indices = [i for i, name in enumerate(names) if name in group_names]
        if indices:
            perturbed[0, indices] = 0.0
        altered = risk(float(detector.model.decision_function(perturbed)[0]))
        deltas[group] = round(original - altered, 6)
    return deltas


def _detector_contributions(result: DecisionResult) -> list[dict[str, Any]]:
    signals = result.signals
    signature = next((s for s in signals if s.detector == "open-source-waf-rules"), None)
    if signature is not None and signature.rule_ids:
        return [
            {
                "detector": signal.detector,
                "score": signal.score,
                "confidence": signal.confidence,
                "risk_contribution": 1.0 if signal is signature else 0.0,
                "reasons": list(signal.reasons),
                "rule_ids": list(signal.rule_ids),
            }
            for signal in signals
        ]

    weights = {
        "supervised-v1": 0.55,
        "unsupervised-oneclasssvm-v1": 0.30,
        "behaviour-v1": 0.15,
    }
    max_score = max((signal.score for signal in signals), default=0.0)
    maxima = [signal for signal in signals if signal.score == max_score and max_score > 0.0]
    rows: list[dict[str, Any]] = []
    for signal in signals:
        weighted = weights.get(signal.detector, 0.0) * signal.score * 0.65
        max_term = (0.35 * max_score / len(maxima)) if signal in maxima else 0.0
        rows.append(
            {
                "detector": signal.detector,
                "score": signal.score,
                "confidence": signal.confidence,
                "risk_contribution": round(weighted + max_term, 6),
                "reasons": list(signal.reasons),
                "rule_ids": list(signal.rule_ids),
            }
        )
    return rows


def build_decision_evidence(
    request: RequestEnvelope,
    features: FeatureVector,
    result: DecisionResult,
    ml_runtime: Any,
) -> DecisionEvidence:
    group_values = _group_snapshot(features)
    attribution: dict[str, dict[str, float]] = {}
    attribution["supervised-v1"] = _supervised_group_deltas(ml_runtime.supervised, features)
    attribution["unsupervised-oneclasssvm-v1"] = _anomaly_group_deltas(ml_runtime.anomaly, features)
    attribution["behaviour-v1"] = {"behaviour_state": round(float(next((s.score for s in result.signals if s.detector == "behaviour-v1"), 0.0)), 6)}
    attribution["open-source-waf-rules"] = {
        "attack_signatures": 1.0 if any(s.detector == "open-source-waf-rules" and s.rule_ids for s in result.signals) else 0.0
    }

    model_metadata = ml_runtime.metadata()
    versions = {
        "pipeline_version": result.pipeline_version,
        "feature_schema": features.schema_version,
        "model_version": model_metadata.get("model_version"),
        "dataset_version": model_metadata.get("dataset_version"),
        "baseline_version": model_metadata.get("baseline_version"),
        "ruleset": "builtin-open-source-waf-v2",
        "evidence_schema": "evidence-v1",
    }
    explanation = _human_explanation(result, group_values)
    return DecisionEvidence(
        schema_version="evidence-v1",
        decision=result.decision.value,
        risk_score=result.risk_score,
        detector_contributions=tuple(_detector_contributions(result)),
        feature_groups=group_values,
        feature_attribution=attribution,
        reasons=result.reasons,
        rule_ids=result.rule_ids,
        versions=versions,
        explanation=explanation,
        privacy={
            "raw_payload_retained": False,
            "raw_headers_retained": False,
            "raw_query_retained": False,
            "request_id_only": True,
            "numeric_feature_snapshot": True,
        },
        request_id=request.request_id,
        feature_snapshot=_feature_snapshot(features),
    )


def _human_explanation(result: DecisionResult, groups: dict[str, dict[str, Any]]) -> str:
    active = [name for name, value in groups.items() if value["active_features"]]
    if result.rule_ids:
        return f"Decision {result.decision.value}: signature rule(s) {', '.join(result.rule_ids)} matched; risk was forced to 1.0 by edge policy."
    if active:
        return f"Decision {result.decision.value}: model and behavioural signals combined to risk {result.risk_score:.6f}; active feature groups: {', '.join(active)}."
    return f"Decision {result.decision.value}: no signature rule matched and model signals combined to risk {result.risk_score:.6f}."


def evidence_to_dict(evidence: DecisionEvidence) -> dict[str, Any]:
    return asdict(evidence)
