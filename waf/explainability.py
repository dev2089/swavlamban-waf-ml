from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from waf.core.models import DecisionEvidence, DecisionResult, FeatureVector

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "request_shape": (
        "method_code", "method_known", "scheme_https", "host_length", "path_length",
        "normalized_path_length", "query_length", "body_length", "header_count", "header_bytes",
    ),
    "query_structure": (
        "query_param_count", "unique_query_key_count", "duplicate_query_key_count",
        "query_parse_overflow", "query_entropy", "query_key_entropy",
    ),
    "payload_structure": (
        "has_json_body", "has_form_body", "has_xml_body", "has_multipart_body",
        "has_content_length", "content_length_mismatch", "body_entropy", "body_utf8_replacement_ratio",
    ),
    "encoding_obfuscation": (
        "percent_encoded_ratio", "malformed_percent_flag", "double_encoded_flag",
        "null_byte_flag", "control_char_ratio",
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


def _group_indices(names: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    name_to_index = {name: index for index, name in enumerate(names)}
    return [(group, [name_to_index[name] for name in group_names if name in name_to_index]) for group, group_names in FEATURE_GROUPS.items()]


def _supervised_group_deltas(detector, features: FeatureVector) -> dict[str, float]:
    names = tuple(detector.feature_names)
    base = np.asarray([[float(features.values.get(name, 0.0)) for name in names]], dtype=float)
    original = float(detector.model.predict_proba(base)[0, 1])
    rows: list[np.ndarray] = []
    groups: list[str] = []
    for group, indices in _group_indices(names):
        if indices:
            perturbed = base.copy()
            perturbed[0, indices] = 0.0
            rows.append(perturbed[0])
            groups.append(group)
    altered = detector.model.predict_proba(np.asarray(rows, dtype=float))[:, 1] if rows else np.asarray([])
    deltas = {group: 0.0 for group in FEATURE_GROUPS}
    for group, score in zip(groups, altered):
        deltas[group] = round(original - float(score), 6)
    return deltas


def _anomaly_group_deltas(detector, features: FeatureVector) -> dict[str, float]:
    names = tuple(detector.feature_names)
    base = np.asarray([[float(features.values.get(name, 0.0)) for name in names]], dtype=float)

    def risk(decisions: np.ndarray) -> np.ndarray:
        values = 1.0 / (1.0 + np.exp(8.0 * (decisions - detector.decision_threshold)))
        return np.clip(values, 0.0, 1.0)

    original = float(risk(detector.model.decision_function(base))[0])
    rows: list[np.ndarray] = []
    groups: list[str] = []
    for group, indices in _group_indices(names):
        if indices:
            perturbed = base.copy()
            perturbed[0, indices] = 0.0
            rows.append(perturbed[0])
            groups.append(group)
    altered = risk(detector.model.decision_function(np.asarray(rows, dtype=float))) if rows else np.asarray([])
    deltas = {group: 0.0 for group in FEATURE_GROUPS}
    for group, score in zip(groups, altered):
        deltas[group] = round(original - float(score), 6)
    return deltas


def _detector_contributions(result: DecisionResult) -> list[dict[str, Any]]:
    signature = next((s for s in result.signals if s.detector == "open-source-waf-rules"), None)
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
            for signal in result.signals
        ]

    weights = {
        "supervised-v1": 0.55,
        "unsupervised-oneclasssvm-v1": 0.30,
        "behaviour-v1": 0.15,
        "ml-runtime-failure": 1.0,
    }
    ml_signals = [signal for signal in result.signals if signal.detector != "open-source-waf-rules"]
    max_score = max((signal.score for signal in ml_signals), default=0.0)
    maxima = [signal for signal in ml_signals if signal.score == max_score and max_score > 0.0]
    rows: list[dict[str, Any]] = []
    for signal in result.signals:
        if signal.detector == "open-source-waf-rules":
            contribution = 0.0
        elif signal.detector == "ml-runtime-failure":
            contribution = 1.0
        else:
            weighted = weights.get(signal.detector, 0.0) * signal.score * 0.65
            max_term = (0.35 * max_score / len(maxima)) if signal in maxima else 0.0
            contribution = weighted + max_term
        rows.append({
            "detector": signal.detector,
            "score": signal.score,
            "confidence": signal.confidence,
            "risk_contribution": round(contribution, 6),
            "reasons": list(signal.reasons),
            "rule_ids": list(signal.rule_ids),
        })
    return rows


def _human_explanation(result: DecisionResult, groups: dict[str, dict[str, Any]]) -> str:
    if any(s.detector == "ml-runtime-failure" for s in result.signals):
        return f"Decision {result.decision.value}: ML inference failed, so the WAF used fail-closed enforcement at risk 1.0."
    if result.rule_ids:
        return f"Decision {result.decision.value}: signature rule(s) {', '.join(result.rule_ids)} matched; edge policy forced risk to 1.0."
    active = [name for name, value in groups.items() if value["active_features"]]
    if active:
        return f"Decision {result.decision.value}: model and behavioural signals combined to risk {result.risk_score:.6f}; active feature groups: {', '.join(active)}."
    return f"Decision {result.decision.value}: no signature rule matched and model signals combined to risk {result.risk_score:.6f}."


def build_decision_evidence(request, features: FeatureVector, result: DecisionResult, ml_runtime: Any) -> DecisionEvidence:
    feature_snapshot = _feature_snapshot(features)
    if len(feature_snapshot) != 40:
        raise ValueError("decision evidence requires the complete 40-feature http-v2 snapshot")
    if any(not 0.0 <= value <= 1.0 for value in feature_snapshot.values()):
        raise ValueError("feature snapshot must be bounded")

    group_values = _group_snapshot(features)
    if hasattr(ml_runtime, "supervised") and hasattr(ml_runtime, "anomaly"):
        supervised_attribution = _supervised_group_deltas(ml_runtime.supervised, features)
        anomaly_attribution = _anomaly_group_deltas(ml_runtime.anomaly, features)
        model_metadata = ml_runtime.metadata()
    else:
        supervised_attribution = {group: 0.0 for group in FEATURE_GROUPS}
        anomaly_attribution = {group: 0.0 for group in FEATURE_GROUPS}
        model_metadata = {}

    attribution = {
        "supervised-v1": supervised_attribution,
        "unsupervised-oneclasssvm-v1": anomaly_attribution,
        "behaviour-v1": {
            "behaviour_risk_score": round(float(next((s.score for s in result.signals if s.detector == "behaviour-v1"), 0.0)), 6)
        },
        "open-source-waf-rules": {
            "attack_signatures": 1.0 if any(s.detector == "open-source-waf-rules" and s.rule_ids for s in result.signals) else 0.0
        },
    }
    if any(s.detector == "ml-runtime-failure" for s in result.signals):
        attribution["ml-runtime-failure"] = {"fail_closed": 1.0}

    versions = {
        "pipeline_version": result.pipeline_version,
        "feature_schema": features.schema_version,
        "model_version": model_metadata.get("model_version"),
        "dataset_version": model_metadata.get("dataset_version"),
        "baseline_version": model_metadata.get("baseline_version"),
        "ruleset": "builtin-open-source-waf-v2",
        "evidence_schema": "evidence-v1",
    }

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
        explanation=_human_explanation(result, group_values),
        privacy={
            "raw_payload_retained": False,
            "raw_headers_retained": False,
            "raw_query_retained": False,
            "source_ip_retained": False,
            "host_retained": False,
            "request_id_only": True,
            "numeric_feature_snapshot": True,
        },
        request_id=request.request_id,
        feature_snapshot=feature_snapshot,
    )


def evidence_to_dict(evidence: DecisionEvidence) -> dict[str, Any]:
    return asdict(evidence)
