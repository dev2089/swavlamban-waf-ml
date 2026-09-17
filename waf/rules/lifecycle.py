from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib, json, re
from threading import RLock
from typing import Any, Mapping

RULE_SCHEMA_VERSION = "rule-v1"
DEPLOYMENT_SCHEMA_VERSION = "deployment-v1"


class RuleStatus(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class RuleValidation:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class RuleRecord:
    rule_id: str
    name: str
    description: str
    matcher_type: str
    matcher: Mapping[str, Any]
    action: str
    confidence: float
    source: str
    source_request_id: str
    source_detector: str
    source_rule_ids: tuple[str, ...] = ()
    schema_version: str = RULE_SCHEMA_VERSION
    version: int = 1
    status: RuleStatus = RuleStatus.GENERATED
    validation: RuleValidation | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    deployed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        n = _now()
        self.created_at = self.created_at or n
        self.updated_at = self.updated_at or n
        self.source_rule_ids = tuple(self.source_rule_ids)
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    def canonical(self):
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "matcher_type": self.matcher_type,
            "matcher": dict(sorted(self.matcher.items())),
            "action": self.action,
            "confidence": round(float(self.confidence), 6),
            "source": self.source,
            "source_request_id": self.source_request_id,
            "source_detector": self.source_detector,
            "source_rule_ids": sorted(self.source_rule_ids),
            "schema_version": self.schema_version,
            "version": self.version,
        }

    def to_dict(self):
        payload = asdict(self)
        payload["status"] = self.status.value
        if self.validation:
            payload["validation"] = asdict(self.validation)
        return payload


@dataclass(frozen=True, slots=True)
class _Deployment:
    deployment_id: str
    revision: int
    active_rule_ids: tuple[str, ...]
    ruleset_sha256: str
    deployed_at: str
    status: str
    rollback_of: str | None = None


FEATURE_RULES = {
    "has_sql_keyword": ("SQL-like signature", "WAF-ML-SQL"),
    "has_xss_token": ("XSS-like signature", "WAF-ML-XSS"),
    "has_traversal": ("path traversal-like signature", "WAF-ML-TRAV"),
    "has_command_token": ("command-injection-like signature", "WAF-ML-CMD"),
    "double_encoded_flag": ("double-encoding obfuscation", "WAF-ML-DOUBLE-ENCODE"),
    "null_byte_flag": ("null-byte obfuscation", "WAF-ML-NULL-BYTE"),
    "malformed_percent_flag": ("malformed percent encoding", "WAF-ML-MALFORMED-PERCENT"),
}
_SAFE_ACTIONS = {"block"}
_RULE_ID = re.compile(r"^WAF-ML-[A-Z0-9-]+-[0-9A-F]{8}$")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(m, action):
    return hashlib.sha256(json.dumps({"matcher": dict(sorted(m.items())), "action": action}, sort_keys=True).encode()).hexdigest()[:8].upper()


def _ruleset_hash(rules):
    return hashlib.sha256(json.dumps([r.canonical() for r in sorted(rules, key=lambda x: x.rule_id)], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RuleValidator:
    def validate(self, rule):
        errors = []
        matcher = rule.matcher
        feature = matcher.get("feature")
        operator = matcher.get("operator")
        threshold = matcher.get("threshold")
        if not _RULE_ID.fullmatch(rule.rule_id):
            errors.append("rule_id does not match the Phase 6 managed rule format")
        if rule.matcher_type != "feature_threshold":
            errors.append("only feature_threshold matcher_type is deployable")
        if feature not in FEATURE_RULES:
            errors.append("feature outside the allowlist")
        if operator != ">=":
            errors.append("only >= operator is supported")
        if threshold != 1.0:
            errors.append("threshold must equal 1.0")
        if rule.action not in _SAFE_ACTIONS:
            errors.append("action outside safe deployable action set")
        if rule.confidence < 0.80:
            errors.append("confidence below deployable floor")
        if rule.source != "phase5-decision-evidence":
            errors.append("unsupported rule source")
        if not rule.name.strip() or not rule.description.strip():
            errors.append("name and description are required")
        if rule.version < 1:
            errors.append("rule version must be positive")
        if rule.source_request_id and any(c in rule.source_request_id for c in "\r\n"):
            errors.append("source_request_id contains a prohibited control character")
        if rule.source.startswith("raw:"):
            errors.append("raw request material cannot be a rule source")
        return RuleValidation(not errors, tuple(errors), ())


class RuleLifecycleManager:
    def __init__(self, min_confidence=0.70):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        self.min_confidence = min_confidence
        self._rules = {}
        self._deployments = []
        self._audit = []
        self._current_revision = 0
        self._lock = RLock()
        self.validator = RuleValidator()

    def _require(self, rule_id):
        if rule_id not in self._rules:
            raise KeyError(f"unknown rule: {rule_id}")
        return self._rules[rule_id]

    def generate_from_evidence(self, evidence):
        if evidence.schema_version != "evidence-v1":
            raise ValueError("unsupported evidence schema")
        if evidence.decision == "allow":
            return []
        output = []
        with self._lock:
            for row in evidence.detector_contributions:
                detector = str(row.get("detector", ""))
                contribution = float(row.get("risk_contribution", 0))
                reasons = tuple(str(x).lower() for x in row.get("reasons", ()))
                if detector not in {"supervised-v1", "unsupervised-oneclasssvm-v1"}:
                    continue
                attribution = float(evidence.feature_attribution.get(detector, {}).get("attack_signatures", 0))
                for feature, (label, prefix) in FEATURE_RULES.items():
                    if float(evidence.feature_snapshot.get(feature, 0)) < 0.5:
                        continue
                    named = any(feature in reason for reason in reasons)
                    if detector == "supervised-v1" and not named:
                        continue
                    if detector == "unsupervised-oneclasssvm-v1" and not named and attribution < 0.25:
                        continue
                    confidence = min(.99, max(self.min_confidence, .70 + contribution * .30 + float(row.get("confidence", 0)) * .20 + attribution * .20))
                    matcher = {"feature": feature, "operator": ">=", "threshold": 1.0}
                    rule_id = f"{prefix}-{_fingerprint(matcher, 'block')}"
                    if rule_id in self._rules:
                        continue
                    rule = RuleRecord(
                        rule_id,
                        f"ML-derived {label} rule",
                        f"Generated from Phase 5 decision evidence when {feature} was active; deployment remains human-approved.",
                        "feature_threshold",
                        matcher,
                        "block",
                        round(confidence, 6),
                        "phase5-decision-evidence",
                        evidence.request_id,
                        detector,
                        tuple(sorted(evidence.rule_ids)),
                    )
                    self._rules[rule_id] = rule
                    output.append(rule)
                    self._audit.append({"event": "rule_generated", "rule_id": rule_id, "status": rule.status.value, "source": rule.source, "source_request_id": rule.source_request_id, "at": rule.created_at})

            # Signature-dominant decisions can legitimately contain no detector reason text.
            # Bridge the decision evidence into an allowlisted ML-derived feature rule rather
            # than silently claiming that rule recommendation succeeded with an empty result.
            if not output:
                candidates = [
                    (feature, float(value))
                    for feature, value in evidence.feature_snapshot.items()
                    if feature in FEATURE_RULES and float(value) >= 0.5
                ]
                if candidates:
                    feature, _ = max(candidates, key=lambda item: item[1])
                    label, prefix = FEATURE_RULES[feature]
                    matcher = {"feature": feature, "operator": ">=", "threshold": 1.0}
                    rule_id = f"{prefix}-{_fingerprint(matcher, 'block')}"
                    if rule_id not in self._rules:
                        rule = RuleRecord(
                            rule_id,
                            f"ML-derived {label} rule",
                            f"Generated from structured Phase 5 evidence for active feature {feature}; signature context is recorded but deployment remains human-approved.",
                            "feature_threshold",
                            matcher,
                            "block",
                            0.82,
                            "phase5-decision-evidence",
                            evidence.request_id,
                            "evidence-feature-fallback-v1",
                            tuple(sorted(evidence.rule_ids)),
                        )
                        self._rules[rule_id] = rule
                        output.append(rule)
                        self._audit.append({"event": "rule_generated_fallback", "rule_id": rule_id, "status": rule.status.value, "source": rule.source, "source_request_id": rule.source_request_id, "at": rule.created_at})
        return output

    def validate(self, rule_id):
        with self._lock:
            rule = self._require(rule_id)
            validation = self.validator.validate(rule)
            rule.validation = validation
            rule.status = RuleStatus.PENDING_APPROVAL if validation.valid else RuleStatus.REJECTED
            rule.updated_at = _now()
            self._audit.append({"event": "rule_validated", "rule_id": rule_id, "status": rule.status.value, "valid": validation.valid, "errors": list(validation.errors), "at": rule.updated_at})
            return validation

    def approve(self, rule_id, approver):
        approver = approver.strip()
        if not approver:
            raise ValueError("approver is required")
        with self._lock:
            rule = self._require(rule_id)
            if rule.status != RuleStatus.PENDING_APPROVAL or not rule.validation or not rule.validation.valid:
                raise ValueError("rule must pass validation before approval")
            rule.status = RuleStatus.APPROVED
            rule.approved_by = approver
            rule.approved_at = _now()
            rule.updated_at = rule.approved_at
            self._audit.append({"event": "rule_approved", "rule_id": rule_id, "status": rule.status.value, "approved_by": approver, "at": rule.approved_at})
            return rule

    def _active_map(self):
        return {rule_id: self._rules[rule_id] for rule_id in self._current_active_ids()}

    def deploy_approved(self):
        with self._lock:
            approved = [rule for rule in self._rules.values() if rule.status == RuleStatus.APPROVED]
            if not approved:
                raise ValueError("no approved rules ready for deployment")
            active = self._active_map()
            active.update({rule.rule_id: rule for rule in approved})
            rules = tuple(sorted(active.values(), key=lambda rule: rule.rule_id))
            revision = self._current_revision + 1
            timestamp = _now()
            for rule in approved:
                rule.status = RuleStatus.DEPLOYED
                rule.deployed_at = timestamp
                rule.updated_at = timestamp
            deployment_id = f'DEPLOY-{revision:04d}-{hashlib.sha256((timestamp+"".join(rule.rule_id for rule in rules)).encode()).hexdigest()[:8].upper()}'
            snapshot = _Deployment(deployment_id, revision, tuple(rule.rule_id for rule in rules), _ruleset_hash(rules), timestamp, "deployed")
            self._deployments.append(snapshot)
            self._current_revision = revision
            self._audit.append({"event": "rules_deployed", "deployment_id": deployment_id, "revision": revision, "active_rule_ids": list(snapshot.active_rule_ids), "ruleset_sha256": snapshot.ruleset_sha256, "at": timestamp})
            return {"schema_version": DEPLOYMENT_SCHEMA_VERSION, "deployment_id": deployment_id, "revision": revision, "active_rule_ids": list(snapshot.active_rule_ids), "ruleset_sha256": snapshot.ruleset_sha256, "deployed_at": timestamp, "status": "deployed"}

    def rollback(self, deployment_id):
        with self._lock:
            target = next((item for item in self._deployments if item.deployment_id == deployment_id), None)
            if target is None:
                raise KeyError(f"unknown deployment: {deployment_id}")
            index = self._deployments.index(target)
            previous = self._deployments[index - 1] if index else None
            if index == len(self._deployments) - 1:
                restore = previous.active_rule_ids if previous else ()
            else:
                restore = target.active_rule_ids
            active = set(self._current_active_ids())
            restore_set = set(restore)
            revision = self._current_revision + 1
            timestamp = _now()
            rollback_id = f'ROLLBACK-{revision:04d}-{hashlib.sha256((timestamp+deployment_id).encode()).hexdigest()[:8].upper()}'
            for rule in self._rules.values():
                if rule.rule_id in restore_set:
                    if rule.status == RuleStatus.ROLLED_BACK:
                        rule.status = RuleStatus.DEPLOYED
                elif rule.rule_id in active:
                    rule.status = RuleStatus.ROLLED_BACK
                    rule.updated_at = timestamp
            rules = [self._rules[rule_id] for rule_id in restore]
            snapshot = _Deployment(rollback_id, revision, tuple(restore), _ruleset_hash(rules), timestamp, "rollback", target.deployment_id)
            self._deployments.append(snapshot)
            self._current_revision = revision
            self._audit.append({"event": "rules_rolled_back", "deployment_id": rollback_id, "rollback_of": deployment_id, "revision": revision, "active_rule_ids": list(restore), "ruleset_sha256": snapshot.ruleset_sha256, "at": timestamp})
            return {"schema_version": DEPLOYMENT_SCHEMA_VERSION, "deployment_id": rollback_id, "revision": revision, "active_rule_ids": list(restore), "ruleset_sha256": snapshot.ruleset_sha256, "deployed_at": timestamp, "status": "rollback", "rollback_of": deployment_id}

    def active_rule_ids(self):
        return self._current_active_ids()

    def _current_active_ids(self):
        return tuple(self._deployments[-1].active_rule_ids) if self._deployments else ()

    def active_rules(self):
        return tuple(self._rules[rule_id] for rule_id in self._current_active_ids())

    def match(self, features):
        return tuple(rule.rule_id for rule in self.active_rules() if rule.matcher.get("operator") == ">=" and float(features.values.get(rule.matcher.get("feature"), 0)) >= float(rule.matcher.get("threshold", 1)))

    def get(self, rule_id):
        return self._require(rule_id)

    def audit_log(self):
        return tuple(self._audit)

    def export_snapshot(self):
        return {"schema_version": DEPLOYMENT_SCHEMA_VERSION, "manager": "phase6-rule-lifecycle-v1", "revision": self._current_revision, "rules": [rule.to_dict() for rule in self._rules.values()], "deployments": [asdict(item) for item in self._deployments], "audit": list(self._audit)}

    def ruleset_metadata(self):
        active = self.active_rules()
        return {"schema_version": RULE_SCHEMA_VERSION, "ruleset_sha256": _ruleset_hash(active), "active_rule_count": len(active), "deployment_revision": self._current_revision, "manager": "phase6-rule-lifecycle-v1"}
