"""
Automatic Security Rule Generator for WAF-ML

This module provides functionality to automatically generate security rules
based on machine learning analysis and security best practices.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Enumeration for threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleType(Enum):
    """Enumeration for different types of security rules."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    XXJSON_INJECTION = "xxjson_injection"
    RATE_LIMIT = "rate_limit"
    GEO_BLOCKING = "geo_blocking"
    BOT_PROTECTION = "bot_protection"
    CUSTOM = "custom"


@dataclass
class SecurityRule:
    """Data class representing a security rule."""
    rule_id: str
    rule_type: str
    name: str
    description: str
    pattern: str
    threat_level: str
    actions: List[str]
    enabled: bool = True
    confidence: float = 0.0
    created_at: str = ""
    last_modified: str = ""
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.last_modified:
            self.last_modified = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary format."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert rule to JSON format."""
        return json.dumps(self.to_dict(), indent=2)


class RuleGenerator:
    """
    Automatic security rule generator using ML-based recommendations.
    """

    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize the rule generator.

        Args:
            min_confidence: Minimum confidence threshold for rule generation (0-1)
        """
        self.min_confidence = min_confidence
        self.generated_rules = []
        logger.info(f"RuleGenerator initialized with confidence threshold: {min_confidence}")

    def generate_rule_id(self, rule_type: str, pattern: str) -> str:
        """
        Generate a unique rule ID based on type and pattern.

        Args:
            rule_type: Type of the security rule
            pattern: The pattern/regex for the rule

        Returns:
            Unique rule ID
        """
        content = f"{rule_type}:{pattern}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"RULE-{rule_type.upper()}-{hash_digest}"

    def analyze_threat_pattern(self, pattern: str, attack_vectors: List[str]) -> Tuple[ThreatLevel, float]:
        """
        Analyze a threat pattern to determine threat level and confidence.

        Args:
            pattern: The security pattern to analyze
            attack_vectors: List of known attack vectors

        Returns:
            Tuple of (threat_level, confidence_score)
        """
        confidence = 0.0
        threat_level = ThreatLevel.LOW

        # Simple pattern matching for threat analysis
        pattern_lower = pattern.lower()
        
        if any(vector in pattern_lower for vector in attack_vectors):
            confidence = 0.85
            threat_level = ThreatLevel.HIGH
        elif any(keyword in pattern_lower for keyword in ['union', 'select', 'inject', 'execute']):
            confidence = 0.75
            threat_level = ThreatLevel.MEDIUM
        elif any(keyword in pattern_lower for keyword in ['script', 'onclick', 'onerror']):
            confidence = 0.8
            threat_level = ThreatLevel.MEDIUM
        else:
            confidence = 0.5
            threat_level = ThreatLevel.LOW

        return threat_level, confidence

    def generate_sql_injection_rules(self) -> List[SecurityRule]:
        """
        Generate SQL injection prevention rules.

        Returns:
            List of generated SQL injection rules
        """
        rules = []
        sql_patterns = [
            {
                "name": "SQL UNION-based Injection",
                "pattern": r"(?i)(\bunion\b.*\bselect\b|\bselect\b.*\bunion\b)",
                "description": "Detects UNION-based SQL injection attempts"
            },
            {
                "name": "SQL Comment Injection",
                "pattern": r"(-{2}|/\*|\*/|;)",
                "description": "Detects SQL comment-based injection patterns"
            },
            {
                "name": "SQL Boolean-based Injection",
                "pattern": r"(?i)(\bor\b\s*1\s*=\s*1|\band\b\s*1\s*=\s*1)",
                "description": "Detects boolean-based SQL injection"
            }
        ]

        for pattern_config in sql_patterns:
            threat_level, confidence = self.analyze_threat_pattern(
                pattern_config["pattern"],
                ["union", "select", "inject", "drop", "delete"]
            )

            if confidence >= self.min_confidence:
                rule = SecurityRule(
                    rule_id=self.generate_rule_id(RuleType.SQL_INJECTION.value, pattern_config["pattern"]),
                    rule_type=RuleType.SQL_INJECTION.value,
                    name=pattern_config["name"],
                    description=pattern_config["description"],
                    pattern=pattern_config["pattern"],
                    threat_level=threat_level.value,
                    actions=["block", "log"],
                    confidence=confidence,
                    tags=["sql", "injection", "database"]
                )
                rules.append(rule)
                logger.info(f"Generated SQL injection rule: {rule.rule_id}")

        return rules

    def generate_xss_rules(self) -> List[SecurityRule]:
        """
        Generate XSS (Cross-Site Scripting) prevention rules.

        Returns:
            List of generated XSS rules
        """
        rules = []
        xss_patterns = [
            {
                "name": "Script Tag Injection",
                "pattern": r"(?i)<\s*script[^>]*>",
                "description": "Detects script tag injection attempts"
            },
            {
                "name": "Event Handler Injection",
                "pattern": r"(?i)(on\w+\s*=|javascript:|vbscript:)",
                "description": "Detects event handler-based XSS"
            },
            {
                "name": "HTML Entity Encoding Bypass",
                "pattern": r"(?i)(%3Cscript|&#x3C;|&#60;)",
                "description": "Detects encoded script tag attempts"
            }
        ]

        for pattern_config in xss_patterns:
            threat_level, confidence = self.analyze_threat_pattern(
                pattern_config["pattern"],
                ["script", "onclick", "onerror", "javascript"]
            )

            if confidence >= self.min_confidence:
                rule = SecurityRule(
                    rule_id=self.generate_rule_id(RuleType.XSS.value, pattern_config["pattern"]),
                    rule_type=RuleType.XSS.value,
                    name=pattern_config["name"],
                    description=pattern_config["description"],
                    pattern=pattern_config["pattern"],
                    threat_level=threat_level.value,
                    actions=["block", "log"],
                    confidence=confidence,
                    tags=["xss", "web", "injection"]
                )
                rules.append(rule)
                logger.info(f"Generated XSS rule: {rule.rule_id}")

        return rules

    def generate_path_traversal_rules(self) -> List[SecurityRule]:
        """
        Generate path traversal prevention rules.

        Returns:
            List of generated path traversal rules
        """
        rules = []
        path_patterns = [
            {
                "name": "Directory Traversal",
                "pattern": r"(\.\./|\.\.\\)",
                "description": "Detects directory traversal attempts"
            },
            {
                "name": "Null Byte Injection",
                "pattern": r"(%00|\\x00|\0)",
                "description": "Detects null byte injection in paths"
            },
            {
                "name": "Encoded Path Traversal",
                "pattern": r"(%2e%2e/|%2e%2e\\)",
                "description": "Detects URL-encoded path traversal"
            }
        ]

        for pattern_config in path_patterns:
            threat_level, confidence = self.analyze_threat_pattern(
                pattern_config["pattern"],
                ["traversal", "path", "directory"]
            )

            if confidence >= self.min_confidence:
                rule = SecurityRule(
                    rule_id=self.generate_rule_id(RuleType.PATH_TRAVERSAL.value, pattern_config["pattern"]),
                    rule_type=RuleType.PATH_TRAVERSAL.value,
                    name=pattern_config["name"],
                    description=pattern_config["description"],
                    pattern=pattern_config["pattern"],
                    threat_level=threat_level.value,
                    actions=["block", "log"],
                    confidence=confidence,
                    tags=["path", "traversal", "directory"]
                )
                rules.append(rule)
                logger.info(f"Generated path traversal rule: {rule.rule_id}")

        return rules

    def generate_rate_limit_rules(self) -> List[SecurityRule]:
        """
        Generate rate limiting rules.

        Returns:
            List of generated rate limit rules
        """
        rules = []
        
        rule = SecurityRule(
            rule_id=self.generate_rule_id(RuleType.RATE_LIMIT.value, "rate_limit_general"),
            rule_type=RuleType.RATE_LIMIT.value,
            name="General Rate Limiting",
            description="Rate limit requests from single IP to prevent brute force attacks",
            pattern="rate_limit",
            threat_level=ThreatLevel.MEDIUM.value,
            actions=["throttle", "log"],
            confidence=0.9,
            tags=["rate_limit", "brute_force", "dos"],
            metadata={
                "requests_per_minute": 100,
                "requests_per_hour": 5000
            }
        )
        rules.append(rule)
        logger.info(f"Generated rate limit rule: {rule.rule_id}")

        return rules

    def generate_all_rules(self) -> List[SecurityRule]:
        """
        Generate all available security rules.

        Returns:
            List of all generated security rules
        """
        logger.info("Starting generation of all security rules...")
        
        all_rules = []
        all_rules.extend(self.generate_sql_injection_rules())
        all_rules.extend(self.generate_xss_rules())
        all_rules.extend(self.generate_path_traversal_rules())
        all_rules.extend(self.generate_rate_limit_rules())
        
        self.generated_rules = all_rules
        logger.info(f"Generated {len(all_rules)} security rules total")
        
        return all_rules

    def generate_custom_rule(
        self,
        name: str,
        description: str,
        pattern: str,
        threat_level: str,
        actions: List[str],
        tags: Optional[List[str]] = None
    ) -> SecurityRule:
        """
        Generate a custom security rule.

        Args:
            name: Name of the rule
            description: Description of the rule
            pattern: Pattern/regex for the rule
            threat_level: Threat level (critical, high, medium, low, info)
            actions: List of actions (block, log, alert, etc.)
            tags: Optional tags for categorization

        Returns:
            Generated custom rule
        """
        if tags is None:
            tags = ["custom"]
        
        rule = SecurityRule(
            rule_id=self.generate_rule_id(RuleType.CUSTOM.value, pattern),
            rule_type=RuleType.CUSTOM.value,
            name=name,
            description=description,
            pattern=pattern,
            threat_level=threat_level,
            actions=actions,
            confidence=0.75,
            tags=tags
        )
        
        logger.info(f"Generated custom rule: {rule.rule_id}")
        return rule

    def export_rules(self, format: str = "json") -> str:
        """
        Export generated rules in specified format.

        Args:
            format: Export format (json, csv, yaml)

        Returns:
            Exported rules as string
        """
        if not self.generated_rules:
            logger.warning("No rules to export")
            return ""

        if format == "json":
            return json.dumps(
                [rule.to_dict() for rule in self.generated_rules],
                indent=2
            )
        else:
            logger.warning(f"Format {format} not yet implemented")
            return ""

    def filter_rules_by_threat_level(self, threat_level: str) -> List[SecurityRule]:
        """
        Filter generated rules by threat level.

        Args:
            threat_level: Threat level to filter by

        Returns:
            Filtered list of rules
        """
        return [rule for rule in self.generated_rules if rule.threat_level == threat_level]

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated rules.

        Returns:
            Dictionary containing rule statistics
        """
        if not self.generated_rules:
            return {"total_rules": 0}

        stats = {
            "total_rules": len(self.generated_rules),
            "by_type": {},
            "by_threat_level": {},
            "average_confidence": 0.0
        }

        total_confidence = 0.0
        for rule in self.generated_rules:
            # Count by type
            rule_type = rule.rule_type
            stats["by_type"][rule_type] = stats["by_type"].get(rule_type, 0) + 1

            # Count by threat level
            threat = rule.threat_level
            stats["by_threat_level"][threat] = stats["by_threat_level"].get(threat, 0) + 1

            # Accumulate confidence
            total_confidence += rule.confidence

        stats["average_confidence"] = total_confidence / len(self.generated_rules)

        return stats


def main():
    """Main function to demonstrate rule generator usage."""
    # Initialize the rule generator
    generator = RuleGenerator(min_confidence=0.7)

    # Generate all rules
    rules = generator.generate_all_rules()

    # Print statistics
    stats = generator.get_rule_statistics()
    print("\n=== Rule Generation Statistics ===")
    print(json.dumps(stats, indent=2))

    # Export rules to JSON
    print("\n=== Generated Rules (JSON) ===")
    exported_rules = generator.export_rules(format="json")
    print(exported_rules)

    # Filter high-threat rules
    print("\n=== High Threat Rules ===")
    high_threat_rules = generator.filter_rules_by_threat_level("high")
    for rule in high_threat_rules:
        print(f"- {rule.name} ({rule.rule_id})")


if __name__ == "__main__":
    main()
