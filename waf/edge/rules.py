from __future__ import annotations
import re
from waf.core.models import DetectionSignal, FeatureVector, RequestEnvelope
from waf.features.http_v2 import normalize_target
from waf.rules.lifecycle import RuleLifecycleManager

class OpenSourceWAFRuleEngine:
    """Deterministic signatures plus approved Phase 6 managed rules."""
    name='open-source-waf-rules'
    RULES=(
      ('WAF-SQL-001',re.compile(r'(?:union\s+select|or\s+\d\s*=\s*\d|drop\s+table)',re.I),'SQL injection signature'),
      ('WAF-XSS-001',re.compile(r'<\s*script\b|javascript\s*:|on(?:error|load|click)\s*=',re.I),'XSS signature'),
      ('WAF-TRAV-001',re.compile(r'(?:\.\./|\.\.\\)',re.I),'path traversal signature'),
      ('WAF-CMD-001',re.compile(r'(?:;\s*(?:cat|id|uname|curl|wget|bash|sh)\b|\$\(|`[^`]+`|&&|\|\|)',re.I),'command injection signature'))
    def __init__(self,rule_lifecycle:RuleLifecycleManager|None=None): self.rule_lifecycle=rule_lifecycle or RuleLifecycleManager()
    def detect(self,request:RequestEnvelope,features:FeatureVector)->DetectionSignal:
        target=normalize_target(request.path,request.query)+'\n'+request.body.decode('utf-8',errors='replace'); reasons=[]; ids=[]
        for rid,pat,reason in self.RULES:
            if pat.search(target): ids.append(rid); reasons.append(reason)
        for rid in self.rule_lifecycle.match(features): ids.append(rid); reasons.append(f'managed rule: {self.rule_lifecycle.get(rid).name}')
        matched=bool(ids)
        return DetectionSignal(self.name,1.0 if matched else 0.0,0.99 if matched else 0.10,tuple(reasons),tuple(dict.fromkeys(ids)),{'ruleset':'builtin-open-source-waf-v2','feature_schema':features.schema_version,'managed_ruleset':self.rule_lifecycle.ruleset_metadata()})
