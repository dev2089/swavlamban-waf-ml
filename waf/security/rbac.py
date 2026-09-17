from __future__ import annotations
from enum import StrEnum
from .auth import Actor
class Action(StrEnum):
    VIEW_EVIDENCE='view_evidence'; RECORD_FEEDBACK='record_feedback'; APPROVE_RULE='approve_rule'; DEPLOY_RULE='deploy_rule'; ROLLBACK_RULE='rollback_rule'; APPROVE_MODEL='approve_model'; PROMOTE_MODEL='promote_model'; ROLLBACK_MODEL='rollback_model'; MANAGE_RETENTION='manage_retention'; MANAGE_SECURITY='manage_security'
ROLE_PERMISSIONS={'viewer':frozenset({Action.VIEW_EVIDENCE}),'analyst':frozenset({Action.VIEW_EVIDENCE,Action.RECORD_FEEDBACK}),'rule_approver':frozenset({Action.VIEW_EVIDENCE,Action.RECORD_FEEDBACK,Action.APPROVE_RULE,Action.DEPLOY_RULE,Action.ROLLBACK_RULE}),'model_approver':frozenset({Action.VIEW_EVIDENCE,Action.RECORD_FEEDBACK,Action.APPROVE_MODEL,Action.PROMOTE_MODEL,Action.ROLLBACK_MODEL}),'admin':frozenset(Action)}
class AuthorizationError(PermissionError): pass
def can(actor: Actor, action: Action) -> bool:
    try: requested=Action(action)
    except ValueError: return False
    return requested in ROLE_PERMISSIONS.get(actor.role,frozenset())
def authorize(actor: Actor, action: Action) -> Actor:
    if not can(actor,action): raise AuthorizationError(f"role {actor.role!r} is not authorized for {Action(action).value}")
    return actor
