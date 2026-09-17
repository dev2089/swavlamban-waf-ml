"""Production FastAPI adapter for the verified WAF edge and security controls."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF
from waf.security.production_security import AuthError, SecurityConfig, authorize, parse_bearer_token, security_headers, validate_production_config
from waf.storage.production import MemorySecurityStore, SupabaseRESTStore


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = Field(min_length=1, max_length=16)
    uri: str = Field(min_length=1, max_length=8192)
    source_ip: str | None = Field(default=None, max_length=128)
    headers: dict[str, str] = Field(default_factory=dict)
    query: str = Field(default="", max_length=8192)
    body: str = Field(default="", max_length=1_048_576)


class RuleApproval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approver: str = Field(min_length=1, max_length=128)


def _request_id(value: str | None) -> str:
    if value and 8 <= len(value) <= 128 and all(c.isalnum() or c in "._:-" for c in value):
        return value
    return str(uuid.uuid4())


def create_app(*, env: dict[str, str] | None = None) -> FastAPI:
    source = os.environ if env is None else env
    security = SecurityConfig.from_env(source)
    findings = validate_production_config(security)
    if security.environment in {"production", "prod"} and findings:
        raise RuntimeError("production security configuration rejected: " + "; ".join(findings))

    cors = list(security.allowed_origins)
    app = FastAPI(title="Swavlamban WAF ML API", version="10.0.0", docs_url=None if security.environment in {"production", "prod"} else "/docs")
    if cors:
        app.add_middleware(CORSMiddleware, allow_origins=cors, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"])

    waf = EdgeWAF(WAFConfig.from_env(source))
    store: Any
    if security.environment in {"production", "prod"}:
        store = SupabaseRESTStore(security.supabase_url, security.supabase_service_role_key)
    else:
        store = MemorySecurityStore()
    app.state.waf = waf
    app.state.store = store
    app.state.security = security

    dashboard_dir = Path(__file__).resolve().parents[2] / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    @app.middleware("http")
    async def harden_response(request: Request, call_next):
        response = await call_next(request)
        for name, value in security_headers().items():
            response.headers.setdefault(name, value)
        response.headers.setdefault("X-Request-ID", request.headers.get("X-Request-ID", ""))
        return response

    def principal(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        try:
            claims = parse_bearer_token(authorization or "", secret=security.auth_secret, issuer=security.issuer, audience=security.audience, clock_skew_seconds=security.clock_skew_seconds)
            return claims
        except AuthError as exc:
            raise HTTPException(status_code=401, detail="authentication required") from exc

    def require(permission: str):
        def dependency(claims: dict[str, Any] = Depends(principal)) -> dict[str, Any]:
            try:
                authorize(claims, permission)
                return claims
            except AuthError as exc:
                raise HTTPException(status_code=403, detail="forbidden") from exc
        return dependency

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "healthy", "service": "swavlamban-waf-api", "pipeline_version": WAFConfig.from_env(source).pipeline_version, "security_config_valid": not findings, "phase": 10}

    @app.get("/api/release")
    async def release(claims: dict[str, Any] = Depends(require("read:stats"))) -> dict[str, Any]:
        return {
            "phase": 10,
            "service": "swavlamban-waf-api",
            "status": "release-candidate",
            "evidence": {
                "full_regression": "82/82 PASS (Phase 9 baseline)",
                "live_supabase": "schema/RLS/privileges verified",
                "local_tls": "allow 200; SQL block 403",
                "public_https": "not claimed",
                "modsecurity_coraza": "not claimed",
                "internet_scale": "not claimed",
            },
        }

    @app.get("/api/security/me")
    async def me(claims: dict[str, Any] = Depends(principal)) -> dict[str, str]:
        return {"subject": str(claims["sub"]), "role": str(claims["role"])}

    @app.post("/api/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request, claims: dict[str, Any] = Depends(require("analyze:requests"))) -> dict[str, Any]:
        rid = _request_id(request.headers.get("X-Request-ID"))
        body = payload.body.encode("utf-8")
        config = WAFConfig.from_env(source)
        if len(body) > config.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body exceeds configured limit")
        envelope = RequestEnvelope(rid, payload.method.upper(), request.url.scheme, request.url.hostname or "localhost", payload.uri, payload.query, payload.headers, body, payload.source_ip, time.time())
        try:
            result = waf.analyze(envelope)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid security input") from exc
        evidence = result.evidence
        response = {
            "request_id": rid,
            "decision": result.decision.value,
            "blocked": result.decision is Decision.BLOCK,
            "risk_score": result.risk_score,
            "reasons": list(result.reasons),
            "rule_ids": list(result.rule_ids),
            "detectors": [signal.detector for signal in result.signals],
            "evidence": {
                "schema_version": evidence.schema_version if evidence else None,
                "explanation": evidence.explanation if evidence else None,
                "versions": dict(evidence.versions) if evidence else {},
                "privacy": dict(evidence.privacy) if evidence else {},
            },
        }
        safe_summary = {"threat_detected": result.decision is not Decision.ALLOW, "threat_type": "WAF_DECISION", "severity": "high" if result.decision is Decision.BLOCK else "low", "risk_score": result.risk_score, "blocked": result.decision is Decision.BLOCK}
        try:
            store.record_decision(request_id=rid, source_ip=payload.source_ip, method=payload.method.upper(), uri=payload.uri, result=safe_summary)
            store.record_audit(actor=str(claims["sub"]), action="analyze:requests", target=rid, outcome=result.decision.value, request_id=rid)
        except Exception as exc:
            if security.environment in {"production", "prod"}:
                raise HTTPException(status_code=503, detail="security storage unavailable") from exc
        return response

    @app.get("/api/threats")
    async def threats(limit: int = 100, claims: dict[str, Any] = Depends(require("read:threats"))) -> dict[str, Any]:
        return {"threats": store.recent_threats(limit)} if hasattr(store, "recent_threats") else {"threats": []}

    @app.get("/api/stats")
    async def stats(claims: dict[str, Any] = Depends(require("read:stats"))) -> dict[str, Any]:
        return store.stats() if hasattr(store, "stats") else {"status": "remote_store"}

    @app.post("/api/rules/{rule_id}/approve")
    async def approve_rule(rule_id: str, payload: RuleApproval, request: Request, claims: dict[str, Any] = Depends(require("approve:rules"))) -> dict[str, Any]:
        try:
            approved = waf.approve_rule(rule_id, payload.approver)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="rule cannot be approved in current state") from exc
        rid = _request_id(request.headers.get("X-Request-ID"))
        try:
            store.record_audit(actor=str(claims["sub"]), action="approve:rules", target=rule_id, outcome="approved", request_id=rid)
        except Exception as exc:
            if security.environment in {"production", "prod"}:
                raise HTTPException(status_code=503, detail="security storage unavailable") from exc
        return approved.to_dict()

    @app.post("/api/rules/deploy")
    async def deploy_rules(request: Request, claims: dict[str, Any] = Depends(require("manage:deployments"))) -> dict[str, Any]:
        try:
            result = waf.deploy_approved_rules()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="no approved rules ready for deployment") from exc
        rid = _request_id(request.headers.get("X-Request-ID"))
        try:
            store.record_audit(actor=str(claims["sub"]), action="manage:deployments", target=result["deployment_id"], outcome="deployed", request_id=rid)
        except Exception as exc:
            if security.environment in {"production", "prod"}:
                raise HTTPException(status_code=503, detail="security storage unavailable") from exc
        return result

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        auth_header = websocket.headers.get("authorization", "")
        try:
            claims = parse_bearer_token(auth_header, secret=security.auth_secret, issuer=security.issuer, audience=security.audience, clock_skew_seconds=security.clock_skew_seconds)
            authorize(claims, "read:threats")
        except AuthError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    return app


app = create_app()
