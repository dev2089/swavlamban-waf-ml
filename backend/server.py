"""
FastAPI Backend Server for WAF ML System
Provides REST API endpoints and WebSocket support for real-time threat detection
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_model import HybridAnomalyDetector, create_sample_data
from rule_generator import RuleGenerator, SecurityRule
from data_processor import DataProcessor
import numpy as np
from supabase import create_client, Client

# Initialize FastAPI app
app = FastAPI(
    title="WAF ML API",
    description="Advanced Web Application Firewall with Machine Learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize ML components
ml_detector = None
rule_generator = RuleGenerator(min_confidence=0.7)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Request Models
class ThreatAnalysisRequest(BaseModel):
    method: str
    uri: str
    source_ip: str
    headers: Dict[str, str] = {}
    query_params: Dict[str, str] = {}
    body: Optional[str] = None
    user_agent: Optional[str] = None

class RuleCreationRequest(BaseModel):
    name: str
    description: str
    pattern: str
    threat_level: str
    actions: List[str]
    tags: List[str] = []

# Helper Functions
def extract_features(request: ThreatAnalysisRequest) -> np.ndarray:
    """Extract features from request for ML model"""
    features = []

    # Basic numeric features
    features.append(len(request.uri))
    features.append(len(request.body or ""))
    features.append(len(request.headers))
    features.append(len(request.query_params))

    # Pattern-based features
    suspicious_patterns = ['union', 'select', 'script', 'alert', '../', '<', '>']
    uri_lower = request.uri.lower()
    body_lower = (request.body or "").lower()

    for pattern in suspicious_patterns:
        features.append(1 if pattern in uri_lower else 0)
        features.append(1 if pattern in body_lower else 0)

    # Pad to required length (15 features)
    while len(features) < 15:
        features.append(0)

    return np.array(features).reshape(1, -1)

async def log_threat_to_db(threat_data: dict):
    """Log detected threat to database"""
    try:
        result = supabase.table('threats').insert(threat_data).execute()

        # Also create an alert for high/critical threats
        if threat_data['severity'] in ['high', 'critical']:
            alert_data = {
                'alert_type': 'threat_detected',
                'severity': threat_data['severity'],
                'title': f"{threat_data['threat_type']} Detected",
                'message': f"Threat detected from {threat_data['source_ip']}",
                'threat_id': result.data[0]['id'] if result.data else None,
                'metadata': {'confidence': threat_data['confidence']}
            }
            supabase.table('alerts').insert(alert_data).execute()

        # Broadcast to WebSocket clients
        await manager.broadcast({
            'type': 'threat_detected',
            'data': threat_data
        })
    except Exception as e:
        print(f"Error logging threat: {e}")

async def log_request_to_db(request_data: dict):
    """Log request to database"""
    try:
        supabase.table('request_logs').insert(request_data).execute()
    except Exception as e:
        print(f"Error logging request: {e}")

async def update_analytics(metric_name: str, value: float):
    """Update analytics metrics"""
    try:
        analytics_data = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_type': 'counter',
            'timestamp': datetime.utcnow().isoformat()
        }
        supabase.table('analytics').insert(analytics_data).execute()
    except Exception as e:
        print(f"Error updating analytics: {e}")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global ml_detector
    print("Initializing ML models...")

    try:
        # Create sample training data
        X_train, _ = create_sample_data(n_samples=500, n_features=15, anomaly_ratio=0.1)

        # Initialize and train hybrid detector
        ml_detector = HybridAnomalyDetector(if_contamination=0.1, ae_encoding_dim=8)
        ml_detector.fit(X_train, ae_epochs=30)

        print("ML models initialized successfully")
    except Exception as e:
        print(f"Error initializing ML models: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "WAF ML API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_models": "loaded" if ml_detector else "not loaded",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/analyze")
async def analyze_request(
    request: ThreatAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze request for threats using ML"""
    try:
        # Extract features
        features = extract_features(request)

        # Get ML predictions
        ml_scores = ml_detector.get_anomaly_scores(features)
        prediction = ml_detector.predict(features, voting='hard')[0]

        # Determine threat level
        is_threat = prediction == -1
        confidence = float(abs(ml_scores['autoencoder'][0]))

        # Determine threat type based on patterns
        threat_type = "UNKNOWN"
        if 'union' in request.uri.lower() or 'select' in request.uri.lower():
            threat_type = "SQL_INJECTION"
        elif 'script' in request.uri.lower() or 'alert' in request.uri.lower():
            threat_type = "XSS"
        elif '../' in request.uri:
            threat_type = "PATH_TRAVERSAL"
        elif is_threat:
            threat_type = "ANOMALY"

        # Determine severity
        if confidence > 0.9:
            severity = "critical"
        elif confidence > 0.7:
            severity = "high"
        elif confidence > 0.5:
            severity = "medium"
        else:
            severity = "low"

        # Prepare response
        analysis_result = {
            "threat_detected": is_threat,
            "threat_type": threat_type,
            "severity": severity,
            "confidence": confidence,
            "should_block": is_threat and confidence > 0.7,
            "ml_scores": {
                "isolation_forest": float(ml_scores['isolation_forest'][0]),
                "autoencoder": float(ml_scores['autoencoder'][0]),
                "reconstruction_error": float(ml_scores['reconstruction_error'][0])
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log threat if detected
        if is_threat:
            threat_data = {
                "threat_type": threat_type,
                "severity": severity,
                "source_ip": request.source_ip,
                "target_endpoint": request.uri,
                "payload": request.body,
                "confidence": confidence,
                "blocked": analysis_result["should_block"],
                "metadata": {"ml_scores": analysis_result["ml_scores"]},
                "detected_at": datetime.utcnow().isoformat()
            }
            background_tasks.add_task(log_threat_to_db, threat_data)

        # Log request
        request_log = {
            "request_id": f"{request.source_ip}-{datetime.utcnow().timestamp()}",
            "method": request.method,
            "uri": request.uri,
            "source_ip": request.source_ip,
            "user_agent": request.user_agent,
            "headers": request.headers,
            "query_params": request.query_params,
            "body": request.body,
            "threat_detected": is_threat,
            "blocked": analysis_result["should_block"],
            "ml_scores": analysis_result["ml_scores"],
            "timestamp": datetime.utcnow().isoformat()
        }
        background_tasks.add_task(log_request_to_db, request_log)

        # Update analytics
        background_tasks.add_task(update_analytics, "threat_count", 1 if is_threat else 0)

        return analysis_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules")
async def get_rules():
    """Get all security rules"""
    try:
        result = supabase.table('security_rules').select("*").execute()
        return {"rules": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rules/generate")
async def generate_rules():
    """Generate security rules automatically"""
    try:
        # Generate rules
        rules = rule_generator.generate_all_rules()

        # Insert into database
        for rule in rules:
            rule_data = {
                "rule_id": rule.rule_id,
                "rule_type": rule.rule_type,
                "name": rule.name,
                "description": rule.description,
                "pattern": rule.pattern,
                "threat_level": rule.threat_level,
                "actions": rule.actions,
                "enabled": rule.enabled,
                "confidence": rule.confidence,
                "tags": rule.tags,
                "metadata": rule.metadata
            }
            supabase.table('security_rules').upsert(rule_data, on_conflict='rule_id').execute()

        return {
            "message": "Rules generated successfully",
            "count": len(rules),
            "rules": [rule.to_dict() for rule in rules]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rules")
async def create_rule(rule: RuleCreationRequest):
    """Create a new security rule"""
    try:
        custom_rule = rule_generator.generate_custom_rule(
            name=rule.name,
            description=rule.description,
            pattern=rule.pattern,
            threat_level=rule.threat_level,
            actions=rule.actions,
            tags=rule.tags
        )

        rule_data = {
            "rule_id": custom_rule.rule_id,
            "rule_type": custom_rule.rule_type,
            "name": custom_rule.name,
            "description": custom_rule.description,
            "pattern": custom_rule.pattern,
            "threat_level": custom_rule.threat_level,
            "actions": custom_rule.actions,
            "enabled": custom_rule.enabled,
            "confidence": custom_rule.confidence,
            "tags": custom_rule.tags,
            "metadata": custom_rule.metadata
        }

        result = supabase.table('security_rules').insert(rule_data).execute()
        return {"rule": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/threats")
async def get_threats(limit: int = 100):
    """Get recent threats"""
    try:
        result = supabase.table('threats').select("*").order('detected_at', desc=True).limit(limit).execute()
        return {"threats": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        threats_result = supabase.table('threats').select("*", count='exact').execute()
        rules_result = supabase.table('security_rules').select("*", count='exact').eq('enabled', True).execute()
        logs_result = supabase.table('request_logs').select("blocked", count='exact').execute()

        blocked_count = sum(1 for log in logs_result.data if log.get('blocked'))

        return {
            "total_threats": threats_result.count,
            "active_rules": rules_result.count,
            "blocked_requests": blocked_count,
            "total_requests": logs_result.count,
            "detection_rate": (blocked_count / logs_result.count * 100) if logs_result.count > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
