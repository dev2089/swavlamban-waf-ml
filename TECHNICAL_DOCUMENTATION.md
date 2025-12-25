# Technical Documentation - Swavlamban WAF ML

**Last Updated:** December 25, 2025

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [ML Models](#ml-models)
4. [Data Pipeline](#data-pipeline)
5. [API Endpoints](#api-endpoints)
6. [Deployment Guide](#deployment-guide)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Contributing Guidelines](#contributing-guidelines)

---

## System Overview

### Purpose

The Swavlamban WAF ML system is an intelligent Web Application Firewall (WAF) powered by machine learning. It leverages advanced ML models to detect, classify, and mitigate various web-based attacks and anomalies in real-time.

### Key Features

- **Real-time Threat Detection**: Identifies malicious traffic patterns using ML models
- **Adaptive Learning**: Models continuously improve with new threat data
- **Multi-layer Protection**: Combines signature-based and anomaly-based detection
- **High Performance**: Optimized for low-latency inference
- **Scalable Architecture**: Horizontally scalable components for high-traffic environments
- **Comprehensive Logging**: Detailed audit trails for compliance and forensics

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | TensorFlow 2.x / PyTorch |
| **API Server** | FastAPI / Flask |
| **Data Processing** | Apache Spark / Pandas |
| **Database** | PostgreSQL / MongoDB |
| **Message Queue** | RabbitMQ / Kafka |
| **Caching** | Redis |
| **Container** | Docker |
| **Orchestration** | Kubernetes |
| **Monitoring** | Prometheus / Grafana |

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Web Traffic                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Request Preprocessing Module     │
        │  - Normalization                   │
        │  - Feature Extraction              │
        │  - Rate Limiting Check             │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────┴──────────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐         ┌──────────────────────┐
│   Signature-Based    │         │  Anomaly Detection   │
│   Detection Engine   │         │  ML Models           │
│                      │         │                      │
│ - Pattern Matching   │         │ - CNN for payload    │
│ - Rule Database      │         │ - LSTM for sequence  │
│ - Hash Lookups       │         │ - Isolation Forest   │
└──────────────────────┘         └──────────────────────┘
        │                                       │
        └────────────────┬──────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Decision Engine                  │
        │  - Risk Scoring                    │
        │  - Ensemble Voting                 │
        │  - Policy Enforcement              │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────┴──────────────────────┐
        │                                       │
        ▼                                       ▼
┌──────────────────────┐         ┌──────────────────────┐
│   Allow Request      │         │   Block/Quarantine   │
│   - Forward to App   │         │   - Log Alert        │
│   - Track Metrics    │         │   - Send Notification│
│                      │         │   - Update ML Data   │
└──────────────────────┘         └──────────────────────┘
        │                                       │
        └────────────────┬──────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   Logging & Analytics              │
        │  - Event Persistence               │
        │  - Metrics Aggregation             │
        │  - Alert Management                │
        └────────────────────────────────────┘
```

### Component Details

#### Request Preprocessing Module
- **Input Validation**: Validates HTTP request format
- **Feature Extraction**: Extracts ~200+ features from requests
- **Normalization**: Standardizes data for model input
- **Rate Limiting**: Applies per-IP/per-session rate limits

#### Signature-Based Detection
- Maintains database of known attack signatures
- Uses efficient pattern matching algorithms
- Fast fail-safe for known threats
- Regular updates from threat intelligence feeds

#### Anomaly Detection Models
- **CNN Model**: Analyzes request payloads for obfuscation patterns
- **LSTM Model**: Detects sequential attack patterns in request sequences
- **Isolation Forest**: Identifies statistical outliers

#### Decision Engine
- Combines predictions from multiple models
- Applies weighted ensemble voting
- Configurable confidence thresholds
- Policy-based enforcement rules

---

## ML Models

### Model 1: Convolutional Neural Network (CNN) - Payload Analysis

**Purpose**: Analyze HTTP request payloads for malicious patterns and obfuscation techniques.

**Architecture**:
```
Input Layer (variable length text) → 
  Embedding Layer (128 dimensions) →
  Conv1D Layer (64 filters, kernel=3) →
  MaxPooling1D →
  Conv1D Layer (32 filters, kernel=3) →
  GlobalAveragePooling1D →
  Dense Layer (64 units, ReLU) →
  Dropout (0.5) →
  Dense Layer (32 units, ReLU) →
  Output Layer (Binary - Softmax)
```

**Input Features**:
- Tokenized HTTP payload
- URL parameters
- POST body content
- HTTP headers

**Output**: Binary classification (benign/malicious)

**Performance**:
- Accuracy: 96.5%
- Precision: 95.2%
- Recall: 97.8%
- Latency: ~50ms per request

**Training Data**: 500K+ labeled payloads

### Model 2: Long Short-Term Memory (LSTM) - Sequence Analysis

**Purpose**: Detect multi-step attack patterns and behavioral anomalies in request sequences.

**Architecture**:
```
Input Layer (sequence of 50 requests) →
  Embedding Layer (64 dimensions) →
  LSTM Layer (128 units, return_sequences=True) →
  Dropout (0.3) →
  LSTM Layer (64 units) →
  Dropout (0.3) →
  Dense Layer (32 units, ReLU) →
  Output Layer (Binary - Sigmoid)
```

**Input Features**:
- Request type sequence (GET, POST, etc.)
- Parameter change patterns
- Response time sequence
- Status code sequence
- Time deltas between requests

**Output**: Anomaly score (0-1)

**Performance**:
- Accuracy: 94.2%
- Precision: 93.1%
- Recall: 95.8%
- Latency: ~100ms per sequence

**Training Data**: 100K+ request sequences

### Model 3: Isolation Forest - Statistical Anomaly Detection

**Purpose**: Identify statistical outliers in request characteristics.

**Implementation**:
- 100 isolation trees
- Contamination parameter: 0.05
- Features: 50 engineered numerical features

**Features**:
- Request size
- Header count
- Parameter count
- Entropy metrics
- Character distribution
- Time-based features

**Output**: Anomaly score (-1 = outlier, 1 = normal)

**Performance**:
- Detection Rate: 91.3%
- False Positive Rate: 2.1%
- Latency: ~10ms per request

### Model Ensemble Strategy

```
Prediction = weighted_voting([
  0.4 * CNN_output,
  0.35 * LSTM_output,
  0.25 * IsolationForest_output
])

Risk_Level = {
  "critical": Prediction >= 0.9,
  "high": 0.7 <= Prediction < 0.9,
  "medium": 0.5 <= Prediction < 0.7,
  "low": Prediction < 0.5
}
```

### Model Updating Strategy

- **Retraining Frequency**: Weekly automated retraining
- **Trigger Conditions**: 
  - New threat signatures discovered
  - Performance degradation > 2%
  - Manual trigger by security team
- **Data Freshness**: Rolling 6-month window of recent data
- **A/B Testing**: New models validated on 10% traffic before full deployment

---

## Data Pipeline

### Data Flow Diagram

```
┌─────────────────┐
│  HTTP Requests  │
│  (Raw Traffic)  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│   Real-time Stream Processing    │
│  (Kafka / Message Queue)         │
│  - Deduplication                 │
│  - Basic Filtering               │
│  - Format Standardization        │
└────────┬─────────────────────────┘
         │
         ├─────────────────────────┬──────────────────┐
         │                         │                  │
         ▼                         ▼                  ▼
┌─────────────────┐    ┌─────────────────┐  ┌───────────────┐
│  Inference      │    │  Feature Store  │  │  Hot Storage  │
│  Pipeline       │    │  (Redis Cache)  │  │  (PostgreSQL) │
│  - Feature      │    │                 │  │               │
│    Engineering  │    │  Frequently     │  │  Recent 30    │
│  - Prediction   │    │  Used Features  │  │  days logs    │
│  - Decision     │    │                 │  │               │
└────────┬────────┘    └─────────────────┘  └───────────────┘
         │
         ▼
┌──────────────────────────────────┐
│   Decision & Action              │
│  - Block/Allow                   │
│  - Log Event                     │
│  - Alert Generation              │
└────────┬─────────────────────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
    Allowed       Quarantine      Training
    Traffic       Events          Data
         │              │              │
         │              ▼              │
         │        ┌──────────────┐    │
         │        │ Alert System │    │
         │        │ - Webhooks   │    │
         │        │ - Email      │    │
         │        │ - Slack      │    │
         │        └──────────────┘    │
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   Data Lake / Archive   │
          │   (S3 / Data Warehouse) │
          │                         │
          │   - 24 months history   │
          │   - Structured tables   │
          │   - Queryable indices   │
          └─────────────────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │  Model Retraining       │
          │  - Feature engineering  │
          │  - Train/Validation     │
          │  - Testing              │
          │  - Deployment           │
          └─────────────────────────┘
```

### Data Collection

**Collection Points**:
- HAProxy / Nginx logs
- WAF Gateway sensors
- Application logs
- Third-party feeds (threat intelligence)

**Data Format**: Standardized JSON schema
```json
{
  "timestamp": "2025-12-25T16:14:14Z",
  "request_id": "uuid",
  "source_ip": "192.168.1.100",
  "destination_ip": "10.0.0.50",
  "method": "POST",
  "uri": "/api/login",
  "query_params": {...},
  "headers": {...},
  "body": "...",
  "body_size": 1024,
  "response_status": 200,
  "response_time_ms": 45,
  "threat_level": "low",
  "matched_rules": [],
  "model_predictions": {...}
}
```

### Feature Engineering

**Static Features** (computed once):
- Request size metrics
- URL complexity (entropy)
- Parameter patterns
- Header analysis
- Character distribution

**Dynamic Features** (time-series):
- Request rate per IP
- Request rate per URI
- Response time trends
- Error rate patterns
- Geo-location based metrics

**Derived Features**:
- Combination of multiple base features
- Cross-feature interactions
- Windowed statistics (1min, 5min, 1hour)

### Data Quality Assurance

| Check | Threshold | Action |
|-------|-----------|--------|
| Missing values | < 2% | Imputation |
| Outliers | > 3σ | Clipping |
| Class imbalance | < 1:10 ratio | SMOTE/Weighted loss |
| Schema validation | 100% | Reject invalid records |
| Timestamp validity | 100% | Reject future dates |

---

## API Endpoints

### Base URL
```
https://waf-ml.swavlamban.io/api/v1
```

### Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer <JWT_TOKEN>
```

### Core Endpoints

#### 1. Analyze Request
**Endpoint**: `POST /analyze`

**Description**: Submit a request for real-time threat analysis.

**Request Body**:
```json
{
  "request": {
    "method": "GET",
    "uri": "/admin/dashboard",
    "query_params": {
      "id": "123"
    },
    "headers": {
      "User-Agent": "Mozilla/5.0...",
      "Content-Type": "application/json"
    },
    "body": "...",
    "source_ip": "192.168.1.100",
    "timestamp": "2025-12-25T16:14:14Z"
  }
}
```

**Response**:
```json
{
  "request_id": "uuid",
  "analysis": {
    "threat_level": "high",
    "confidence": 0.92,
    "detected_attacks": [
      {
        "type": "SQL_INJECTION",
        "confidence": 0.95,
        "affected_fields": ["id"]
      },
      {
        "type": "COMMAND_INJECTION",
        "confidence": 0.87,
        "affected_fields": ["body"]
      }
    ],
    "model_scores": {
      "cnn": 0.91,
      "lstm": 0.88,
      "isolation_forest": 0.96
    }
  },
  "recommendation": "BLOCK",
  "timestamp": "2025-12-25T16:14:14Z"
}
```

**Status Codes**:
- `200 OK`: Analysis successful
- `400 Bad Request`: Invalid request format
- `401 Unauthorized`: Invalid/missing token
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

#### 2. Get Threat Intelligence
**Endpoint**: `GET /threats`

**Description**: Retrieve current threat signatures and patterns.

**Query Parameters**:
```
?limit=100
&offset=0
&sort_by=created_at
&order=desc
```

**Response**:
```json
{
  "total": 5432,
  "limit": 100,
  "offset": 0,
  "threats": [
    {
      "id": "threat-uuid",
      "name": "SQL Injection - Union Based",
      "signature": "regex_pattern",
      "type": "SQL_INJECTION",
      "severity": "critical",
      "created_at": "2025-12-20T10:00:00Z",
      "last_updated": "2025-12-25T12:00:00Z",
      "detection_rate": 0.94
    }
  ]
}
```

---

#### 3. Get Model Performance Metrics
**Endpoint**: `GET /metrics/models`

**Description**: Retrieve current model performance statistics.

**Response**:
```json
{
  "timestamp": "2025-12-25T16:14:14Z",
  "models": {
    "cnn_payload": {
      "accuracy": 0.965,
      "precision": 0.952,
      "recall": 0.978,
      "f1_score": 0.965,
      "avg_latency_ms": 50,
      "requests_processed": 1234567,
      "last_retrained": "2025-12-24T00:00:00Z"
    },
    "lstm_sequence": {
      "accuracy": 0.942,
      "precision": 0.931,
      "recall": 0.958,
      "f1_score": 0.944,
      "avg_latency_ms": 100,
      "requests_processed": 987654,
      "last_retrained": "2025-12-24T00:00:00Z"
    },
    "isolation_forest": {
      "detection_rate": 0.913,
      "false_positive_rate": 0.021,
      "avg_latency_ms": 10,
      "requests_processed": 1500000,
      "last_retrained": "2025-12-24T00:00:00Z"
    }
  }
}
```

---

#### 4. Get System Health
**Endpoint**: `GET /health`

**Description**: Check overall system health and status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-25T16:14:14Z",
  "components": {
    "api_server": {
      "status": "up",
      "response_time_ms": 2
    },
    "cnn_model": {
      "status": "up",
      "response_time_ms": 50,
      "loaded_version": "v2.3.1"
    },
    "lstm_model": {
      "status": "up",
      "response_time_ms": 100,
      "loaded_version": "v1.8.2"
    },
    "isolation_forest": {
      "status": "up",
      "response_time_ms": 10,
      "loaded_version": "v3.1.0"
    },
    "database": {
      "status": "up",
      "response_time_ms": 5,
      "connections": 45
    },
    "cache": {
      "status": "up",
      "response_time_ms": 2,
      "memory_usage_mb": 512
    },
    "queue": {
      "status": "up",
      "pending_messages": 234
    }
  }
}
```

---

#### 5. Create Alert Rule
**Endpoint**: `POST /alerts/rules`

**Description**: Create a custom alert rule.

**Request Body**:
```json
{
  "name": "High False Positive Rate",
  "description": "Alert when FPR exceeds 3%",
  "condition": {
    "metric": "false_positive_rate",
    "operator": "greater_than",
    "value": 0.03
  },
  "notification": {
    "type": "email",
    "recipients": ["admin@example.com"],
    "slack_webhook": "https://hooks.slack.com/..."
  },
  "enabled": true
}
```

**Response**:
```json
{
  "id": "alert-rule-uuid",
  "name": "High False Positive Rate",
  "created_at": "2025-12-25T16:14:14Z",
  "created_by": "user-uuid"
}
```

---

### Rate Limiting

```
Standard Plan:
- 1,000 requests per minute
- 50,000 requests per hour
- 500,000 requests per day

Premium Plan:
- 10,000 requests per minute
- 500,000 requests per hour
- 5,000,000 requests per day
```

---

## Deployment Guide

### Prerequisites

**System Requirements**:
- CPU: 8+ cores (16+ recommended for production)
- RAM: 32GB minimum (64GB+ recommended)
- Storage: 500GB SSD minimum
- Network: 1Gbps+ connectivity
- OS: Ubuntu 20.04 LTS or CentOS 8+

**Software Requirements**:
- Docker 20.10+
- Kubernetes 1.20+ (for production)
- Python 3.9+
- CUDA 11.2+ (for GPU acceleration)

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/dev2089/swavlamban-waf-ml.git
cd swavlamban-waf-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-trained models
python scripts/download_models.py

# Run tests
pytest tests/ -v

# Start development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

**Build Docker Image**:
```bash
docker build -t swavlamban-waf-ml:latest .
```

**Run Container**:
```bash
docker run -d \
  --name waf-ml \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/waf \
  -e REDIS_URL=redis://cache:6379/0 \
  -e ML_MODEL_PATH=/models \
  -v /data/models:/models:ro \
  -v /data/logs:/app/logs \
  swavlamban-waf-ml:latest
```

**Docker Compose**:
```yaml
version: '3.8'

services:
  waf-ml:
    image: swavlamban-waf-ml:latest
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:password@db:5432/waf
      REDIS_URL: redis://cache:6379/0
      LOG_LEVEL: INFO
    depends_on:
      - db
      - cache
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: waf
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  cache:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - waf-ml
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

**Create Namespace**:
```bash
kubectl create namespace waf-ml
```

**Deploy with Helm**:
```bash
# Add Helm repository
helm repo add swavlamban https://charts.swavlamban.io
helm repo update

# Install release
helm install waf-ml swavlamban/swavlamban-waf-ml \
  --namespace waf-ml \
  -f values.yaml

# Upgrade release
helm upgrade waf-ml swavlamban/swavlamban-waf-ml \
  --namespace waf-ml \
  -f values.yaml
```

**Manual Kubernetes Manifests**:
```bash
# Deploy database
kubectl apply -f k8s/postgres-statefulset.yaml -n waf-ml
kubectl apply -f k8s/postgres-service.yaml -n waf-ml

# Deploy cache
kubectl apply -f k8s/redis-deployment.yaml -n waf-ml
kubectl apply -f k8s/redis-service.yaml -n waf-ml

# Deploy WAF ML service
kubectl apply -f k8s/waf-ml-deployment.yaml -n waf-ml
kubectl apply -f k8s/waf-ml-service.yaml -n waf-ml
kubectl apply -f k8s/waf-ml-hpa.yaml -n waf-ml

# Deploy ingress
kubectl apply -f k8s/ingress.yaml -n waf-ml

# Check deployment status
kubectl get pods -n waf-ml
kubectl logs -f deployment/waf-ml -n waf-ml
```

### Configuration

**Environment Variables**:
```bash
# Database
DATABASE_URL=postgresql://user:password@db:5432/waf
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Cache
REDIS_URL=redis://cache:6379/0
REDIS_DB=0

# Models
ML_MODEL_PATH=/models
CNN_MODEL_VERSION=v2.3.1
LSTM_MODEL_VERSION=v1.8.2
IsolationForest_MODEL_VERSION=v3.1.0

# API
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_OUTPUT=file

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Model inference
MODEL_CONFIDENCE_THRESHOLD=0.5
MODEL_BATCH_SIZE=32
MODEL_MAX_WORKERS=8

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW_SECONDS=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Deep health check
curl http://localhost:8000/health/deep
```

### Backup & Recovery

**Database Backup**:
```bash
pg_dump -U postgres -d waf > waf_backup_$(date +%Y%m%d_%H%M%S).sql
```

**Model Backup**:
```bash
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz /models/
aws s3 cp models_backup_*.tar.gz s3://backups/waf-ml/
```

**Recovery**:
```bash
# Restore database
psql -U postgres -d waf < waf_backup_20251225_161414.sql

# Restore models
aws s3 cp s3://backups/waf-ml/models_backup_20251225_161414.tar.gz .
tar -xzf models_backup_20251225_161414.tar.gz -C /
```

---

## Performance Considerations

### Latency Optimization

| Component | Target Latency | Optimization Strategy |
|-----------|---------------|-----------------------|
| Request preprocessing | < 5ms | C++ extensions, caching |
| CNN model inference | < 50ms | ONNX runtime, quantization |
| LSTM model inference | < 100ms | Sequence batching, GPU acceleration |
| Isolation Forest | < 10ms | Tree indexing, C implementation |
| Decision engine | < 10ms | Early exit, rule ordering |
| **Total end-to-end** | **< 200ms** | Pipeline optimization, caching |

### Throughput Optimization

**Single Node Capacity**:
```
Target: 10,000 requests/second (RPS)
Per-core capacity: ~1,250 RPS
Required cores: 8 (with headroom)
```

**Scaling Strategy**:
- **Horizontal**: Multiple WAF instances behind load balancer
- **Vertical**: Increase CPU/memory per instance
- **Caching**: Redis cache for repeated patterns
- **Batching**: Process requests in batches for ML inference

### Memory Management

```
Memory Usage Breakdown:
- CNN Model: ~800MB
- LSTM Model: ~600MB
- Isolation Forest: ~400MB
- Feature cache: ~2GB (LRU)
- Connection pool: ~200MB
- Request buffer: ~500MB
- Overhead: ~500MB
─────────────────────
Total: ~5.5GB per instance
```

**Optimization Techniques**:
- Model quantization (FP32 → INT8): 75% reduction
- Knowledge distillation: Smaller models, similar performance
- Pruning: Remove unused weights
- Compression: Archive models between deployments

### Database Performance

**Indexing Strategy**:
```sql
-- Primary indices
CREATE INDEX idx_request_timestamp ON requests(timestamp DESC);
CREATE INDEX idx_source_ip ON requests(source_ip);
CREATE INDEX idx_threat_level ON requests(threat_level);

-- Compound indices
CREATE INDEX idx_ip_timestamp ON requests(source_ip, timestamp DESC);
CREATE INDEX idx_uri_method ON requests(uri, method);

-- Partial indices
CREATE INDEX idx_blocked_requests ON requests(id) 
  WHERE threat_level IN ('critical', 'high');
```

**Query Optimization**:
- Connection pooling (min=10, max=50)
- Read replicas for analytics queries
- Caching query results (1 hour TTL)
- Partitioning by date for historical data

### Caching Strategy

**Cache Layers**:

1. **L1 - Application Memory Cache** (Redis)
   - TTL: 1 hour
   - Size: 2GB
   - Contents: Model predictions, threat patterns

2. **L2 - Feature Cache** (Redis)
   - TTL: 24 hours
   - Size: 5GB
   - Contents: Computed features for IPs, URIs

3. **L3 - Database Query Cache** (Redis)
   - TTL: 1 hour
   - Size: 1GB
   - Contents: Frequent queries

**Cache Invalidation**:
- Time-based: TTL expiration
- Event-based: When new threats detected
- Manual: Administrative purge

### Network Optimization

**Bandwidth Requirements**:
- Typical request size: 2-5KB
- At 10,000 RPS: ~20-50 Mbps
- With SSL/TLS overhead: ~30-75 Mbps
- Recommended: 1Gbps+ link for headroom

**Optimization**:
- HTTP/2 multiplexing
- Connection keep-alive
- Request compression
- Load balancing (round-robin preferred)

### Monitoring & Alerting

**Key Metrics to Monitor**:

```
Performance Metrics:
├── Response Time
│   ├── p50 latency
│   ├── p95 latency
│   └── p99 latency
├── Throughput
│   ├── Requests per second
│   ├── Errors per second
│   └── Queue depth
├── Resource Utilization
│   ├── CPU usage (%)
│   ├── Memory usage (%)
│   ├── Disk I/O (%)
│   └── Network I/O (%)
└── Model Performance
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── False positive rate
    └── Model latency

Security Metrics:
├── Blocked requests
├── Detected attack types
├── Blocked IPs
├── Attack trends
└── Signature hits
```

**Alert Thresholds**:
```yaml
Alerts:
  - name: "High Response Time"
    condition: "p99_latency > 500ms"
    severity: "warning"
  
  - name: "High CPU Usage"
    condition: "cpu_usage > 85%"
    severity: "critical"
  
  - name: "Model Inference Error"
    condition: "model_error_rate > 1%"
    severity: "critical"
  
  - name: "Cache Hit Ratio Low"
    condition: "cache_hit_ratio < 50%"
    severity: "warning"
  
  - name: "Database Connection Pool Exhausted"
    condition: "db_connections > 45"
    severity: "critical"
```

### Capacity Planning

**Growth Projections** (12-month forecast):

| Metric | Current | Month 6 | Month 12 |
|--------|---------|---------|----------|
| RPS | 5,000 | 7,500 | 10,000 |
| Requests/day | 432M | 648M | 864M |
| Storage (requests) | 50GB | 100GB | 200GB |
| CPU cores needed | 8 | 12 | 16 |
| Memory per instance | 32GB | 48GB | 64GB |
| Database size | 100GB | 200GB | 400GB |
| Redis cache | 5GB | 10GB | 15GB |

**Cost Optimization**:
- Use reserved instances for baseline load
- Auto-scaling for traffic spikes
- Spot instances for batch retraining
- Compression for long-term storage

---

## Troubleshooting

### Common Issues

#### Issue: High Model Inference Latency

**Symptoms**:
- API responses > 500ms
- p99 latency degradation
- Model timeout errors

**Diagnosis**:
```bash
# Check model status
curl http://localhost:8000/health/deep | jq '.models'

# Monitor system resources
top -b -n 1 | head -20

# Check model queue depth
redis-cli INFO stats
```

**Solutions**:
1. Enable GPU acceleration
2. Reduce batch size
3. Use quantized models
4. Increase number of worker processes
5. Scale horizontally

---

#### Issue: High False Positive Rate

**Symptoms**:
- Legitimate requests being blocked
- User complaints
- FPR > 3%

**Diagnosis**:
```bash
# Get FPR metrics
curl http://localhost:8000/metrics/models | jq '.models.*.false_positive_rate'

# Check recent blocked requests
SELECT * FROM blocked_requests 
WHERE timestamp > now() - interval '1 hour'
ORDER BY timestamp DESC;

# Analyze false positives
SELECT threat_type, COUNT(*) as count 
FROM blocked_requests 
WHERE actual_threat = false 
GROUP BY threat_type;
```

**Solutions**:
1. Adjust confidence thresholds
2. Retrain models with additional benign samples
3. Review and update signature rules
4. Implement per-application whitelisting

---

#### Issue: Database Connection Pool Exhaustion

**Symptoms**:
- Connection refused errors
- "Too many connections" errors
- Increased latency

**Diagnosis**:
```bash
# Check active connections
SELECT count(*) FROM pg_stat_activity;

# Check connection pool status
curl http://localhost:8000/metrics/database

# Check for long-running queries
SELECT pid, query, state_change 
FROM pg_stat_activity 
WHERE state != 'idle';
```

**Solutions**:
1. Increase pool size (carefully)
2. Kill long-running queries
3. Optimize slow queries
4. Add read replicas
5. Implement query timeout

---

#### Issue: Memory Leak

**Symptoms**:
- Increasing memory usage over time
- Eventual out-of-memory crash
- Slow performance increase

**Diagnosis**:
```bash
# Monitor memory usage
watch -n 5 'ps aux | grep waf-ml'

# Use memory profiler
python -m memory_profiler app/main.py

# Check for unreleased resources
lsof -p <PID> | wc -l
```

**Solutions**:
1. Profile code with memory profiler
2. Check for circular references
3. Ensure proper cleanup in exception handlers
4. Review model loading/unloading
5. Implement garbage collection tuning

---

### Debug Logging

**Enable Debug Mode**:
```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
```

**View Detailed Logs**:
```bash
# Real-time logs
tail -f logs/application.log

# Search for errors
grep "ERROR" logs/application.log

# Filter by component
grep "cnn_model" logs/application.log

# Time range filtering
awk '$2 >= "10:00:00" && $2 <= "10:05:00"' logs/application.log
```

---

## Contributing Guidelines

### Code Quality Standards

- **Language**: Python 3.9+
- **Linting**: Black, Flake8
- **Type Hints**: Mypy
- **Testing**: Pytest with >80% coverage
- **Documentation**: Sphinx + Markdown

### Contribution Process

1. Create feature branch from `develop`
2. Make changes with descriptive commits
3. Write/update tests
4. Run linting and type checking
5. Submit pull request with detailed description
6. Pass code review and CI/CD
7. Merge to develop, then main on release

### Security Considerations

- No hardcoded credentials
- Use environment variables
- Validate all inputs
- Sanitize logging output
- Regular dependency updates
- Security scanning in CI/CD

---

## Support & Contact

- **Documentation**: https://docs.swavlamban.io/waf-ml
- **Issue Tracker**: https://github.com/dev2089/swavlamban-waf-ml/issues
- **Email**: support@swavlamban.io
- **Slack**: #waf-ml-support

---

**Document Version**: 1.0  
**Last Updated**: December 25, 2025  
**Maintained By**: dev2089 Team
