# Phase 4 ML Architecture

## Fast path

RequestEnvelope -> http-v2 -> deterministic signatures + supervised + unsupervised + learned behavioural ML -> deterministic edge policy.

The ML path performs inference only. No database writes occur during request-time ML inference. Raw request payloads are not stored in the FeatureVector.

## Models

Supervised: HistGradientBoostingClassifier trained on a deterministic labelled HTTP dataset.

Unsupervised: OneClassSVM trained only on benign baseline rows. The anomaly threshold is learned from the benign fit.

Behavioural: LogisticRegression over bounded per-source sliding-window features for request count/rate, unique path churn and inter-arrival behavior. The window is process-local and capped at 128 events per source.

Known signature matches remain authoritative hard blocks. The learned policy weights supervised/unsupervised/behavioural signals 0.55/0.30/0.15, with an additional 0.35 strongest-signal component and 0.65 weighted component. ML inference failure becomes a risk=1.0 fail-closed signal.

Scores are risk scores, not calibrated probabilities.

## Evaluation

Supervised held-out result: accuracy=1.0, precision=1.0, recall=1.0, F1=1.0, FPR=0.0, n=1500.

Unsupervised result: benign FPR=0.017333, attack detection=0.888, benign evaluation n=750, attack evaluation n=3000.

Behavioural workload: normal max=0.406313, burst final=0.980486, escalation=true, with 10 normal and 40 burst requests.

All quality measurements are deterministic synthetic evidence only.

## Artifact

Artifact version: phase4-model-v1.
Model version: phase4-ml-v1.
Feature schema: http-v2.
Feature count: 40.
Generate with: python scripts/train_phase4_models.py --output models/phase4_models.joblib

Two local deterministic generations matched at 183235 bytes with SHA-256 8fe56e23ea19a6dc2e82563b0ae4f1d0754a3df2ee5be1e4042a51910edabf50. The binary is intentionally generated, not required to be committed.

## Verification

53/53 tests passed and compileall passed. Latest local direct benchmark was 2000 requests at 323.94 req/s. Latest local E2E benchmark was 1000 requests at concurrency 50, 161.89 req/s, with 850 HTTP 200, 150 HTTP 403, 0 HTTP 500 and 850 upstream hits. Phase 3 Nginx integration remains valid because the Nginx configuration was not changed by Phase 4.

## Carry-forward

Semi-supervised learning, distributed behavioural state, feedback/drift/retraining, TLS, separate ModSecurity/Coraza verification, production persistence/auth/RBAC, large-scale and failure testing, dashboard migration, final scenarios/demo and submission package remain later work.

Phase 4 is a completed milestone, not overall Challenge 3 completion.
