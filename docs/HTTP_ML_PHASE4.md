# Phase 4 ML Architecture

## Scope
Phase 4 adds three independent security signals behind the canonical `RequestEnvelope` and `http-v2` feature contracts:

1. **Supervised classifier**: `HistGradientBoostingClassifier` trained on a deterministic labelled HTTP benchmark dataset.
2. **Unsupervised anomaly detector**: `OneClassSVM` fitted only on a varied benign HTTP baseline; its anomaly threshold is learned from that benign baseline.
3. **Learned behavioural detector**: stateful per-source sliding-window features scored by a trained `LogisticRegression` model for burst/churn behaviour.

The deterministic WAF signature layer remains authoritative for known attack signatures.

## Fast path

`RequestEnvelope -> http-v2 -> signature + supervised + unsupervised + learned behaviour -> EdgeDecisionPolicy`

The ML detectors receive numeric feature vectors. Raw payloads are not placed into feature vectors.

## Risk semantics

Scores are **risk scores**, not calibrated probabilities.

- Known signature hit: risk is forced to `1.0` and blocks under the default threshold.
- Otherwise the policy combines the strongest signal with weighted detector scores.
- Weights: supervised `0.55`, unsupervised `0.30`, behaviour `0.15`.
- Behavioural state is intentionally per-process in this phase and is reset when a new runtime is created.

## Supervised evaluation

The included labelled benchmark is synthetic and deterministic. A stratified 25% holdout is used for evaluation.

Measured result: accuracy `1.0`, precision `1.0`, recall `1.0`, F1 `1.0`, FPR `0.0`, test samples `1500`.

These metrics demonstrate reproducibility on this synthetic benchmark only. They are not real-world WAF accuracy claims.

## Unsupervised evaluation

The anomaly model is fitted only on benign examples. A held-out portion of that benign baseline is used to measure false positives, while attack examples are used only as an external anomaly challenge set.

Measured result: benign evaluation `750`, attack evaluation `3000`, FPR `0.0227`, attack detection rate `0.8980`, mean benign score `0.2531`, mean attack score `0.8980`.

The anomaly score is derived from the one-class decision function relative to the learned benign threshold. It is not a calibrated probability.

## Learned behavioural evaluation

The detector maintains a per-source time window and derives request count, rate, path churn and inter-arrival statistics. A trained logistic-regression classifier maps these window features to a behavioural risk score.

Measured result: normal max score `0.406313`, burst final score `0.980486`, escalation `true`, normal requests `10`, burst requests `40`.

This is a deterministic synthetic behavioural workload, not a claim about real-world bot detection accuracy.

## Model artifact

`models/phase4_models.joblib` stores the fitted supervised classifier, benign-only anomaly detector and learned behavioural classifier plus schema/version metadata. The per-source behavioural event window is created fresh for each runtime, so live state is not serialized.

Artifact: 199123 bytes, SHA-256 `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Reproducibility

```bash
python scripts/train_phase4_models.py --output models/phase4_models.joblib
python phase4_self_test.py
python scripts/phase4_master_exam.py
python phase4_e2e_benchmark.py
```

## Final runtime evidence

- Direct ML: `2000 requests, 531.1 req/s, 1900 allow, 100 block, 0 alert`.
- Mandatory bounded E2E: `1000 requests, 428.6 req/s, 850 HTTP 200, 150 HTTP 403, 0 errors, p50 59.639 ms, p95 77.706 ms`.
- Extended E2E: `5000 requests, 410.4 req/s, 4250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms, 4250 upstream hits, 1000 events`.

## Known limitations carried forward
- Training/evaluation data are deterministic synthetic HTTP data and must not be presented as Internet-scale accuracy evidence.
- No semi-supervised learner is introduced in this milestone.
- Behavioural state is process-local; distributed/shared state belongs to later telemetry/storage work.
- Calibration, drift detection and controlled retraining belong to later phases.
- TLS termination and separately installed ModSecurity/Coraza verification remain later work.
