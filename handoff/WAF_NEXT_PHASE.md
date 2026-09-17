# Next Execution Step: Phase 4

Attach supervised, unsupervised and behavioural ML to the existing canonical `RequestEnvelope -> http-v2 FeatureVector -> DetectionSignal -> DecisionResult` seam.

## Phase 4 goal
Build real ML detectors using reproducible datasets and actual HTTP-derived features. Keep the security fast path bounded and deterministic. Do not claim accuracy, precision, recall or latency numbers without an executable evaluation run.

## Phase 4 acceptance
- supervised detector with train/validation/test separation;
- unsupervised detector for anomaly detection;
- behavioural detector over request windows;
- calibrated/explicit risk semantics rather than fake probabilities;
- model/feature/dataset versioning;
- explainable detector signals with reasons/features;
- regression against known SQLi/XSS/traversal/command signatures;
- reproducible evaluation report with confusion matrix and latency;
- fuzz/failure tests for model absence, malformed inputs and extreme feature values;
- 9.9+ self-exam with critical-defect fail rule.
