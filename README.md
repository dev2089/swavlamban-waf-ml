# Swavlamban WAF ML

ML-augmented Web Application Firewall prototype for **Challenge 3**.

## Current verified state

- Phase 1: PASS, 10.0/10.0
- Phase 2: PASS, 10.0/10.0
- Phase 3: PASS, 10.0/10.0
- Phase 4: PASS, 10.0/10.0, critical defects 0
- Authoritative development branch: `phase4-final`
- `main` remains untouched by milestone work

## Phase 4 ML

The live edge uses the canonical `RequestEnvelope` + `http-v2` feature seam and consumes:

1. supervised `HistGradientBoostingClassifier`;
2. benign-only unsupervised `OneClassSVM`;
3. learned stateful behavioural `LogisticRegression`;
4. deterministic signature rules for known attacks.

Model artifact: `models/phase4_models.joblib`.

## Reproducibility

Install Phase 4 dependencies:

```bash
python -m pip install -r requirements-phase4.txt
```

Train/refresh the model artifact:

```bash
python scripts/train_phase4_models.py
```

Run the full regression:

```bash
python -m pytest -q
```

Run the milestone self-test:

```bash
python phase4_self_test.py
```

Run the master gate:

```bash
python scripts/phase4_master_exam.py
```

Run the extended local E2E benchmark:

```bash
python phase4_e2e_benchmark.py
```

## Evidence

Read `handoff/START_HERE.md` first for continuation state and exact evidence locations. The project ledger in `state/project_ledger.db` is the portable execution/state database; it is separate from the future runtime WAF traffic store.

## Important honesty boundary

Phase 4 ML evaluation uses deterministic synthetic HTTP data and deterministic behavioural workloads. Those metrics are reproducibility evidence only and are not claims of real-world Internet WAF accuracy or production capacity.

The complete Challenge 3 submission is **not finished yet**. Remaining milestones include TLS/HTTPS handling, expanded explainability, ML-generated rule lifecycle, production baseline/feedback/drift/retraining, secure production storage/authentication, full scenario evidence, dashboard migration, the five-minute demo, submission documentation/slides and the final release gate.
