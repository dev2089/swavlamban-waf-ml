-- Portable project-execution ledger schema.
-- This is evidence/state, not the runtime WAF traffic database.
CREATE TABLE IF NOT EXISTS project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS phases (phase INTEGER PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, score REAL, cutoff REAL, notes TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, task TEXT NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS test_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, score REAL, details TEXT NOT NULL, run_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS change_log (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, path TEXT NOT NULL, version TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, metadata TEXT NOT NULL, recorded_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS environment_snapshot (key TEXT PRIMARY KEY, value TEXT NOT NULL, captured_at TEXT NOT NULL);

-- Phase 7 learning-control evidence tables.
CREATE TABLE IF NOT EXISTS phase7_baselines (baseline_version TEXT PRIMARY KEY, schema_version TEXT NOT NULL, feature_schema TEXT NOT NULL, sample_count INTEGER NOT NULL, source TEXT NOT NULL, row_sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS phase7_feedback (record_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, schema_version TEXT NOT NULL, feature_schema TEXT NOT NULL, feature_snapshot_json TEXT NOT NULL, observed_decision TEXT NOT NULL, reviewed_label INTEGER, review_state TEXT NOT NULL, reviewer TEXT, review_note TEXT NOT NULL, evidence_rule_ids_json TEXT NOT NULL, model_version TEXT NOT NULL, baseline_version TEXT NOT NULL, privacy_json TEXT NOT NULL, created_at TEXT NOT NULL, reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS phase7_drift_reports (report_id TEXT PRIMARY KEY, baseline_version TEXT NOT NULL, feature_schema TEXT NOT NULL, sample_count INTEGER NOT NULL, mean_psi REAL NOT NULL, max_psi REAL NOT NULL, material_drift INTEGER NOT NULL, alert_level TEXT NOT NULL, top_features_json TEXT NOT NULL, thresholds_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS phase7_model_runs (run_id TEXT PRIMARY KEY, schema_version TEXT NOT NULL, role TEXT NOT NULL, model_version TEXT NOT NULL, dataset_version TEXT NOT NULL, baseline_version TEXT NOT NULL, artifact_path TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, artifact_bytes INTEGER NOT NULL, evaluation_json TEXT NOT NULL, source_feedback_ids_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS phase7_model_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, run_id TEXT, from_run_id TEXT, to_run_id TEXT, actor TEXT, status TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL);
