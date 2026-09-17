-- Phase 7 learning-control persistence contract.
-- Not live-applied in this milestone. Phase 8 production storage/auth/RBAC is still open.
create table if not exists phase7_baselines (
  baseline_version text primary key,
  schema_version text not null,
  feature_schema text not null,
  sample_count integer not null check (sample_count >= 32),
  source text not null,
  row_sha256 text not null,
  metadata jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists phase7_feedback (
  record_id text primary key,
  request_id text not null,
  schema_version text not null,
  feature_schema text not null,
  feature_snapshot jsonb not null,
  observed_decision text not null,
  reviewed_label smallint check (reviewed_label in (0,1)),
  review_state text not null check (review_state in ('pending','reviewed','rejected')),
  reviewer text,
  review_note text not null default '',
  evidence_rule_ids jsonb not null default '[]'::jsonb,
  model_version text not null,
  baseline_version text not null,
  privacy jsonb not null,
  created_at timestamptz not null default now(),
  reviewed_at timestamptz
);

create table if not exists phase7_drift_reports (
  report_id text primary key,
  baseline_version text not null,
  feature_schema text not null,
  sample_count integer not null check (sample_count >= 32),
  mean_psi numeric not null,
  max_psi numeric not null,
  material_drift boolean not null,
  alert_level text not null,
  top_features jsonb not null,
  thresholds jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists phase7_model_runs (
  run_id text primary key,
  schema_version text not null,
  role text not null,
  model_version text not null,
  dataset_version text not null,
  baseline_version text not null,
  artifact_path text not null,
  artifact_sha256 text not null,
  artifact_bytes bigint not null,
  evaluation jsonb not null,
  source_feedback_ids jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists phase7_model_events (
  event_id bigint generated always as identity primary key,
  event_type text not null,
  run_id text,
  from_run_id text,
  to_run_id text,
  actor text,
  status text not null,
  details jsonb not null,
  created_at timestamptz not null default now()
);
