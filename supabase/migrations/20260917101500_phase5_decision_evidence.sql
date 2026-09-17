/* Phase 5 decision evidence storage.
   This table intentionally stores structured numeric/metadata evidence only.
   It does NOT store request payloads, raw query strings, raw headers, or source IPs.
*/

CREATE TABLE IF NOT EXISTS decision_evidence (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('allow', 'alert', 'block')),
  risk_score numeric NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
  evidence_schema text NOT NULL DEFAULT 'evidence-v1',
  detector_contributions jsonb NOT NULL DEFAULT '[]'::jsonb,
  feature_groups jsonb NOT NULL DEFAULT '{}'::jsonb,
  feature_attribution jsonb NOT NULL DEFAULT '{}'::jsonb,
  reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
  rule_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
  versions jsonb NOT NULL DEFAULT '{}'::jsonb,
  explanation text NOT NULL DEFAULT '',
  privacy jsonb NOT NULL DEFAULT '{"raw_payload_retained":false,"raw_headers_retained":false,"raw_query_retained":false}'::jsonb,
  feature_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_decision_evidence_request_id ON decision_evidence(request_id);
CREATE INDEX IF NOT EXISTS idx_decision_evidence_decision ON decision_evidence(decision);
CREATE INDEX IF NOT EXISTS idx_decision_evidence_created_at ON decision_evidence(created_at DESC);

ALTER TABLE decision_evidence ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow authenticated read access to decision evidence"
  ON decision_evidence FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated insert of decision evidence"
  ON decision_evidence FOR INSERT
  TO authenticated
  WITH CHECK (
    (privacy->>'raw_payload_retained')::boolean = false
    AND (privacy->>'raw_headers_retained')::boolean = false
    AND (privacy->>'raw_query_retained')::boolean = false
  );
