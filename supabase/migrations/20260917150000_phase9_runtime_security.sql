-- Phase 9 runtime-security migration.
-- Goal: move the legacy WAF tables behind the Phase 8 role gate and
-- prevent future persistence of raw request material in runtime telemetry.

REVOKE ALL ON public.threats FROM anon;
REVOKE ALL ON public.security_rules FROM anon;
REVOKE ALL ON public.request_logs FROM anon;
REVOKE ALL ON public.analytics FROM anon;
REVOKE ALL ON public.alerts FROM anon;
REVOKE ALL ON public.ml_models FROM anon;

DO $$
DECLARE
  r record;
BEGIN
  FOR r IN
    SELECT schemaname, tablename, policyname
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename IN ('threats','security_rules','request_logs','analytics','alerts','ml_models')
      AND policyname IN (
        'Allow public read access to threats', 'Allow public insert to threats',
        'Allow public read access to security_rules', 'Allow public insert to security_rules',
        'Allow public read access to request_logs', 'Allow public insert to request_logs',
        'Allow public read access to analytics', 'Allow public insert to analytics',
        'Allow public read access to alerts', 'Allow public insert to alerts',
        'Allow public read access to ml_models', 'Allow public insert to ml_models'
      )
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I.%I', r.policyname, r.schemaname, r.tablename);
  END LOOP;
END $$;

ALTER TABLE public.threats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.security_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.request_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ml_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "phase9_view_threats" ON public.threats
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));
CREATE POLICY "phase9_view_rules" ON public.security_rules
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));
CREATE POLICY "phase9_view_logs" ON public.request_logs
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));
CREATE POLICY "phase9_view_analytics" ON public.analytics
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));
CREATE POLICY "phase9_view_alerts" ON public.alerts
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));
CREATE POLICY "phase9_view_models" ON public.ml_models
  FOR SELECT TO authenticated USING (public.waf_has_role('viewer'));

REVOKE ALL ON public.threats, public.security_rules, public.request_logs,
  public.analytics, public.alerts, public.ml_models FROM authenticated;
GRANT SELECT ON public.threats, public.security_rules, public.request_logs,
  public.analytics, public.alerts, public.ml_models TO authenticated;
GRANT ALL ON public.threats, public.security_rules, public.request_logs,
  public.analytics, public.alerts, public.ml_models TO service_role;

-- Privacy remediation for pre-Phase-9 rows. Keep the record, remove raw material.
UPDATE public.threats SET payload = NULL WHERE payload IS NOT NULL;
UPDATE public.request_logs
SET body = NULL,
    headers = '{}'::jsonb,
    query_params = '{}'::jsonb
WHERE body IS NOT NULL
   OR headers <> '{}'::jsonb
   OR query_params <> '{}'::jsonb;

ALTER TABLE public.threats
  ADD CONSTRAINT phase9_threat_payload_null CHECK (payload IS NULL) NOT VALID;
ALTER TABLE public.request_logs
  ADD CONSTRAINT phase9_request_body_null CHECK (
    body IS NULL AND headers = '{}'::jsonb AND query_params = '{}'::jsonb
  ) NOT VALID;

ALTER TABLE public.threats
  ADD CONSTRAINT phase9_threat_source_ip_hash CHECK (source_ip ~ '^[0-9a-f]{64}$' OR source_ip = 'unknown') NOT VALID;
ALTER TABLE public.request_logs
  ADD CONSTRAINT phase9_request_source_ip_hash CHECK (source_ip ~ '^[0-9a-f]{64}$' OR source_ip = 'unknown') NOT VALID;

COMMENT ON CONSTRAINT phase9_threat_payload_null ON public.threats IS
  'Phase 9 privacy boundary: runtime WAF telemetry never persists raw payloads.';
COMMENT ON CONSTRAINT phase9_request_body_null ON public.request_logs IS
  'Phase 9 privacy boundary: runtime WAF telemetry never persists raw body, headers, or query parameters.';
