-- Phase 9 live-DB privilege remediation.
-- Keep public/anonymous access closed and make runtime persistence server-only.
REVOKE ALL ON public.decision_evidence FROM anon, authenticated;
GRANT SELECT ON public.decision_evidence TO authenticated;
GRANT ALL ON public.decision_evidence TO service_role;
REVOKE ALL ON public.waf_runtime_events FROM anon;
REVOKE ALL ON public.waf_security_audit FROM anon;
REVOKE ALL ON public.waf_user_roles FROM anon;
REVOKE ALL ON public.phase7_baselines, public.phase7_feedback, public.phase7_drift_reports, public.phase7_model_runs, public.phase7_model_events FROM anon;
REVOKE INSERT, UPDATE, DELETE ON public.waf_security_audit FROM authenticated;
REVOKE INSERT, UPDATE, DELETE ON public.waf_user_roles FROM authenticated;
REVOKE INSERT, UPDATE, DELETE ON public.waf_runtime_events FROM authenticated;
GRANT SELECT ON public.waf_runtime_events TO authenticated;
GRANT SELECT ON public.waf_security_audit TO authenticated;
GRANT SELECT ON public.waf_user_roles TO authenticated;
GRANT ALL ON public.waf_runtime_events, public.waf_security_audit, public.waf_user_roles TO service_role;
