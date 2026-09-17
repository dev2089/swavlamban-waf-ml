-- Phase 10 performance advisor remediation.
create index if not exists idx_alerts_threat_id on public.alerts(threat_id);
create index if not exists idx_request_logs_threat_id on public.request_logs(threat_id);

drop policy if exists waf_user_roles_self_select on public.waf_user_roles;
create policy waf_user_roles_self_select
  on public.waf_user_roles for select to authenticated
  using (user_id = (select auth.uid()));
