-- swavlamban-waf Phase 10 sanitized execution log.
create table if not exists public.swavlamban_waf_phase_execution_log (
  event_id bigint generated always as identity primary key,
  phase integer not null check (phase >= 1),
  event_type text not null,
  status text not null,
  details jsonb not null default '{}'::jsonb,
  occurred_at timestamptz not null default now(),
  expires_at timestamptz not null
);

alter table public.swavlamban_waf_phase_execution_log enable row level security;
drop policy if exists swavlamban_waf_phase_execution_log_read on public.swavlamban_waf_phase_execution_log;
create policy swavlamban_waf_phase_execution_log_read on public.swavlamban_waf_phase_execution_log
  for select to authenticated using (public.waf_has_role('reviewer'));

revoke all on public.swavlamban_waf_phase_execution_log from anon, authenticated;
grant select on public.swavlamban_waf_phase_execution_log to authenticated;
grant all on public.swavlamban_waf_phase_execution_log to service_role;

create index if not exists idx_swavlamban_waf_phase_execution_log_phase_time
  on public.swavlamban_waf_phase_execution_log(phase, occurred_at desc);

comment on table public.swavlamban_waf_phase_execution_log is
  'Sanitized WAF phase execution history. Never store raw request payloads, secrets or credentials.';
