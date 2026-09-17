-- Phase 10 release audit checkpoint. No raw request material is stored.
create table if not exists public.waf_phase_release_audit (
  release_id uuid primary key default gen_random_uuid(),
  phase integer not null check (phase >= 1),
  branch text not null,
  commit_sha text not null,
  gate_score numeric not null check (gate_score between 0 and 10),
  gate_cutoff numeric not null check (gate_cutoff between 0 and 10),
  result text not null check (result in ('PASS','FAIL')),
  regression_summary text not null,
  live_db_verified boolean not null,
  evidence jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);

alter table public.waf_phase_release_audit enable row level security;
drop policy if exists waf_phase_release_audit_read on public.waf_phase_release_audit;
create policy waf_phase_release_audit_read on public.waf_phase_release_audit
  for select to authenticated using (public.waf_has_role('reviewer'));
revoke all on public.waf_phase_release_audit from anon, authenticated;
grant select on public.waf_phase_release_audit to authenticated;
grant all on public.waf_phase_release_audit to service_role;
create index if not exists idx_waf_phase_release_audit_created on public.waf_phase_release_audit(created_at desc);
