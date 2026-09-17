-- Phase 8: production security control-plane schema.
-- This migration is intentionally additive and is not proof of live deployment.
-- Service-role credentials must stay server-side and are never stored here.

create table if not exists public.waf_user_roles (
    user_id uuid primary key references auth.users(id) on delete cascade,
    role text not null check (role in ('viewer','operator','reviewer','admin')),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.waf_security_audit (
    event_id uuid primary key default gen_random_uuid(),
    actor_id uuid references auth.users(id) on delete set null,
    role text not null check (role in ('viewer','operator','reviewer','admin','system')),
    action text not null,
    target text not null,
    outcome text not null check (outcome in ('allow','deny','success','failure')),
    request_id text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_waf_security_audit_actor_created
    on public.waf_security_audit(actor_id, created_at desc);
create index if not exists idx_waf_security_audit_target_created
    on public.waf_security_audit(target, created_at desc);

alter table public.waf_user_roles enable row level security;
alter table public.waf_security_audit enable row level security;

create or replace function public.waf_role_rank(role_name text)
returns integer
language sql
immutable
as $$
    select case role_name
        when 'viewer' then 10
        when 'operator' then 20
        when 'reviewer' then 30
        when 'admin' then 40
        when 'system' then 50
        else 0
    end;
$$;

create or replace function public.waf_current_role()
returns text
language sql
stable
as $$
    select coalesce(auth.jwt() -> 'app_metadata' ->> 'waf_role', 'viewer');
$$;

create or replace function public.waf_has_role(required_role text)
returns boolean
language sql
stable
as $$
    select public.waf_role_rank(public.waf_current_role()) >= public.waf_role_rank(required_role);
$$;

-- Users may read only their own role mapping. Server-side service-role access is not
-- restricted by RLS and is the path for administrative role changes.
drop policy if exists waf_user_roles_self_select on public.waf_user_roles;
create policy waf_user_roles_self_select
    on public.waf_user_roles for select to authenticated
    using (user_id = auth.uid());

-- Audit history is immutable to end users: authenticated callers may read only when
-- their role is reviewer/admin; writes are intentionally service-role-only.
drop policy if exists waf_security_audit_reviewer_select on public.waf_security_audit;
create policy waf_security_audit_reviewer_select
    on public.waf_security_audit for select to authenticated
    using (public.waf_has_role('reviewer'));

grant usage on schema public to authenticated;
grant select on public.waf_user_roles to authenticated;
grant select on public.waf_security_audit to authenticated;
grant all on public.waf_user_roles to service_role;
grant all on public.waf_security_audit to service_role;
revoke all on public.waf_user_roles from anon;
revoke all on public.waf_security_audit from anon;

comment on table public.waf_user_roles is
    'Phase 8 RBAC mapping. Keep authoritative role decisions server-side and in JWT app_metadata.';
comment on table public.waf_security_audit is
    'Phase 8 privacy-safe control-plane audit. Never store request bodies, raw headers, or secrets.';
