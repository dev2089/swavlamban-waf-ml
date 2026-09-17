-- Phase 10 Supabase advisor remediation: pin function search_path.
create or replace function public.waf_role_rank(role_name text)
returns integer language sql immutable
set search_path = public, pg_temp
as $$ select case role_name when 'viewer' then 10 when 'operator' then 20 when 'reviewer' then 30 when 'admin' then 40 when 'system' then 50 else 0 end $$;

create or replace function public.waf_current_role()
returns text language sql stable
set search_path = public, pg_temp
as $$ select coalesce(auth.jwt() -> 'app_metadata' ->> 'waf_role', 'viewer') $$;

create or replace function public.waf_has_role(required_role text)
returns boolean language sql stable
set search_path = public, pg_temp
as $$ select public.waf_role_rank(public.waf_current_role()) >= public.waf_role_rank(required_role) $$;

create or replace function public.waf_actor_role()
returns text language sql stable
set search_path = public, pg_temp
as $$ select coalesce(auth.jwt() -> 'app_metadata' ->> 'role', '') $$;

create or replace function public.waf_is_operator()
returns boolean language sql stable
set search_path = public, pg_temp
as $$ select auth.role() = 'service_role' or public.waf_actor_role() in ('analyst','rule_approver','model_approver','admin') $$;

create or replace function public.waf_purge_expired_runtime_events(p_before timestamptz)
returns integer language plpgsql security invoker
set search_path = public, pg_temp
as $$
declare deleted_count integer;
begin
  if auth.role() <> 'service_role' and public.waf_actor_role() <> 'admin' then raise exception 'admin authorization required'; end if;
  delete from public.waf_runtime_events where expires_at <= p_before;
  get diagnostics deleted_count = row_count;
  return deleted_count;
end;
$$;
