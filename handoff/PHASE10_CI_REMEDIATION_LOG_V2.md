# Phase 10 CI Remediation Log V2

This file preserves the second CI failure discovered after the first Phase 10 remediation.

## Run 59 failure
- workflow: `phase10-release-candidate`
- run: `35259785621`
- job: `105332320512`
- commit tested: `2b68b1b17cf955ca9adf7ffda97aeaa60c071487`
- failing step: isolated gateway startup smoke

## Exact failure
`ValueError: could not convert string to float: "<member 'request_timeout_seconds' of 'GatewayConfig' objects>"`

## Root cause
`GatewayConfig` is declared as a slotted dataclass. Inside `from_env()`, fallback expressions such as `cls.request_timeout_seconds` resolve to dataclass member descriptors, not default values. The failure therefore appears as soon as an environment variable is absent.

## Fix
Replace all `cls.<field>` environment fallbacks with explicit default literals matching the declared dataclass defaults. Add a regression suite covering default construction and environment overrides.

## Connector decision
This is a WAF code defect, not a Vercel or Supabase integration issue. No new connector is required.

## Completion gate
A fresh green Phase 10 release-gate run is required before Phase 10 can be closed.
