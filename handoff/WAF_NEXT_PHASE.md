# Next Execution Step: Phase 2

Attach the new `WAFPipeline` to an actual HTTP interception path using an open-source WAF/reverse-proxy adapter. The acceptance test must prove externally that malicious traffic is prevented from reaching the protected upstream while benign traffic passes.

Do not add dashboard polish before the real enforcement path works.
