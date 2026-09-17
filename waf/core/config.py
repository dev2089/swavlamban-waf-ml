from __future__ import annotations
import os
from dataclasses import dataclass
@dataclass(frozen=True,slots=True)
class WAFConfig:
    """Runtime policy. Secrets are loaded separately by waf.security.secrets."""
    pipeline_version:str='phase3'; block_threshold:float=.80; alert_threshold:float=.50; max_body_bytes:int=1048576; feature_schema_version:str='http-v2'; upstream_url:str='http://127.0.0.1:9000'; listen_host:str='127.0.0.1'; listen_port:int=8080; request_timeout_seconds:float=10.0; max_response_bytes:int=10485760; environment:str='development'; storage_backend:str='memory'; storage_url:str='state/runtime_events.db'; auth_required:bool=False; retention_days:int=30
    @classmethod
    def from_env(cls)->'WAFConfig':
        def bounded_float(name,default):
            raw=os.getenv(name); value=default if raw is None else float(raw)
            if not 0<=value<=1: raise ValueError(f'{name} must be in [0, 1]')
            return value
        max_body=int(os.getenv('WAF_MAX_BODY_BYTES','1048576'))
        if max_body<=0: raise ValueError('WAF_MAX_BODY_BYTES must be positive')
        max_response=int(os.getenv('WAF_MAX_RESPONSE_BYTES','10485760'))
        if max_response<=0: raise ValueError('WAF_MAX_RESPONSE_BYTES must be positive')
        listen_port=int(os.getenv('WAF_LISTEN_PORT','8080'))
        if not 1<=listen_port<=65535: raise ValueError('WAF_LISTEN_PORT must be in [1, 65535]')
        timeout=float(os.getenv('WAF_REQUEST_TIMEOUT_SECONDS','10.0'))
        if timeout<=0: raise ValueError('WAF_REQUEST_TIMEOUT_SECONDS must be positive')
        environment=os.getenv('WAF_ENV','development').strip().lower()
        if environment not in {'development','test','staging','production'}: raise ValueError('WAF_ENV must be development, test, staging or production')
        storage_backend=os.getenv('WAF_STORAGE_BACKEND','memory').strip().lower()
        if storage_backend not in {'memory','sqlite','supabase'}: raise ValueError('WAF_STORAGE_BACKEND must be memory, sqlite or supabase')
        auth_default=environment=='production'; auth_raw=os.getenv('WAF_AUTH_REQUIRED'); auth_required=auth_default if auth_raw is None else auth_raw.strip().lower() in {'1','true','yes','on'}
        retention_days=int(os.getenv('WAF_RETENTION_DAYS','30'))
        if not 1<=retention_days<=3650: raise ValueError('WAF_RETENTION_DAYS must be between 1 and 3650')
        if environment=='production' and storage_backend=='memory': raise ValueError('production requires persistent WAF_STORAGE_BACKEND')
        if environment=='production' and not auth_required: raise ValueError('production requires WAF_AUTH_REQUIRED')
        return cls(os.getenv('WAF_PIPELINE_VERSION','phase3'),bounded_float('WAF_BLOCK_THRESHOLD',.80),bounded_float('WAF_ALERT_THRESHOLD',.50),max_body,os.getenv('WAF_FEATURE_SCHEMA','http-v2'),os.getenv('WAF_UPSTREAM_URL','http://127.0.0.1:9000'),os.getenv('WAF_LISTEN_HOST','127.0.0.1'),listen_port,timeout,max_response,environment,storage_backend,os.getenv('WAF_STORAGE_URL','state/runtime_events.db'),auth_required,retention_days)
