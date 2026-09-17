from __future__ import annotations
from dataclasses import dataclass
import os,re
from typing import Any,Mapping
_SECRET_NAME_RE=re.compile(r'(secret|token|password|private|service[_-]?role|api[_-]?key)',re.I)
_PLACEHOLDER_RE=re.compile(r'^(?:change[-_]?me|replace[-_]?me|your[-_].*|example|test|dummy|null|none)$',re.I)
class SecretError(ValueError): pass
@dataclass(frozen=True,slots=True)
class SecretProvider:
    environ: Mapping[str,str]|None=None
    def _env(self): return os.environ if self.environ is None else self.environ
    def get(self,name:str,*,required:bool=False,min_bytes:int=32):
        value=self._env().get(name)
        if value is None or not value.strip():
            if required: raise SecretError(f'required secret {name} is missing')
            return None
        value=value.strip()
        if min_bytes and len(value.encode())<min_bytes: raise SecretError(f'secret {name} must be at least {min_bytes} bytes')
        if _PLACEHOLDER_RE.match(value): raise SecretError(f'secret {name} uses a placeholder value')
        return value
    def production(self):
        names=('WAF_AUTH_SECRET','SUPABASE_URL','SUPABASE_SERVICE_ROLE_KEY'); result={}
        for name in names:
            minimum=32 if 'SECRET' in name or 'KEY' in name else 1
            value=self.get(name,required=True,min_bytes=minimum); assert value is not None; result[name]=value
        return result
def redact_mapping(value:Mapping[str,Any])->dict[str,Any]:
    output={}
    for key,item in value.items():
        if _SECRET_NAME_RE.search(str(key)): output[str(key)]='[REDACTED]'
        elif isinstance(item,Mapping): output[str(key)]=redact_mapping(item)
        elif isinstance(item,(list,tuple)): output[str(key)]=[redact_mapping(x) if isinstance(x,Mapping) else x for x in item]
        else: output[str(key)]=item
    return output
