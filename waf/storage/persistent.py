from __future__ import annotations
from datetime import datetime,timedelta,timezone
import json,sqlite3
from pathlib import Path
from typing import Any,Mapping
FORBIDDEN_FIELDS=frozenset({'body','payload','query','raw_query','headers','raw_headers','source_ip','client_ip','uri'})
RUNTIME_SCHEMA_VERSION='storage-v1'
class StoragePrivacyError(ValueError): pass
def _contains_forbidden(value:Any)->str|None:
    if isinstance(value,Mapping):
        for key,item in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:return str(key)
            nested=_contains_forbidden(item)
            if nested:return nested
    elif isinstance(value,(list,tuple)):
        for item in value:
            nested=_contains_forbidden(item)
            if nested:return nested
    return None
def _json(value:Any)->str:return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)
class SQLiteSecurityStore:
    def __init__(self,path:str|Path)->None:
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.conn=sqlite3.connect(self.path); self.conn.row_factory=sqlite3.Row; self._setup()
    def _setup(self)->None:
        self.conn.executescript('''PRAGMA foreign_keys = ON; CREATE TABLE IF NOT EXISTS waf_decision_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT NOT NULL,event_type TEXT NOT NULL,schema_version TEXT NOT NULL,decision TEXT NOT NULL CHECK (decision IN ('allow','alert','block')),risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),event_json TEXT NOT NULL,occurred_at TEXT NOT NULL,expires_at TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_waf_decision_events_request ON waf_decision_events(request_id); CREATE INDEX IF NOT EXISTS idx_waf_decision_events_expiry ON waf_decision_events(expires_at); CREATE TABLE IF NOT EXISTS waf_security_audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT,actor_id TEXT NOT NULL,role TEXT NOT NULL,action TEXT NOT NULL,target_type TEXT NOT NULL,target_id TEXT NOT NULL,status TEXT NOT NULL,reason TEXT NOT NULL,metadata_json TEXT NOT NULL,occurred_at TEXT NOT NULL,expires_at TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_waf_audit_expiry ON waf_security_audit(expires_at); CREATE TABLE IF NOT EXISTS waf_store_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at TEXT NOT NULL); INSERT OR REPLACE INTO waf_store_meta(key,value,updated_at) VALUES ('schema_version','storage-v1',datetime('now'));'''); self.conn.commit()
    @staticmethod
    def _validate_event(event:Mapping[str,Any])->None:
        bad=_contains_forbidden(event)
        if bad: raise StoragePrivacyError(f'raw request field {bad!r} cannot be persisted')
        evidence=event.get('evidence')
        if isinstance(evidence,Mapping):
            privacy=evidence.get('privacy',{})
            for key in ('raw_payload_retained','raw_headers_retained','raw_query_retained'):
                if privacy.get(key) is not False: raise StoragePrivacyError(f'evidence privacy flag {key} must be false')
    def publish(self,event:Mapping[str,Any],retention_days:int=30)->int:
        self._validate_event(event)
        if retention_days<=0: raise ValueError('retention_days must be positive')
        now=datetime.now(timezone.utc); occurred_at=str(event.get('occurred_at') or now.isoformat()); expires_at=(now+timedelta(days=retention_days)).isoformat()
        row=(str(event.get('request_id','')),str(event.get('event_type','waf.decision')),str(event.get('schema_version','event-v2')),str(event.get('decision','allow')),float(event.get('risk_score',0.0)),_json(event),occurred_at,expires_at)
        cur=self.conn.execute('INSERT INTO waf_decision_events(request_id,event_type,schema_version,decision,risk_score,event_json,occurred_at,expires_at) VALUES (?,?,?,?,?,?,?,?)',row); self.conn.commit(); return int(cur.lastrowid)
    def audit(self,actor_id:str,role:str,action:str,target_type:str,target_id:str,status:str,reason:str,metadata:Mapping[str,Any]|None=None,retention_days:int=365)->int:
        metadata=dict(metadata or {}); bad=_contains_forbidden(metadata)
        if bad: raise StoragePrivacyError(f'raw request field {bad!r} cannot be persisted in audit metadata')
        if not actor_id.strip() or not role.strip() or not reason.strip(): raise ValueError('actor_id, role and reason are required for an audit record')
        now=datetime.now(timezone.utc); expires_at=now+timedelta(days=retention_days)
        cur=self.conn.execute('INSERT INTO waf_security_audit(actor_id,role,action,target_type,target_id,status,reason,metadata_json,occurred_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?,?)',(actor_id,role,action,target_type,target_id,status,reason[:240],_json(metadata),now.isoformat(),expires_at.isoformat())); self.conn.commit(); return int(cur.lastrowid)
    def purge_expired(self,now:datetime|None=None)->dict[str,int]:
        now=now or datetime.now(timezone.utc); stamp=now.isoformat(); e=self.conn.execute('DELETE FROM waf_decision_events WHERE expires_at <= ?',(stamp,)).rowcount; a=self.conn.execute('DELETE FROM waf_security_audit WHERE expires_at <= ?',(stamp,)).rowcount; self.conn.commit(); return {'decision_events':max(0,int(e)),'audit_records':max(0,int(a))}
    def count(self,table:str)->int:
        if table not in {'waf_decision_events','waf_security_audit'}: raise ValueError('unsupported table')
        return int(self.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
    def close(self)->None:self.conn.close()
