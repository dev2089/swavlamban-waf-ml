from __future__ import annotations
import json,time
from statistics import mean,median
from waf.core.config import WAFConfig
from waf.core.models import RequestEnvelope
from waf.edge.policy import EdgeDecisionPolicy
from waf.explainability import build_decision_evidence
from waf.edge.pipeline import EdgeWAF

def pct(v,p):
    x=sorted(v); return x[min(len(x)-1,max(0,int(round((len(x)-1)*p))))] if x else 0.0

def run(samples=100):
    waf=EdgeWAF(WAFConfig(pipeline_version="phase5"))
    core=[]; evidence=[]
    for i in range(samples):
        q="q=1%20UNION%20SELECT%20password%20FROM%20users" if i%10==0 else f"q={i}&page=2"
        req=RequestEnvelope(f"bench-{i}","GET","https","example.test",f"/api/item/{i%23}",q,source_ip=f"10.10.{i//250}.{i%250}",timestamp=1000+i)
        f=waf.features.extract(req)
        t=time.perf_counter()
        s=waf.signature_detector.detect(req,f); ml=waf.ml.detect(req,f); r=waf.policy.decide(req,(s,*ml),waf.config.pipeline_version)
        core.append((time.perf_counter()-t)*1000)
        t=time.perf_counter(); e=build_decision_evidence(req,f,r,waf.ml); evidence.append((time.perf_counter()-t)*1000)
        assert e.request_id==req.request_id
    cm=mean(core); em=mean(evidence)
    return {"samples":samples,"core_mean_ms":round(cm,4),"core_p50_ms":round(median(core),4),"core_p95_ms":round(pct(core,.95),4),"evidence_mean_ms":round(em,4),"evidence_p50_ms":round(median(evidence),4),"evidence_p95_ms":round(pct(evidence,.95),4),"evidence_time_percent_of_core":round((em/cm)*100 if cm else 0,2),"scope":"local deterministic benchmark; timing is environment-dependent and not a production latency claim"}
if __name__=="__main__": print(json.dumps(run(),indent=2))
