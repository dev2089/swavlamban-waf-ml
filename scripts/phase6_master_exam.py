#!/usr/bin/env python3
from __future__ import annotations
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); RESULT=ROOT/'phase6_master_exam_result.json'; CUTOFF=9.9

def run(name,cmd):
 p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); return {'name':name,'passed':p.returncode==0,'returncode':p.returncode,'tail':(p.stdout+'\n'+p.stderr).strip()[-2500:]}

def smoke():
 from waf.core.config import WAFConfig
 from waf.core.models import RequestEnvelope
 from waf.edge.pipeline import EdgeWAF
 waf=EdgeWAF(WAFConfig(pipeline_version='phase6-master-exam')); req=RequestEnvelope('phase6-smoke','GET','https','example.test','/','q=1 union select x'); result=waf.analyze(req); c=waf.recommend_rules(result)[0]; v=waf.validate_rule(c.rule_id)
 if not v.valid: raise AssertionError(v.errors)
 waf.approve_rule(c.rule_id,'phase6-reviewer'); d=waf.deploy_approved_rules(); post=waf.analyze(req)
 if c.rule_id not in d['active_rule_ids'] or c.rule_id not in post.rule_ids or post.decision.value!='block': raise AssertionError('managed rule enforcement failed')
 rb=waf.rollback_rules(d['deployment_id'])
 if waf.rule_lifecycle.active_rule_ids(): raise AssertionError('rollback failed to restore empty baseline')
 return {'candidate_rule_id':c.rule_id,'deployment':d,'rollback':rb,'post_rollback_active_rule_ids':list(waf.rule_lifecycle.active_rule_ids()),'raw_request_material_recorded':False}

def privacy():
 text=(ROOT/'waf/rules/lifecycle.py').read_text(); bad=[x for x in ['request.body','request.query','request.headers','request.host','request.source_ip'] if x in text]
 if bad or 'phase5-decision-evidence' not in text or '_SAFE_ACTIONS = {"block"}' not in text: raise AssertionError(f'privacy/safety contract failed: {bad}')
 return {'passed':True,'forbidden_direct_access':bad,'deployable_actions':['block'],'raw_request_fields_persisted':False}

def main():
 checks=[run('full-regression',[sys.executable,'-m','pytest','-q']),run('compileall',[sys.executable,'-m','compileall','-q','waf','tests','scripts']),run('phase6-lifecycle-tests',[sys.executable,'-m','pytest','-q','tests/test_phase6_rule_lifecycle.py'])]; critical=[]
 try: s=smoke(); checks.append({'name':'lifecycle-smoke','passed':True,'tail':json.dumps(s,sort_keys=True)})
 except Exception as e: s={}; checks.append({'name':'lifecycle-smoke','passed':False,'tail':repr(e)}); critical.append('lifecycle-smoke')
 try: checks.append({'name':'privacy-static-gate','passed':True,'tail':json.dumps(privacy(),sort_keys=True)})
 except Exception as e: checks.append({'name':'privacy-static-gate','passed':False,'tail':repr(e)}); critical.append('privacy-static-gate')
 score=round(sum(c['passed'] for c in checks)/len(checks)*10,2)
 if score<CUTOFF: critical.append('score-below-cutoff')
 out={'phase':6,'status':'PASS' if score>=CUTOFF and not critical else 'FAIL','score':score,'cutoff':CUTOFF,'critical_defects':critical,'checks':checks,'smoke':s,'evaluation_scope':'repository regression, compile gate, Phase 6 lifecycle tests, deterministic lifecycle smoke and privacy/static gate','notes':['Managed rules use a bounded feature_threshold DSL only.','Only block is deployable on the current edge seam.','Generated candidates contain evidence metadata and feature-level matchers, not raw request material.','Lifecycle state is in-memory plus portable ledger evidence; production storage/auth/RBAC remains future work.']}
 RESULT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2)); return 0 if out['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
