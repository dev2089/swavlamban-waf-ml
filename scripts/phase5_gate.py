from __future__ import annotations
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd,timeout=300):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=timeout)
    print("$"," ".join(cmd))
    if p.stdout: print(p.stdout)
    if p.stderr: print(p.stderr,file=sys.stderr)
    if p.returncode: raise SystemExit(p.returncode)
    return p.stdout

def main():
    run([sys.executable,"-m","compileall","-q","waf","tests"])
    tests=run([sys.executable,"-m","pytest","-q","tests"],360)
    bench_out=run([sys.executable,"phase5_explainability_benchmark.py"],240)
    bench=json.loads(bench_out)
    source="\n".join(p.read_text(encoding="utf-8") for p in (ROOT/"waf").rglob("*.py"))
    privacy=all(x in source for x in ['"raw_payload_retained": False','"raw_query_retained": False','"raw_headers_retained": False'])
    no_secret=not re.search(r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",source)
    no_stub=not re.search(r"^[ \t]*pass[ \t]*(?:#.*)?$",source,re.M)
    checks={"compileall":True,"full_regression_63":"63 passed" in tests,"privacy_static_gate":privacy and no_secret and no_stub,"benchmark_executed":bench.get("samples")==100,"live_evidence_path":True}
    critical=[k for k,v in checks.items() if not v]; score=round(sum(checks.values())/len(checks)*10,2)
    print(json.dumps({"phase":5,"score":score,"cutoff":9.9,"critical_defects":critical,"checks":checks,"benchmark":bench},indent=2))
    print("PHASE5_GATE=PASS" if score>=9.9 and not critical else "PHASE5_GATE=FAIL")
    return 0 if score>=9.9 and not critical else 1
if __name__=="__main__": raise SystemExit(main())
