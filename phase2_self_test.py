import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "compileall", "-q", "waf", "run_proxy.py", "phase2_demo.py", "phase2_benchmark.py"],
    [sys.executable, "-m", "pytest", "-q", "tests"],
    [sys.executable, "phase2_demo.py"],
    [sys.executable, "phase2_benchmark.py", "--requests", "5000", "--concurrency", "100"],
    ["bash", "tests/test_nginx_integration.sh"],
]

for command in COMMANDS:
    print("$", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

print("PHASE2_SELF_TEST=PASS")
