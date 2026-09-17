import subprocess
import sys

COMMANDS = [
    [sys.executable, '-m', 'compileall', '-q', 'waf', 'run_proxy.py'],
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_phase2.py'],
    ['bash', 'tests/test_nginx_integration.sh'],
]

for command in COMMANDS:
    print('$', ' '.join(command))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

print('PHASE2_SELF_TEST=PASS')
