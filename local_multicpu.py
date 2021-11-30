import sys
import subprocess

ncpu = 10
procs = []
for i in range(ncpu):
    proc = subprocess.Popen([sys.executable, 'script.py', '{}', '{}'.format(i)])
    procs.append(proc)

for proc in procs:
    proc.wait()