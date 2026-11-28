@echo off
rem Our method (diffsolver, order 4, N 160, horizon 2) on all 32 datasets, 6 runs at a time; finished runs are skipped,
rem so re-running this file resumes. Progress: experiments\results\logs\sweep.log; per-run logs: logs\sweep_*.log
rem Summary table at the end: experiments\results\sweep_summary.txt  (any time: python experiments\run_sweep.py --summary)
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
if not exist experiments\results\logs mkdir experiments\results\logs
start "" /b cmd /c "%PY% -u experiments\run_sweep.py --parallel 6 > experiments\results\logs\sweep.log 2>&1"
