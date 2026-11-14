@echo off
rem Sparse / irregular data sweep (case 1, COMSOL). Logs: experiments\results\logs\sparse_*.log
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
set S=experiments\sparse_inverse.py --param 1 --N 160 --iters 2000
if not exist experiments\results\logs mkdir experiments\results\logs
start "" /b cmd /c "%PY% -u %S% --space_frac 1.0 --time_frac 1.0 > experiments\results\logs\sparse_p1_s1_t1.log 2>&1"
start "" /b cmd /c "%PY% -u %S% --space_frac 0.1 --space_mode fixed --time_frac 1.0 > experiments\results\logs\sparse_p1_s0.1f_t1.log 2>&1"
start "" /b cmd /c "%PY% -u %S% --space_frac 0.02 --space_mode random --time_frac 1.0 > experiments\results\logs\sparse_p1_s0.02r_t1.log 2>&1"
start "" /b cmd /c "%PY% -u %S% --space_frac 0.1 --space_mode fixed --time_frac 0.333 --max_gap 60 > experiments\results\logs\sparse_p1_s0.1f_t0.333.log 2>&1"
