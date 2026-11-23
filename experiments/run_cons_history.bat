@echo off
rem Consistency PINN, case 1: pre-training of u,v then stage B directly (fresh Du), L_normalized logged every 10 steps.
rem Output: experiments\results\cons_pinn_p1\ (history.json, Du.png, model.pt). Log: results\logs\cons_pinn_p1.log
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
set L=experiments\results\logs
if not exist %L% mkdir %L%
start "" /b cmd /c "%PY% -u experiments\cons_pinn_history.py --param 1 > %L%\cons_pinn_p1.log 2>&1"
