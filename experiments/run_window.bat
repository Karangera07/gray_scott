@echo off
rem Window-length / late-scoring test for coarse sensors (case 1 COMSOL, 2nd order N=192), plus reruns of the two
rem stopped temporal runs. --burn B skips scoring the first B data intervals of every window. Logs: results\logs\window_*.log
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
set R=%PY% -u experiments\diffsolver_inverse.py --param 1 --order 2 --N 192 --iters 3000
set L=experiments\results\logs
if not exist %L% mkdir %L%
start "" /b cmd /c "%R% --sensor_spacing 0.04 --horizon_t 30 --burn 2 --tag _Ht30b2 > %L%\window_sp0.04_Ht30b2.log 2>&1"
start "" /b cmd /c "%R% --sensor_spacing 0.04 --horizon_t 50 --burn 4 --tag _Ht50b4 > %L%\window_sp0.04_Ht50b4.log 2>&1"
start "" /b cmd /c "%R% --sensor_spacing 0.04 --horizon_t 50 --burn 2 --tag _Ht50b2 > %L%\window_sp0.04_Ht50b2.log 2>&1"
start "" /b cmd /c "%R% --sensor_spacing 0.05 --horizon_t 50 --burn 4 --tag _Ht50b4 > %L%\window_sp0.05_Ht50b4.log 2>&1"
start "" /b cmd /c "%R% --sensor_spacing 0.07 --horizon_t 50 --burn 4 --tag _Ht50b4 > %L%\window_sp0.07_Ht50b4.log 2>&1"
start "" /b cmd /c "%R% --time_stride 10 --horizon_t 100 > %L%\window_rerun_stride10.log 2>&1"
start "" /b cmd /c "%R% --time_frac 0.3 --horizon_t 60 > %L%\window_rerun_tfrac0.3.log 2>&1"
