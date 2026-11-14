@echo off
rem Sparsity study, case 1 COMSOL, 2nd order N=192 (full-data baseline: results\diffsolver_p1_N192_H2 = 4.94%%).
rem Space: evenly spaced sensor sub-grids. Time: every m-th snapshot / random fraction. Logs: results\logs\sparsity_*.log
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
set R=%PY% -u experiments\diffsolver_inverse.py --param 1 --order 2 --N 192 --iters 3000
set L=experiments\results\logs
if not exist %L% mkdir %L%
for %%S in (0.02 0.03 0.04 0.05 0.07 0.1) do start "" /b cmd /c "%R% --sensor_spacing %%S --horizon_t 20 > %L%\sparsity_space%%S.log 2>&1"
start "" /b cmd /c "%R% --time_stride 2 --horizon_t 40 > %L%\sparsity_stride2.log 2>&1"
start "" /b cmd /c "%R% --time_stride 3 --horizon_t 30 > %L%\sparsity_stride3.log 2>&1"
start "" /b cmd /c "%R% --time_stride 5 --horizon_t 50 > %L%\sparsity_stride5.log 2>&1"
start "" /b cmd /c "%R% --time_stride 10 --horizon_t 100 > %L%\sparsity_stride10.log 2>&1"
start "" /b cmd /c "%R% --time_frac 0.3 --horizon_t 60 > %L%\sparsity_tfrac0.3.log 2>&1"
start "" /b cmd /c "%R% --sensor_spacing 0.04 --time_stride 3 --horizon_t 30 > %L%\sparsity_space0.04_stride3.log 2>&1"
