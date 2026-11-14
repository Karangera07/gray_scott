@echo off
rem Control experiment: synthetic data with known D (our solver), sampled like COMSOL, then inverted
rem with window start states from the data-node spline ("data") or the exact field ("fine").
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
if not exist experiments\results\logs mkdir experiments\results\logs
for %%P in (1 3) do %PY% -u experiments\make_synthetic.py --param %%P > experiments\results\logs\make_synthetic_p%%P.log 2>&1
for %%P in (1 3) do (
  for %%S in (data fine) do (
    start "" /b cmd /c "%PY% -u experiments\diffsolver_inverse.py --param %%P --synthetic experiments\cache\synthetic_p%%P.npz --start %%S --order 4 --N 160 --iters 3000 > experiments\results\logs\synthetic_p%%P_start-%%S.log 2>&1"
  )
)
