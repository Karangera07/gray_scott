@echo off
rem Launch the diffsolver sweep on the GPU box. Each job runs in the background and writes
rem experiments\results\logs\<name>.log plus its own result folder (result.json, Du.png, D_model.pt).
cd /d %~dp0..
set PY=C:\ProgramData\miniforge3\envs\stenosis-gpu\python.exe
if not exist experiments\results\logs mkdir experiments\results\logs
for %%P in (1 2 3) do (
  for %%W in (uniform area) do (
    start "" /b cmd /c "%PY% -u experiments\diffsolver_inverse.py --param %%P --order 4 --N 160 --horizon 2 --weight %%W --iters 3000 > experiments\results\logs\p%%P_o4_N160_%%W.log 2>&1"
  )
)
