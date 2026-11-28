"""Queue runner for the 32-dataset sweep (data/<bc>/D<d>/fk<p>): runs jobs with a fixed number in parallel,
skips jobs whose result already exists, and prints a summary table.

  python -u experiments/run_sweep.py                       # our method on all 32 datasets, 6 at a time
  python -u experiments/run_sweep.py --methods diffsolver pinn --parallel 4
  python -u experiments/run_sweep.py --bcs dirichlet --dcases 4 --params 1 4     # a subset
  python -u experiments/run_sweep.py --summary             # only the table (also written to results/sweep_summary.*)
  python -u experiments/run_sweep.py --dry                 # list the jobs, run nothing

Settings = those of the "all cases" table (order 4, N 160, horizon 2, uniform weights, <=3000 iterations).
Logs: experiments/results/logs/sweep_<method>_<bc>_D<d>_fk<p>.log
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from gs_data import case_tag, data_path, FK, BCS, DCASES

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=["diffsolver"], choices=["diffsolver", "pinn"])
ap.add_argument("--bcs", nargs="+", default=list(BCS), choices=BCS)
ap.add_argument("--dcases", nargs="+", type=int, default=list(DCASES), choices=DCASES)
ap.add_argument("--params", nargs="+", type=int, default=sorted(FK), choices=sorted(FK))
ap.add_argument("--parallel", type=int, default=6, help="jobs at the same time on the GPU")
ap.add_argument("--iters", type=int, default=3000, help="diffsolver max iterations")
ap.add_argument("--summary", action="store_true", help="only print the summary table")
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()

PY = sys.executable
RES = os.path.join(HERE, "results"); LOGS = os.path.join(RES, "logs"); os.makedirs(LOGS, exist_ok=True)


def job(method, bc, d, p):
    tag = case_tag(bc, d)
    common = ["--param", str(p), "--bc", bc, "--dcase", str(d)]
    if method == "diffsolver":
        out = os.path.join(RES, f"diffsolver_p{p}{tag}_N160_H2_o4_uniform")
        cmd = [PY, "-u", os.path.join(HERE, "diffsolver_inverse.py"), *common, "--order", "4", "--N", "160",
               "--horizon", "2", "--weight", "uniform", "--iters", str(args.iters)]
        done = os.path.join(out, "result.json")
    else:
        out = os.path.join(RES, f"cons_pinn_p{p}{tag}")
        cmd = [PY, "-u", os.path.join(HERE, "cons_pinn_history.py"), *common]
        done = os.path.join(out, "model.pt")
    log = os.path.join(LOGS, f"sweep_{method}_{bc}_D{d}_fk{p}.log")
    return dict(method=method, bc=bc, d=d, p=p, out=out, cmd=cmd, done=done, log=log)


JOBS = [job(m, bc, d, p) for m in args.methods for bc in args.bcs for d in args.dcases for p in args.params]


def lpaper(j):
    """Final Du L_paper [%] of a finished job, else None."""
    try:
        if j["method"] == "diffsolver":
            return json.load(open(j["done"]))["result"]["Du_Lpaper_pct"]
        h = json.load(open(os.path.join(j["out"], "history.json")))
        r = h.get("result") or {}
        for key in ("Du_Lnorm_grid_pct",):
            if key in r: return r[key]
        return 100 * h["hist"]["L_norm_grid"][-1]
    except Exception:
        return None


def summary():
    lines = []
    for m in args.methods:
        lines.append(f"\n{m}: Du L_paper [%]   (rows: bc / D field, columns: (F,k) pair)")
        lines.append(f"{'':16s}" + "".join(f"{'fk' + str(p):>9s}" for p in args.params))
        for bc in args.bcs:
            for d in args.dcases:
                row = [lpaper(job(m, bc, d, p)) for p in args.params]
                lines.append(f"{bc + ' D' + str(d):16s}" + "".join(f"{v:9.2f}" if v is not None else f"{'-':>9s}" for v in row))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(RES, "sweep_summary.txt"), "w") as f: f.write(txt + "\n")


def run(j):
    if os.path.exists(j["done"]):
        print(f"skip (done)  {os.path.basename(j['out'])}", flush=True); return
    if not os.path.exists(data_path(j["p"], j["bc"], j["d"])):
        print(f"skip (no data, git lfs pull?)  {j['bc']} D{j['d']} fk{j['p']}", flush=True); return
    t0 = time.time()
    print(f"start {time.strftime('%H:%M')}  {os.path.basename(j['out'])}", flush=True)
    with open(j["log"], "w") as f:
        rc = subprocess.call(j["cmd"], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    v = lpaper(j)
    print(f"{'done ' if rc == 0 else 'FAIL '} {time.strftime('%H:%M')}  {os.path.basename(j['out'])}  "
          f"rc={rc}  {(time.time() - t0) / 60:.0f} min  L_paper={'-' if v is None else f'{v:.2f}%'}", flush=True)


if args.summary:
    summary(); sys.exit(0)
print(f"{len(JOBS)} jobs, {args.parallel} in parallel, python {PY}", flush=True)
if args.dry:
    for j in JOBS: print(("done  " if os.path.exists(j["done"]) else "todo  ") + " ".join(j["cmd"][2:]))
    sys.exit(0)
with ThreadPoolExecutor(args.parallel) as ex:
    list(ex.map(run, JOBS))
summary()
