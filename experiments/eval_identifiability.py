"""
Is the remaining Du error an OPTIMISATION problem or an IDENTIFIABILITY problem?

Evaluates the shooting loss of a trained run on a fixed, deterministic set of windows for
  (1) the learned Du_net / Dv, (2) the true Du / Dv, (3) points on the straight line between them.
  - loss(true) clearly below loss(learned)  -> optimiser did not reach the minimum
  - loss(true) ~ loss(learned)              -> the data cannot tell the two D fields apart

  python experiments/eval_identifiability.py --run experiments/results/synthetic_p1_start-fine_N160_o4
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gs_solver import spline_matrix, divflux2, divflux4, STAB

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="result folder containing D_model.pt")
ap.add_argument("--every", type=int, default=5, help="window start stride (data intervals)")
ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--N", type=int, default=None, help="override solver grid (floor test)")
ap.add_argument("--cfl", type=float, default=None, help="override time-step fraction (floor test)")
ap.add_argument("--only_true", action="store_true", help="only evaluate the loss at the true D")
a = ap.parse_args()
dev = torch.device(a.device)
ck = torch.load(os.path.join(a.run, "D_model.pt"), map_location=dev)
args = argparse.Namespace(**{"synthetic": None, "order": 2, "start": "data", "burn": 0, **ck["args"]})
if a.N: args.N = a.N
if a.cfl: args.cfl = a.cfl
T32 = lambda z: torch.tensor(z, dtype=torch.float32, device=dev)

if args.synthetic:
    syn = np.load(args.synthetic if os.path.isabs(args.synthetic) else os.path.join(os.path.dirname(HERE), args.synthetic))
    xs, ys, tdat, U, V = syn["xs"], syn["ys"], syn["t"], syn["U"], syn["V"]; F, k = float(syn["F"]), float(syn["k"])
else:
    g = np.load(os.path.join(HERE, "cache", f"data_{args.param}_grid.npz"))
    xs, ys, tdat, U, V = g["xs"], g["ys"], g["t"], g["U"], g["V"]
    F, k = {1: (0.037, 0.06), 2: (0.014, 0.045), 3: (0.018, 0.051)}[args.param]
N = args.N; hx, hy = (xs[-1] - xs[0]) / N, (ys[-1] - ys[0]) / N
xc = xs[0] + (np.arange(N) + 0.5) * hx; yc = ys[0] + (np.arange(N) + 0.5) * hy
Qx, Qy, Px, Py = map(T32, (spline_matrix(xs, xc), spline_matrix(ys, yc), spline_matrix(xc, xs), spline_matrix(yc, ys)))
Ud, Vd = T32(U), T32(V); sig_u, sig_v = float(U.std()), float(V.std())
use_fine = getattr(args, "start", "data") == "fine"
if use_fine:
    Ns = syn["Ufine"].shape[1]; xsrc = (np.arange(Ns) + 0.5) / Ns
    R = T32(spline_matrix(xsrc, xc)) if Ns != N else torch.eye(N, device=dev)   # re-grid exact fields if N overridden
    Ufine, Vfine = R @ T32(syn["Ufine"]) @ R.T, R @ T32(syn["Vfine"]) @ R.T
dt_data = float(tdat[1] - tdat[0])

D_SCALE = args.D_scale
class Du_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, X): return D_SCALE * nn.functional.softplus(self.network(X))
Du_nn = Du_net().to(dev); Du_nn.load_state_dict(ck["Du_nn"])
Dv_learned = D_SCALE * nn.functional.softplus(ck["Dv_nn"]["Dv_raw"]).item()

if args.order == 2:
    fx_x, fx_y = xs[0] + np.arange(1, N) * hx, yc; fy_x, fy_y = xc, ys[0] + np.arange(1, N) * hy
else:
    fx_x, fx_y = xs[0] + np.arange(0, N + 1) * hx, yc; fy_x, fy_y = xc, ys[0] + np.arange(0, N + 1) * hy
def norm(z, lo, hi): return 2.0 * (z - lo) / (hi - lo) - 1.0
def net_on(xv, yv):
    X, Y = torch.meshgrid(T32(xv), T32(yv), indexing="ij")
    with torch.no_grad(): return Du_nn(torch.stack([norm(X, xs[0], xs[-1]), norm(Y, ys[0], ys[-1])], -1))[..., 0]
def true_on(xv, yv):
    X, Y = np.meshgrid(xv, yv, indexing="ij"); return T32(2e-5 * (1 + np.sin(np.pi * X) * np.sin(np.pi * Y)))
L = (net_on(fx_x, fx_y), net_on(fy_x, fy_y)); Tr = (true_on(fx_x, fx_y), true_on(fy_x, fy_y))
div = (lambda u, p, q: divflux2(u, p, q, hx, hy)) if args.order == 2 else (lambda u, p, q: divflux4(u, p, q, hx, hy))

i_lo = int(np.searchsorted(tdat, args.t_start)); i_hi = len(tdat) - 1 - args.horizon
starts = torch.arange(i_lo, i_hi + 1, a.every)

@torch.no_grad()
def loss_of(Dfx, Dfy, Dv):
    Dmax = max(Dfx.max().item(), Dv)
    lim = 2.0 / (STAB[args.order] * Dmax * (1 / hx**2 + 1 / hy**2))
    n_sub = max(1, math.ceil(dt_data / (args.cfl * lim))); dt = dt_data / n_sub
    tot = 0.0
    for chunk in starts.split(16):
        u, v = (Ufine[chunk], Vfine[chunk]) if use_fine else (Qx @ Ud[chunk] @ Qy.T, Qx @ Vd[chunk] @ Qy.T)
        for j in range(1, args.horizon + 1):
            for _ in range(n_sub):
                uv2 = u * v * v
                u, v = u + dt * (div(u, Dfx, Dfy) - uv2 + F * (1 - u)), v + dt * (Dv * div(v, 1.0, 1.0) + uv2 - (F + k) * v)
            if j <= args.burn: continue
            tot += (((Px @ u @ Py.T - Ud[chunk + j]) / sig_u).pow(2).mean((1, 2)).sum()
                    + ((Px @ v @ Py.T - Vd[chunk + j]) / sig_v).pow(2).mean((1, 2)).sum()).item()
    return tot / (len(starts) * (args.horizon - args.burn))

if a.only_true:
    print(f"N={N} cfl={args.cfl} order={args.order}: loss at TRUE D = {loss_of(Tr[0], Tr[1], 1e-5):.4e}")
    sys.exit(0)
res = {}
for s in (0.0, 0.25, 0.5, 0.75, 1.0):   # s=0 learned, s=1 true
    Dfx = (1 - s) * L[0] + s * Tr[0]; Dfy = (1 - s) * L[1] + s * Tr[1]; Dv = (1 - s) * Dv_learned + s * 1e-5
    res[s] = loss_of(Dfx, Dfy, Dv)
    print(f"s={s:.2f} ({'learned' if s == 0 else 'true' if s == 1 else 'mix'}): loss {res[s]:.4e}", flush=True)
ratio = res[1.0] / res[0.0]
verdict = ("OPTIMISATION: true D fits clearly better" if ratio < 0.8 else
           "IDENTIFIABILITY: data barely distinguishes learned vs true D" if ratio < 1.25 else
           "MODEL MISMATCH: true D fits worse than learned (solver/data error absorbed into D)")
print(f"loss(true)/loss(learned) = {ratio:.3f}  ->  {verdict}")
json.dump({"loss_line": res, "ratio_true_over_learned": ratio, "verdict": verdict},
          open(os.path.join(a.run, "identifiability.json"), "w"), indent=1)
