"""
Differentiable-solver multiple-shooting inverse model for Gray-Scott diffusivities.

WHY (see experiments/README.md for the full diagnosis):
  The data lives on a graded 101x101 grid whose spacing is ~0.018 in the centre.
  Any method that needs d2u/dx2 of the data (vanilla PINN, consistency PINN, weak form)
  inherits an O(h^2) curvature error there, which caps Du's L_paper at ~15-19%
  even with the *exact* functional form of Du. This model never differentiates
  the data: it advances data snapshots with a fine-grid finite-volume solver
  (differentiable, written in torch) and matches the predicted *values* of u and v
  at the next data times. Gradients w.r.t. Du_net / Dv flow through the solver.

  - one training stage (no pre-training, no frozen phases)
  - short windows (default 20 time units) -> smooth, well-conditioned loss
  - explicit time step chosen from the current max(D) -> always stable
  - convergence controlled by loss plateau, not by a hand-picked epoch count

Usage (GPU):
  python experiments/diffsolver_inverse.py --param 1
  python experiments/diffsolver_inverse.py --param 3 --N 192 --iters 3000
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gs_solver import spline_matrix, divflux2 as _df2, divflux4 as _df4, STAB as _STAB


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FK = {1: (0.037, 0.06), 2: (0.014, 0.045), 3: (0.018, 0.051)}

ap = argparse.ArgumentParser()
ap.add_argument("--param", type=int, default=1)
ap.add_argument("--F", type=float, default=None, help="override feed rate")
ap.add_argument("--k", type=float, default=None, help="override kill rate")
ap.add_argument("--data", type=str, default=None, help="path to COMSOL txt (default data/data_{param}.txt)")
ap.add_argument("--N", type=int, default=160, help="fine solver grid (N x N cells)")
ap.add_argument("--horizon", type=int, default=2, help="window length in data intervals (x10 time units)")
ap.add_argument("--order", type=int, default=4, choices=[2, 4], help="spatial order of the solver")
ap.add_argument("--weight", type=str, default="uniform", choices=["uniform", "area"], help="data-node weights")
ap.add_argument("--burn", type=int, default=0, help="first data intervals of each window not scored")
ap.add_argument("--batch", type=int, default=8, help="windows per iteration")
ap.add_argument("--t_start", type=float, default=300.0, help="earliest window start (fields flat before)")
ap.add_argument("--iters", type=int, default=4000, help="max iterations")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--patience", type=int, default=400, help="stop when smoothed loss stalls this long")
ap.add_argument("--D_scale", type=float, default=1e-5)
ap.add_argument("--cfl", type=float, default=0.8, help="fraction of explicit stability limit")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--synthetic", type=str, default=None, help="npz from make_synthetic.py instead of COMSOL data")
ap.add_argument("--start", type=str, default="data", choices=["data", "fine"],
                help="window start state: spline of the data nodes, or exact fine field (synthetic only)")
ap.add_argument("--cache", type=str, default=None, help="npz cache dir (default experiments/cache)")
ap.add_argument("--out", type=str, default=None)
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device(args.device)
F, k = FK.get(args.param, (None, None))
F = args.F if args.F is not None else F
k = args.k if args.k is not None else k
out_dir = args.out or os.path.join(HERE, "results", (f"synthetic_p{args.param}_start-{args.start}_N{args.N}_o{args.order}{args.tag}" if args.synthetic else f"diffsolver_p{args.param}_N{args.N}_H{args.horizon}_o{args.order}_{args.weight}{args.tag}"))
os.makedirs(out_dir, exist_ok=True)
print(f"device {dev} | case {args.param} F={F} k={k} | out {out_dir}", flush=True)

# ------------------------------------------------------------------ data
def load_grid():
    cache_dir = args.cache or os.path.join(HERE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    src = args.data or os.path.join(ROOT, "data", f"data_{args.param}.txt")
    cpath = os.path.join(cache_dir, os.path.basename(src).replace(".txt", "_grid.npz"))
    if os.path.exists(cpath):
        g = np.load(cpath); return g["xs"], g["ys"], g["t"], g["U"], g["V"]
    a = np.loadtxt(src, comments="%")
    xy, f = a[:, :2], a[:, 2:]
    xs = np.unique(np.round(xy[:, 0], 12)); ys = np.unique(np.round(xy[:, 1], 12))
    assert len(xs) * len(ys) == len(xy), "data is not on a tensor-product grid"
    ix = np.searchsorted(xs, np.round(xy[:, 0], 12)); iy = np.searchsorted(ys, np.round(xy[:, 1], 12))
    nt = f.shape[1] // 2
    U = np.zeros((nt, len(xs), len(ys))); V = np.zeros_like(U)
    U[:, ix, iy] = f[:, 0::2].T; V[:, ix, iy] = f[:, 1::2].T
    t = np.arange(nt) * 10.0
    np.savez(cpath, xs=xs, ys=ys, t=t, U=U, V=V)
    return xs, ys, t, U, V

if args.synthetic:
    syn = np.load(args.synthetic)
    xs, ys, tdat, U, V = syn["xs"], syn["ys"], syn["t"], syn["U"], syn["V"]
    F, k = float(syn["F"]), float(syn["k"])
    print(f"synthetic data {args.synthetic} (generated on {int(syn['Ngen'])}^2, F={F} k={k})", flush=True)
else:
    xs, ys, tdat, U, V = load_grid()
dt_data = float(tdat[1] - tdat[0])
Lx, Ly = xs[-1] - xs[0], ys[-1] - ys[0]
N = args.N
hx, hy = Lx / N, Ly / N
xc = xs[0] + (np.arange(N) + 0.5) * hx
yc = ys[0] + (np.arange(N) + 0.5) * hy

# linear (cubic-spline) interpolation operators: data grid <-> solver cell centres
Qx, Qy = spline_matrix(xs, xc), spline_matrix(ys, yc)                # data -> fine
Px, Py = spline_matrix(xc, xs), spline_matrix(yc, ys)                # fine -> data
T32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
Qx, Qy, Px, Py = map(T32, (Qx, Qy, Px, Py))
Ud, Vd = T32(U), T32(V)
sig_u, sig_v = float(U.std()), float(V.std())
if args.start == "fine":
    assert args.synthetic, "--start fine needs --synthetic"
    Ns = syn["Ufine"].shape[1]
    R = T32(spline_matrix((np.arange(Ns) + 0.5) / Ns, xc)) if Ns != N else torch.eye(N, device=dev)
    Ufine, Vfine = R @ T32(syn["Ufine"]) @ R.T, R @ T32(syn["Vfine"]) @ R.T   # re-grid if solver N differs

# ------------------------------------------------------------------ unknowns
D_SCALE = args.D_scale
def norm(z, lo, hi): return 2.0 * (z - lo) / (hi - lo) - 1.0

class Du_net(nn.Module):           # same architecture as the consistency PINN
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                     nn.Linear(32, 1))
    def forward(self, X):
        return D_SCALE * nn.functional.softplus(self.network(X))

class Dv_net(nn.Module):
    def __init__(self):
        super().__init__(); self.Dv_raw = nn.Parameter(torch.tensor(1.0))
    def value(self): return D_SCALE * nn.functional.softplus(self.Dv_raw)

Du_nn, Dv_nn = Du_net().to(dev), Dv_net().to(dev)

def grid_in(xv, yv):
    X, Y = torch.meshgrid(T32(xv), T32(yv), indexing="ij")
    return torch.stack([norm(X, xs[0], xs[-1]), norm(Y, ys[0], ys[-1])], -1)
if args.order == 2:
    Xface_x = grid_in(xs[0] + np.arange(1, N) * hx, yc)   # (N-1, N): interior faces in x
    Xface_y = grid_in(xc, ys[0] + np.arange(1, N) * hy)   # (N, N-1)
else:   # 4th order: all faces m*h, m=0..N (walls included)
    Xface_x = grid_in(xs[0] + np.arange(0, N + 1) * hx, yc)   # (N+1, N)
    Xface_y = grid_in(xc, ys[0] + np.arange(0, N + 1) * hy)   # (N, N+1)

# ------------------------------------------------------------------ solver
divflux = (lambda u, a, b: _df2(u, a, b, hx, hy)) if args.order == 2 else (lambda u, a, b: _df4(u, a, b, hx, hy))
STAB = _STAB[args.order]

def rollout(u, v, Dfx, Dfy, Dv, n_sub, dt):
    for _ in range(n_sub):
        uv2 = u * v * v
        u, v = (u + dt * (divflux(u, Dfx, Dfy) - uv2 + F * (1 - u)),
                v + dt * (Dv * divflux(v, 1.0, 1.0) + uv2 - (F + k) * v))
    return u, v

def dual_widths(z):
    w = np.zeros_like(z); w[1:] += np.diff(z) / 2; w[:-1] += np.diff(z) / 2; return w
if args.weight == "area":   # each data node counts by the area it represents (graded mesh!)
    Wn = np.outer(dual_widths(xs), dual_widths(ys)); Wnode = T32(Wn / Wn.mean())
else:
    Wnode = torch.ones(len(xs), len(ys), device=dev)

to_fine = lambda A: Qx @ A @ Qy.T
to_data = lambda A: Px @ A @ Py.T

def window_loss(i0):
    Dfx, Dfy = Du_nn(Xface_x)[..., 0], Du_nn(Xface_y)[..., 0]
    Dv = Dv_nn.value()
    Dmax = max(Dfx.max().item(), Dv.item())
    lim = 2.0 / (STAB * Dmax * (1 / hx**2 + 1 / hy**2))     # forward-Euler stability limit
    n_sub = max(1, math.ceil(dt_data / (args.cfl * lim)))
    dt = dt_data / n_sub
    if args.start == "fine":
        u, v = Ufine[i0], Vfine[i0]
    else:
        u, v = to_fine(Ud[i0]), to_fine(Vd[i0])
    loss = 0.0
    for j in range(1, args.horizon + 1):
        u, v = torch.utils.checkpoint.checkpoint(rollout, u, v, Dfx, Dfy, Dv, n_sub, dt, use_reentrant=False)
        if j <= args.burn:      # let the interpolation error of the start snapshot decay first
            continue
        lu = (Wnode * ((to_data(u) - Ud[i0 + j]) / sig_u).pow(2)).mean()
        lv = (Wnode * ((to_data(v) - Vd[i0 + j]) / sig_v).pow(2)).mean()
        loss = loss + lu + lv
    return loss / (args.horizon - args.burn), n_sub

# ------------------------------------------------------------------ monitoring (true D used ONLY for reporting)
Xm, Ym = np.meshgrid(xs, ys, indexing="ij")
Du_true = 2e-5 * (1 + np.sin(np.pi * Xm) * np.sin(np.pi * Ym))
Xdat = grid_in(xs, ys)
@torch.no_grad()
def Du_on_data(): return Du_nn(Xdat)[..., 0].cpu().numpy()
def L_paper(pred): return float(np.sqrt(np.mean((pred - Du_true) ** 2) / np.var(Du_true)))

# ------------------------------------------------------------------ training (single stage)
i_lo = int(np.searchsorted(tdat, args.t_start))
i_hi = len(tdat) - 1 - args.horizon
opt = torch.optim.Adam([{"params": Du_nn.parameters()}, {"params": Dv_nn.parameters()}], lr=args.lr)
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 4, min_lr=args.lr / 100)
hist = {"it": [], "loss": [], "Lpaper": [], "Dv": []}
ema, best, since_best = None, float("inf"), 0
t0 = time.time()
for it in range(1, args.iters + 1):
    starts = torch.randint(i_lo, i_hi + 1, (args.batch,))
    loss, n_sub = window_loss(starts)
    opt.zero_grad(); loss.backward(); opt.step()
    lv = loss.item()
    ema = lv if ema is None else 0.98 * ema + 0.02 * lv
    plateau.step(ema)
    if ema < best * (1 - 1e-3): best, since_best = ema, 0
    else: since_best += 1
    if it % 25 == 0 or it == 1:
        Lp = L_paper(Du_on_data())
        hist["it"].append(it); hist["loss"].append(lv); hist["Lpaper"].append(Lp); hist["Dv"].append(Dv_nn.value().item())
        print(f"it {it:5d} | loss {lv:.3e} ema {ema:.3e} | Du L_paper {100*Lp:6.2f}% | Dv {Dv_nn.value().item():.4e} "
              f"| lr {opt.param_groups[0]['lr']:.1e} | substeps {n_sub} | {time.time()-t0:.0f}s", flush=True)
    if since_best > args.patience and opt.param_groups[0]["lr"] <= args.lr / 100 * 1.01:
        print(f"converged (loss plateau) at it {it}"); break

# ------------------------------------------------------------------ outputs
Dp = Du_on_data(); Lp = L_paper(Dp)
res = dict(case=args.param, F=F, k=k, N=N, horizon=args.horizon, iters_run=it, time_s=time.time() - t0,
           Du_Lpaper_pct=100 * Lp, Du_rel_L2_pct=100 * float(np.linalg.norm(Dp - Du_true) / np.linalg.norm(Du_true)),
           Dv=Dv_nn.value().item(), Dv_rel_err_pct=100 * (Dv_nn.value().item() / 1e-5 - 1), final_loss_ema=ema)
print(json.dumps(res, indent=2))
json.dump(dict(result=res, args=vars(args), hist=hist), open(os.path.join(out_dir, "result.json"), "w"), indent=1)
torch.save({"Du_nn": Du_nn.state_dict(), "Dv_nn": Dv_nn.state_dict(), "args": vars(args)}, os.path.join(out_dir, "D_model.pt"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a, Z, ttl in [(ax[0], Du_true, "Du true"), (ax[1], Dp, f"Du learned (L_paper {100*Lp:.2f}%)"),
                      (ax[2], Dp - Du_true, "error")]:
        c = a.contourf(Xm, Ym, Z, 50); plt.colorbar(c, ax=a); a.set_title(ttl); a.set_aspect("equal")
    ax[3].semilogy(hist["it"], np.array(hist["Lpaper"]) * 100); ax[3].set_xlabel("iteration"); ax[3].set_ylabel("Du L_paper [%]"); ax[3].grid(True, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Du.png"), dpi=80)
except Exception as e:
    print("plot failed:", e)
