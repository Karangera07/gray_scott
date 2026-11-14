"""
Neural-field multiple shooting: Gray-Scott diffusivities from SPARSE / IRREGULAR space-time samples.

The full-data model (diffsolver_inverse.py) starts every window from a spline through all data nodes.
With sparse data that spline does not exist, so here a small Fourier-feature neural field
(u, v)(x, y, t) supplies the window start states:

  loss = fit( field , sparse samples )                                  # field follows the data
       + shoot( solver( field(t_i) ; Du, Dv , t_i -> t_j ) , samples(t_j) )   # dynamics, scored on REAL samples only

Gradients of the shooting term reach Du_net, Dv AND the field, so the dynamics also correct the
start states. The data are never differentiated. A short field-only warm-up comes first (cheap).

Sparsity controls:
  --space_frac  fraction of the 101x101 nodes observed (0-1]
  --space_mode  fixed  = the same sensor nodes at every time
                random = a different random subset at every observed time
  --time_frac   fraction of the snapshots (t >= 0) kept, at random -> irregular time gaps
  --max_gap     longest window (time units) used for shooting

  python experiments/sparse_inverse.py --param 1 --space_frac 0.1 --time_frac 0.5
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gs_solver import spline_matrix, divflux4, STAB

FK = {1: (0.037, 0.06), 2: (0.014, 0.045), 3: (0.018, 0.051)}
ap = argparse.ArgumentParser()
ap.add_argument("--param", type=int, default=1)
ap.add_argument("--synthetic", type=str, default=None, help="npz from make_synthetic.py instead of COMSOL")
ap.add_argument("--space_frac", type=float, default=1.0)
ap.add_argument("--space_mode", type=str, default="fixed", choices=["fixed", "random"])
ap.add_argument("--time_frac", type=float, default=1.0)
ap.add_argument("--max_gap", type=float, default=40.0)
ap.add_argument("--t_start", type=float, default=300.0)
ap.add_argument("--N", type=int, default=160)
ap.add_argument("--batch", type=int, default=8, help="windows per iteration")
ap.add_argument("--fit_batch", type=int, default=32768, help="samples per field-fit step")
ap.add_argument("--warm", type=int, default=3000, help="field-only warm-up steps")
ap.add_argument("--iters", type=int, default=3000, help="max joint iterations")
ap.add_argument("--lr_D", type=float, default=3e-3)
ap.add_argument("--lr_field", type=float, default=1e-3)
ap.add_argument("--w_shoot", type=float, default=1.0)
ap.add_argument("--w_cons", type=float, default=1.0, help="solver-vs-field consistency on the full grid")
ap.add_argument("--patience", type=int, default=400)
ap.add_argument("--n_fourier", type=int, default=128)
ap.add_argument("--sig_xy", type=float, default=4.0, help="Fourier std in space (cycles per unit)")
ap.add_argument("--sig_t", type=float, default=3.0, help="Fourier std in normalized time")
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--D_scale", type=float, default=1e-5)
ap.add_argument("--cfl", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", type=str, default=None)
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)
dev = torch.device(args.device)
T32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
src = "syn" if args.synthetic else "comsol"
out_dir = args.out or os.path.join(HERE, "results", f"sparse_{src}_p{args.param}_s{args.space_frac:g}{args.space_mode[0]}_t{args.time_frac:g}{args.tag}")
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------ data + sparse sampling
if args.synthetic:
    d = np.load(args.synthetic); F, k = float(d["F"]), float(d["k"])
else:
    d = np.load(os.path.join(HERE, "cache", f"data_{args.param}_grid.npz")); F, k = FK[args.param]
xs, ys, tdat, U, V = d["xs"], d["ys"], d["t"], d["U"], d["V"]
nx, ny, nt = len(xs), len(ys), len(tdat)
n_nodes = nx * ny
keep_t = np.sort(rng.choice(nt, max(2, int(round(args.time_frac * nt))), replace=False))
keep_t[0] = 0 if args.time_frac >= 1 else keep_t[0]
n_obs = max(1, int(round(args.space_frac * n_nodes)))
fixed_nodes = rng.choice(n_nodes, n_obs, replace=False)
obs_nodes = {int(i): (fixed_nodes if args.space_mode == "fixed" else rng.choice(n_nodes, n_obs, replace=False))
             for i in keep_t}
Uf, Vf = U.reshape(nt, -1), V.reshape(nt, -1)
Xn, Yn = np.meshgrid(xs, ys, indexing="ij"); Xn, Yn = Xn.ravel(), Yn.ravel()
# flat table of all observations (for the field fit)
rows = [(Xn[obs_nodes[i]], Yn[obs_nodes[i]], np.full(n_obs, tdat[i]), Uf[i, obs_nodes[i]], Vf[i, obs_nodes[i]]) for i in keep_t]
OBS = T32(np.concatenate([np.stack(r, 1) for r in rows], 0))          # (n_total, 5): x y t u v
sig_u, sig_v = float(OBS[:, 3].std()), float(OBS[:, 4].std())
# shooting windows: consecutive observed times with a gap <= max_gap, start >= t_start
pairs = [(int(a), int(b)) for a, b in zip(keep_t[:-1], keep_t[1:])
         if tdat[a] >= args.t_start and tdat[b] - tdat[a] <= args.max_gap + 1e-9]
assert pairs, "no shooting windows: increase --time_frac or --max_gap"
print(f"device {dev} | {src} case {args.param} F={F} k={k} | {len(keep_t)}/{nt} times, {n_obs}/{n_nodes} nodes "
      f"({args.space_mode}) -> {len(OBS)} samples ({100*len(OBS)/(nt*n_nodes):.2f}% of full) | {len(pairs)} windows | out {out_dir}", flush=True)

# ------------------------------------------------------------------ models
def norm(z, lo, hi): return 2.0 * (z - lo) / (hi - lo) - 1.0
class Field(nn.Module):
    """(x, y, t) -> (u, v), Fourier features + MLP."""
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        B = torch.randn(3, args.n_fourier, generator=g) * torch.tensor([args.sig_xy, args.sig_xy, args.sig_t]).view(-1, 1)
        self.register_buffer("B", B)
        w = args.width
        self.net = nn.Sequential(nn.Linear(2 * args.n_fourier, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                                 nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 2))
    def forward(self, x, y, t):
        X = torch.stack([norm(x, xs[0], xs[-1]), norm(y, ys[0], ys[-1]), norm(t, tdat[0], tdat[-1])], -1)
        p = 2 * math.pi * X @ self.B
        o = self.net(torch.cat([torch.sin(p), torch.cos(p)], -1))
        return 1.0 - sig_u * o[..., 0], sig_v * o[..., 1]          # u around 1, v around 0

D_SCALE = args.D_scale
class Du_net(nn.Module):           # same as the full-data model
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, X): return D_SCALE * nn.functional.softplus(self.network(X))
class Dv_net(nn.Module):
    def __init__(self): super().__init__(); self.Dv_raw = nn.Parameter(torch.tensor(1.0))
    def value(self): return D_SCALE * nn.functional.softplus(self.Dv_raw)
field, Du_nn, Dv_nn = Field().to(dev), Du_net().to(dev), Dv_net().to(dev)
n_par = lambda m: sum(p.numel() for p in m.parameters())
print(f"trainable parameters: field {n_par(field)}, Du_net {n_par(Du_nn)}, Dv 1", flush=True)

# ------------------------------------------------------------------ solver grid
N = args.N; hx, hy = (xs[-1] - xs[0]) / N, (ys[-1] - ys[0]) / N
xc = xs[0] + (np.arange(N) + 0.5) * hx; yc = ys[0] + (np.arange(N) + 0.5) * hy
Xc, Yc = torch.meshgrid(T32(xc), T32(yc), indexing="ij")
Px, Py = T32(spline_matrix(xc, xs)), T32(spline_matrix(yc, ys))     # solver grid -> data nodes
def grid_in(xv, yv):
    X, Y = torch.meshgrid(T32(xv), T32(yv), indexing="ij")
    return torch.stack([norm(X, xs[0], xs[-1]), norm(Y, ys[0], ys[-1])], -1)
Xface_x = grid_in(xs[0] + np.arange(N + 1) * hx, yc); Xface_y = grid_in(xc, ys[0] + np.arange(N + 1) * hy)
OBS_IDX = {i: torch.tensor(obs_nodes[i], device=dev) for i in keep_t}
Ud, Vd = T32(Uf), T32(Vf)

def rollout(u, v, Dfx, Dfy, Dv, n_sub, dt):
    for _ in range(n_sub):
        uv2 = u * v * v
        u, v = (u + dt * (divflux4(u, Dfx, Dfy, hx, hy) - uv2 + F * (1 - u)),
                v + dt * (Dv * divflux4(v, 1.0, 1.0, hx, hy) + uv2 - (F + k) * v))
    return u, v

def fit_loss():
    b = OBS[torch.randint(0, len(OBS), (args.fit_batch,), device=dev)]
    u, v = field(b[:, 0], b[:, 1], b[:, 2])
    return ((u - b[:, 3]) / sig_u).pow(2).mean() + ((v - b[:, 4]) / sig_v).pow(2).mean()

def shoot_loss():
    Dfx, Dfy, Dv = Du_nn(Xface_x)[..., 0], Du_nn(Xface_y)[..., 0], Dv_nn.value()
    lim = 2.0 / (STAB[4] * max(Dfx.max().item(), Dv.item()) * (1 / hx**2 + 1 / hy**2))
    tot = 0.0
    for w in rng.choice(len(pairs), args.batch, replace=len(pairs) < args.batch):
        a, b = pairs[w]; T = float(tdat[b] - tdat[a])
        n_sub = max(1, math.ceil(T / (args.cfl * lim))); dt = T / n_sub
        u0, v0 = field(Xc, Yc, torch.full_like(Xc, float(tdat[a])))
        u, v = torch.utils.checkpoint.checkpoint(rollout, u0[None], v0[None], Dfx, Dfy, Dv, n_sub, dt, use_reentrant=False)
        idx = OBS_IDX[b]
        pu, pv = (Px @ u[0] @ Py.T).reshape(-1)[idx], (Px @ v[0] @ Py.T).reshape(-1)[idx]
        tot = tot + ((pu - Ud[b, idx]) / sig_u).pow(2).mean() + ((pv - Vd[b, idx]) / sig_v).pow(2).mean()
        if args.w_cons > 0:   # between sensors: solver(field(t_a)) must agree with field(t_b) on the whole grid
            u1, v1 = field(Xc, Yc, torch.full_like(Xc, float(tdat[b])))
            cons = ((u[0] - u1) / sig_u).pow(2).mean() + ((v[0] - v1) / sig_v).pow(2).mean()
            tot = tot + args.w_cons * cons
    return tot / args.batch

# ------------------------------------------------------------------ monitoring (truth used ONLY for reporting)
Du_true = 2e-5 * (1 + np.sin(np.pi * Xn.reshape(nx, ny)) * np.sin(np.pi * Yn.reshape(nx, ny)))
Xdat = grid_in(xs, ys)
@torch.no_grad()
def report():
    Dp = Du_nn(Xdat)[..., 0].cpu().numpy()
    Lp = float(np.sqrt(np.mean((Dp - Du_true) ** 2) / np.var(Du_true)))
    ti = np.linspace(nt // 10, nt - 1, 6).astype(int)          # field error on ALL nodes (held-out incl.)
    e = []
    for i in ti:
        u, v = field(T32(Xn), T32(Yn), torch.full((n_nodes,), float(tdat[i]), device=dev))
        e.append(float(torch.sqrt(((u - Ud[i]) ** 2).mean()) / sig_u))
    return Lp, float(np.mean(e)), Dp

# ------------------------------------------------------------------ training
hist = {"phase": [], "it": [], "fit": [], "shoot": [], "Lpaper": [], "field_err": [], "Dv": []}
t0 = time.time()
opt_f = torch.optim.Adam(field.parameters(), lr=args.lr_field)
for it in range(1, args.warm + 1):                      # warm-up: field follows the sparse data
    L = fit_loss(); opt_f.zero_grad(); L.backward(); opt_f.step()
    if it % 500 == 0 or it == 1:
        _, fe, _ = report()
        print(f"warm {it:5d} | fit {L.item():.3e} | field rel-RMSE (all nodes) {fe:.3e} | {time.time()-t0:.0f}s", flush=True)

opt = torch.optim.Adam([{"params": field.parameters(), "lr": args.lr_field},
                        {"params": list(Du_nn.parameters()) + list(Dv_nn.parameters()), "lr": args.lr_D}])
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 4, min_lr=1e-6)
ema, best, since = None, float("inf"), 0
for it in range(1, args.iters + 1):
    Lf = fit_loss(); Ls = shoot_loss(); L = Lf + args.w_shoot * Ls
    opt.zero_grad(); L.backward(); opt.step()
    ema = L.item() if ema is None else 0.98 * ema + 0.02 * L.item(); plateau.step(ema)
    if ema < best * (1 - 1e-3): best, since = ema, 0
    else: since += 1
    if it % 25 == 0 or it == 1:
        Lp, fe, _ = report()
        for kk, vv in zip(hist, ("joint", it, Lf.item(), Ls.item(), Lp, fe, Dv_nn.value().item())): hist[kk].append(vv)
        print(f"it {it:5d} | fit {Lf.item():.3e} shoot {Ls.item():.3e} | Du L_paper {100*Lp:6.2f}% | Dv {Dv_nn.value().item():.4e} "
              f"| field err {fe:.3e} | lr_D {opt.param_groups[1]['lr']:.1e} | {time.time()-t0:.0f}s", flush=True)
    if since > args.patience and opt.param_groups[1]["lr"] <= args.lr_D / 100 * 1.01:
        print(f"converged (loss plateau) at it {it}"); break

Lp, fe, Dp = report()
res = dict(source=src, case=args.param, space_frac=args.space_frac, space_mode=args.space_mode, time_frac=args.time_frac,
           n_samples=int(len(OBS)), sample_pct=100 * len(OBS) / (nt * n_nodes), n_windows=len(pairs),
           params_field=n_par(field), params_Du=n_par(Du_nn), iters_run=it, time_s=time.time() - t0,
           Du_Lpaper_pct=100 * Lp, Dv=Dv_nn.value().item(), Dv_rel_err_pct=100 * (Dv_nn.value().item() / 1e-5 - 1),
           field_relRMSE_all_nodes=fe)
print(json.dumps(res, indent=2))
json.dump(dict(result=res, args=vars(args), hist=hist), open(os.path.join(out_dir, "result.json"), "w"), indent=1)
torch.save({"Du_nn": Du_nn.state_dict(), "Dv_nn": Dv_nn.state_dict(), "field": field.state_dict(), "args": vars(args)},
           os.path.join(out_dir, "model.pt"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Xm, Ym = Xn.reshape(nx, ny), Yn.reshape(nx, ny)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a_, Z, ttl in [(ax[0], Du_true, "Du true"), (ax[1], Dp, f"Du learned ({100*Lp:.2f}%)"), (ax[2], Dp - Du_true, "error")]:
        c = a_.contourf(Xm, Ym, Z, 50); plt.colorbar(c, ax=a_); a_.set_title(ttl); a_.set_aspect("equal")
    sel = obs_nodes[int(keep_t[-1])]; ax[0].plot(Xn[sel], Yn[sel], "k.", ms=1)
    ax[3].semilogy(hist["it"], np.array(hist["Lpaper"]) * 100); ax[3].set_xlabel("joint iteration"); ax[3].set_ylabel("Du L_paper [%]"); ax[3].grid(True, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Du.png"), dpi=80)
except Exception as e:
    print("plot failed:", e)
