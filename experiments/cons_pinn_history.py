"""
Consistency PINN (backward-Euler) with a recorded L_normalized history.

Pipeline (same networks and settings as models/vannila_pinn/pretrain.py and models/cons_pinn/stage_b.py):
  1. pre-training : fit u_nn and v_nn (5x64 tanh) to 10,000 random nodes x 301 snapshots (data loss only)
  2. stage B      : train u_nn, v_nn, Du_nn, Dv jointly with  data loss + backward-Euler consistency loss,
                    starting from a FRESH Du/Dv (no stage A), 15,000 steps, no early stop.
Every --log_every steps it records L_normalized of Du on the full 101x101 data grid (same metric as
diffsolver_inverse.py) and on the 5,000-node monitor set used by stage_b.py, plus losses and Dv.

Outputs (experiments/results/cons_pinn_p{param}{tag}/): history.json (rewritten during training), Du.png,
model.pt.  Log: pass stdout to a file.
"""
import os, sys, json, math, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--param", type=int, default=1)
ap.add_argument("--dcase", type=int, default=1, help="data/param{p}/D{dcase}/ (D1 = current Du field)")
ap.add_argument("--n_data", type=int, default=10000)
ap.add_argument("--pre_epochs", type=int, default=500)
ap.add_argument("--pretrained", type=str, default=None, help="skip pre-training and load u_nn/v_nn from this .pt")
ap.add_argument("--steps", type=int, default=15000)
ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()

FK = {1: (0.037, 0.060), 2: (0.014, 0.045), 3: (0.018, 0.051)}
F, k = FK[args.param]
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = os.path.join(HERE, "results", f"cons_pinn_p{args.param}{args.tag}"); os.makedirs(out, exist_ok=True)
print(f"device {dev} | case {args.param} F={F} k={k} | out {out}", flush=True)

x_min, x_max, y_min, y_max, t_min, t_max = 0.0, 1.0, 0.0, 1.0, 0.0, 3000.0
def normalize(x, a, b): return 2.0 * (x - a) / (b - a) - 1.0

# ------------------------------------------------------------------ data (as in the original scripts)
src = os.path.join(ROOT, "data", f"param{args.param}", f"D{args.dcase}", f"data_{args.param}.txt")
times = np.arange(0, 3001, 10)
cols = ["x", "y"] + [c for t in times for c in (f"u_{t}", f"v_{t}")]
df = pd.read_csv(src, sep=r"\s+", comment="%", header=None, names=cols)
print("data", src, df.shape, flush=True)
u_s = df[["x", "y"] + [f"u_{t}" for t in times]].sample(n=args.n_data, random_state=42)
v_s = df[["x", "y"] + [f"v_{t}" for t in times]].sample(n=args.n_data, random_state=42)
def flat(s, p):
    X = np.column_stack([np.repeat(s["x"].values, len(times)), np.repeat(s["y"].values, len(times)),
                         np.tile(times, len(s))])
    return (torch.tensor(X, dtype=torch.float32, device=dev),
            torch.tensor(s[[f"{p}_{t}" for t in times]].values.reshape(-1, 1), dtype=torch.float32, device=dev))
X_u, U_true = flat(u_s, "u"); X_v, V_true = flat(v_s, "v")
sigma_u, sigma_v = U_true.std().item(), V_true.std().item()
def norm_in(X):
    return torch.cat([normalize(X[:, 0:1], x_min, x_max), normalize(X[:, 1:2], y_min, y_max),
                      normalize(X[:, 2:3], t_min, t_max)], dim=1)

# ------------------------------------------------------------------ networks (identical architectures)
def mlp(i, w, d, o): return nn.Sequential(*([nn.Linear(i, w), nn.Tanh()] + [m for _ in range(d - 1) for m in (nn.Linear(w, w), nn.Tanh())] + [nn.Linear(w, o)]))
class UV(nn.Module):
    def __init__(s): super().__init__(); s.network = mlp(3, 64, 5, 1)
    def forward(s, X): return s.network(X)
class Du_net(nn.Module):
    def __init__(s): super().__init__(); s.network = mlp(2, 32, 4, 1)
    def forward(s, X): return 1e-5 * torch.nn.functional.softplus(s.network(X))
class Dv_net(nn.Module):
    def __init__(s): super().__init__(); s.Dv_raw = nn.Parameter(torch.tensor(1.0))
    def value(s): return 1e-5 * torch.nn.functional.softplus(s.Dv_raw)
u_nn, v_nn, Du_nn, Dv_nn = UV().to(dev), UV().to(dev), Du_net().to(dev), Dv_net().to(dev)

# ------------------------------------------------------------------ 1. pre-training of u, v
t0 = time.time()
if args.pretrained:
    c = torch.load(args.pretrained, map_location=dev); u_nn.load_state_dict(c["u_nn"]); v_nn.load_state_dict(c["v_nn"])
    print("loaded pre-trained u, v from", args.pretrained, flush=True)
else:
    bs = 16384
    for name, net, Y in (("u", u_nn, U_true), ("v", v_nn, V_true)):
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pre_epochs, eta_min=1e-5)
        for ep in range(args.pre_epochs):
            perm = torch.randperm(X_u.shape[0], device=dev); tot = 0.0; nb = 0
            for i in range(0, X_u.shape[0], bs):
                idx = perm[i:i + bs]
                loss = torch.mean((net(norm_in(X_u[idx])) - Y[idx]) ** 2)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach(); nb += 1
            sch.step()
            if (ep + 1) % 25 == 0:
                print(f"pretrain {name} epoch {ep+1}/{args.pre_epochs} | MSE {tot.item()/nb:.3e} (var {Y.var().item():.2e}) "
                      f"| {time.time()-t0:.0f}s", flush=True)
    torch.save({"u_nn": u_nn.state_dict(), "v_nn": v_nn.state_dict()}, os.path.join(out, "pretrained_uv.pt"))
t_pre = time.time() - t0

# ------------------------------------------------------------------ Du error monitors
xs = df["x"].values; ys = df["y"].values
Dtrue_all = 2e-5 * (1 + np.sin(np.pi * xs) * np.sin(np.pi * ys))
XY_all = torch.tensor(np.column_stack([normalize(xs, x_min, x_max), normalize(ys, y_min, y_max)]), dtype=torch.float32, device=dev)
mon = np.random.RandomState(0).choice(len(df), 5000, replace=False)
@torch.no_grad()
def du_errors():
    D = Du_nn(XY_all).cpu().numpy().ravel()
    Lg = np.sqrt(np.mean((D - Dtrue_all) ** 2) / np.var(Dtrue_all))
    Lm = np.sqrt(np.mean((D[mon] - Dtrue_all[mon]) ** 2) / np.var(Dtrue_all[mon]))
    return float(Lg), float(Lm), D

# ------------------------------------------------------------------ 2. stage B (from a fresh Du, Dv)
dt_be, t_lo_be, N_be, data_bs, lam_data, lam_pde, ramp = 20.0, 500.0, 20000, 8192, 1.0, 1.0, 100
def div_flux(D, f, x, y):
    fx, fy = torch.autograd.grad(f, [x, y], torch.ones_like(f), create_graph=True)
    return (torch.autograd.grad(D * fx, x, torch.ones_like(fx), create_graph=True)[0] +
            torch.autograd.grad(D * fy, y, torch.ones_like(fy), create_graph=True)[0])
def be_loss():
    x = (torch.rand(N_be, 1, device=dev) * (x_max - x_min) + x_min).requires_grad_(True)
    y = (torch.rand(N_be, 1, device=dev) * (y_max - y_min) + y_min).requires_grad_(True)
    t0_ = torch.rand(N_be, 1, device=dev) * (t_max - dt_be - t_lo_be) + t_lo_be; t1_ = t0_ + dt_be
    inp = lambda t: torch.cat([normalize(x, x_min, x_max), normalize(y, y_min, y_max), normalize(t, t_min, t_max)], 1)
    X0, X1 = inp(t0_), inp(t1_)
    u0, v0, u1, v1 = u_nn(X0), v_nn(X0), u_nn(X1), v_nn(X1)
    Du = Du_nn(torch.cat([normalize(x, x_min, x_max), normalize(y, y_min, y_max)], 1)); Dv = Dv_nn.value()
    Ru = div_flux(Du, u1, x, y) - u1 * v1 ** 2 + F * (1 - u1)
    Rv = div_flux(Dv * torch.ones_like(Du), v1, x, y) + u1 * v1 ** 2 - (F + k) * v1
    return torch.mean(((u1 - dt_be * Ru - u0) / sigma_u) ** 2) + torch.mean(((v1 - dt_be * Rv - v0) / sigma_v) ** 2)
def data_loss(idx):
    return (torch.mean(((u_nn(norm_in(X_u[idx])) - U_true[idx]) / sigma_u) ** 2) +
            torch.mean(((v_nn(norm_in(X_v[idx])) - V_true[idx]) / sigma_v) ** 2))

uv_p = list(u_nn.parameters()) + list(v_nn.parameters()); D_p = list(Du_nn.parameters()) + list(Dv_nn.parameters())
opt = torch.optim.Adam([{"params": uv_p, "lr": 5e-4}, {"params": D_p, "lr": 5e-4}])
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
hist = {"it": [], "L_norm_grid": [], "L_norm_monitor": [], "data": [], "pde": [], "Dv": []}
def dump(extra=None):
    json.dump(dict(args=vars(args), F=F, k=k, pretrain_s=t_pre, hist=hist, result=extra or {}),
              open(os.path.join(out, "history.json"), "w"), indent=1)

Lg, Lm, _ = du_errors(); hist["it"].append(0); hist["L_norm_grid"].append(Lg); hist["L_norm_monitor"].append(Lm)
hist["data"].append(None); hist["pde"].append(None); hist["Dv"].append(Dv_nn.value().item())
t1 = time.time()
for step in range(args.steps):
    w = lam_pde * min(1.0, (step + 1) / ramp)
    Ld = data_loss(torch.randint(0, X_u.shape[0], (data_bs,), device=dev)); Lp = be_loss()
    L = lam_data * Ld + w * Lp
    opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(uv_p + D_p, 1.0); opt.step(); sch.step()
    if (step + 1) % args.log_every == 0:
        Lg, Lm, _ = du_errors(); Dv = Dv_nn.value().item()
        for kk, vv in zip(hist, (step + 1, Lg, Lm, Ld.item(), Lp.item(), Dv)): hist[kk].append(vv)
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{args.steps} | data {Ld.item():.3e} | pde {Lp.item():.3e} | Du L_norm grid {100*Lg:.2f}% "
                  f"monitor {100*Lm:.2f}% | Dv {Dv:.4e} | {time.time()-t1:.0f}s", flush=True)
        if (step + 1) % 1000 == 0: dump()

Lg, Lm, D = du_errors(); Dv = Dv_nn.value().item()
res = dict(Du_Lnorm_grid_pct=100 * Lg, Du_Lnorm_monitor_pct=100 * Lm, best_Lnorm_grid_pct=100 * min(hist["L_norm_grid"]),
           Dv=Dv, Dv_rel_err_pct=100 * (Dv / 1e-5 - 1), stageB_s=time.time() - t1, pretrain_s=t_pre)
dump(res); print(json.dumps(res, indent=1), flush=True)
torch.save({"u_nn": u_nn.state_dict(), "v_nn": v_nn.state_dict(), "Du_nn": Du_nn.state_dict(), "Dv_nn": Dv_nn.state_dict(),
            "args": vars(args)}, os.path.join(out, "model.pt"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xu, yu = np.unique(np.round(xs, 12)), np.unique(np.round(ys, 12))
    ix, iy = np.searchsorted(xu, np.round(xs, 12)), np.searchsorted(yu, np.round(ys, 12))
    def G(a):
        Z = np.zeros((len(yu), len(xu))); Z[iy, ix] = a; return Z
    Xm, Ym = np.meshgrid(xu, yu)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a, Z, ttl in [(ax[0], G(Dtrue_all), "Du true"), (ax[1], G(D), f"Du consistency PINN (L_norm {100*Lg:.2f}%)"),
                      (ax[2], G(D - Dtrue_all), "error")]:
        c = a.contourf(Xm, Ym, Z, 50); plt.colorbar(c, ax=a); a.set_title(ttl); a.set_aspect("equal")
    ax[3].semilogy(hist["it"], np.array(hist["L_norm_grid"]) * 100); ax[3].set_xlabel("step")
    ax[3].set_ylabel("Du L_norm [%]"); ax[3].grid(True, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(out, "Du.png"), dpi=80)
except Exception as e:
    print("plot failed:", e)
