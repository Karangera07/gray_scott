"""
Control-experiment data: a Gray-Scott run with KNOWN Du(x,y), Dv, made with our own fine-grid
4th-order solver, sampled exactly like the COMSOL export (same graded 101x101 nodes, every 10 t).

Also stores the exact fields on the inverse solver's grid (--Nsolver), so the inverse model can be
started either from the spline of the data nodes (as with COMSOL) or from the exact state.

  python experiments/make_synthetic.py --param 1            # -> experiments/cache/synthetic_p1.npz
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gs_solver import spline_matrix, divflux4, STAB

FK = {1: (0.037, 0.06), 2: (0.014, 0.045), 3: (0.018, 0.051)}
ap = argparse.ArgumentParser()
ap.add_argument("--param", type=int, default=1)
ap.add_argument("--Ngen", type=int, default=320, help="generator grid")
ap.add_argument("--Nsolver", type=int, default=160, help="grid of the inverse solver (for exact start states)")
ap.add_argument("--cfl", type=float, default=0.5)
ap.add_argument("--T", type=float, default=3000.0)
ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", type=str, default=None)
args = ap.parse_args()
F, k = FK[args.param]
dev = torch.device(args.device); dt64 = torch.float64

g = np.load(os.path.join(HERE, "cache", f"data_{args.param}_grid.npz"))   # only for the node layout
xs, ys = g["xs"], g["ys"]
N = args.Ngen; h = 1.0 / N; xc = (np.arange(N) + 0.5) * h
X, Y = np.meshgrid(xc, xc, indexing="ij")

def Du(x, y): return 2e-5 * (1 + np.sin(np.pi * x) * np.sin(np.pi * y))
Dv = 1e-5
xf = np.arange(N + 1) * h
Dfx = torch.tensor(Du(xf[:, None], xc[None]), dtype=dt64, device=dev)
Dfy = torch.tensor(Du(xc[:, None], xf[None]), dtype=dt64, device=dev)

# initial condition as in COMSOL: 0.1-wide centre square, u=0.5 v=0.25 inside, u=1 v=0 outside
w = 0.004
box = 0.25 * (1 + np.tanh((0.05 - np.abs(X - 0.5)) / w)) * (1 + np.tanh((0.05 - np.abs(Y - 0.5)) / w))
u = torch.tensor(1 - 0.5 * box, dtype=dt64, device=dev)[None]
v = torch.tensor(0.25 * box, dtype=dt64, device=dev)[None]

dt_out = 10.0
lim = 2.0 / (STAB[4] * 4e-5 * 2 / h**2)
n_sub = int(np.ceil(dt_out / (args.cfl * lim))); dt = dt_out / n_sub
T64 = lambda a: torch.tensor(a, dtype=dt64, device=dev)
Px, Py = T64(spline_matrix(xc, xs)), T64(spline_matrix(xc, ys))             # gen grid -> data nodes
Ns = args.Nsolver; xsol = (np.arange(Ns) + 0.5) / Ns
Sx = T64(spline_matrix(xc, xsol))                                           # gen grid -> solver grid
nt = int(round(args.T / dt_out)) + 1
U = np.zeros((nt, len(xs), len(ys))); V = np.zeros_like(U)
Uf = np.zeros((nt, Ns, Ns), np.float32); Vf = np.zeros_like(Uf)
print(f"generating case {args.param} on {N}^2, dt={dt:.4f} ({n_sub} steps per output)", flush=True)
t0 = time.time()
for it in range(nt):
    if it > 0:
        for _ in range(n_sub):
            uv2 = u * v * v
            u, v = (u + dt * (divflux4(u, Dfx, Dfy, h, h) - uv2 + F * (1 - u)),
                    v + dt * (Dv * divflux4(v, 1.0, 1.0, h, h) + uv2 - (F + k) * v))
    U[it] = (Px @ u[0] @ Py.T).cpu().numpy(); V[it] = (Px @ v[0] @ Py.T).cpu().numpy()
    Uf[it] = (Sx @ u[0] @ Sx.T).cpu().numpy(); Vf[it] = (Sx @ v[0] @ Sx.T).cpu().numpy()
    if it % 50 == 0:
        print(f"t={it*dt_out:6.0f}  u[{U[it].min():.3f},{U[it].max():.3f}] v[{V[it].min():.3f},{V[it].max():.3f}]  {time.time()-t0:.0f}s", flush=True)
out = args.out or os.path.join(HERE, "cache", f"synthetic_p{args.param}.npz")
np.savez(out, xs=xs, ys=ys, t=np.arange(nt) * dt_out, U=U, V=V, Ufine=Uf, Vfine=Vf,
         F=F, k=k, Ngen=N, Nsolver=Ns)
print("saved", out)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2)); Xd, Yd = np.meshgrid(xs, ys, indexing="ij")
    for a, ti in zip(ax, (20, 50, 100, 200, 300)):
        c = a.contourf(Xd, Yd, V[ti], 40); a.set_title(f"synthetic v, t={ti*10}"); a.set_aspect("equal")
    plt.tight_layout(); plt.savefig(out.replace(".npz", ".png"), dpi=60)
except Exception as e:
    print("plot failed:", e)
