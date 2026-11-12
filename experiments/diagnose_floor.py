"""
Diagnose WHERE the ~15-19% Du L_paper floor comes from (no neural networks involved).

  1. strong form : spline derivatives of the data + linear least squares for Du
  2. weak form   : Q1 finite elements on the data mesh (consistent / lumped mass)
  3. shooting    : fine-grid solver started from data snapshots, misfit scan over b
                   in Du = a + b*sin(pi x)sin(pi y)

Methods 1-2 need the curvature of the data and stall at ~15-25% even with the exact
functional form of Du. Method 3 only needs data *values* and recovers b to <1-3%.

Run:  python experiments/diagnose_floor.py --param 1
(needs experiments/cache/data_{p}_grid.npz, created by diffsolver_inverse.py, or --data)
"""
import argparse, os
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline, RectBivariateSpline as RBS

HERE = os.path.dirname(os.path.abspath(__file__))
FK = {1: (0.037, 0.06), 2: (0.014, 0.045), 3: (0.018, 0.051)}
ap = argparse.ArgumentParser()
ap.add_argument("--param", type=int, default=1)
ap.add_argument("--cache", type=str, default=os.path.join(HERE, "cache"))
ap.add_argument("--N", type=int, default=160)
args = ap.parse_args()
p = args.param; F, k = FK[p]
g = np.load(os.path.join(args.cache, f"data_{p}_grid.npz"))
xs, ys, U, V = g["xs"], g["ys"], g["U"], g["V"]
X, Y = np.meshgrid(xs, ys, indexing="ij"); s = np.sin(np.pi * X) * np.sin(np.pi * Y)
Dut = 2e-5 * (1 + s)
Lp = lambda D: 100 * np.sqrt(np.mean((D - Dut) ** 2) / np.var(Dut))
t = np.arange(U.shape[0]) * 10.0; tm = slice(50, 300)

# ---------------- 1. strong form
Ut = CubicSpline(t, U, axis=0)(t, 1)
cx, cy = CubicSpline(xs, U, axis=1), CubicSpline(ys, U, axis=2)
Ux, Uy, LU = cx(xs, 1), cy(ys, 1), cx(xs, 2) + cy(ys, 2)
sx, sy = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y), np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y)
lhs = Ut + U * V**2 - F * (1 - U)
A = np.stack([LU[tm].ravel(), (s * LU + sx * Ux + sy * Uy)[tm].ravel()], 1)
c, *_ = np.linalg.lstsq(A, lhs[tm].ravel(), rcond=None)
print(f"[strong form] a={c[0]:.4e} b={c[1]:.4e}  L_paper {Lp(c[0] + c[1] * s):.2f}%")
h = np.maximum.outer(np.gradient(xs), np.gradient(ys))
res = np.abs(lhs - (Dut * LU + 2e-5 * (sx * Ux + sy * Uy)))[tm].mean(0)
for lo, hi in [(0, .004), (.004, .008), (.008, .013), (.013, .02)]:
    m = (h >= lo) & (h < hi); print(f"   residual with TRUE D, mesh h in [{lo},{hi}): {res[m].mean():.2e}")

# ---------------- 2. weak form (Q1, 2x2 Gauss)
def assemble(funcs):
    nx, ny = len(xs), len(ys); idx = np.arange(nx * ny).reshape(nx, ny)
    ex, ey = [a.ravel() for a in np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")]
    hx, hy = xs[ex + 1] - xs[ex], ys[ey + 1] - ys[ey]
    nodes = np.stack([idx[ex, ey], idx[ex + 1, ey], idx[ex + 1, ey + 1], idx[ex, ey + 1]], 1)
    sgx, sgy = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    Me = np.zeros((len(ex), 4, 4)); Ke = [np.zeros_like(Me) for _ in funcs]
    for a in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        for b in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            Nn = 0.25 * (1 + sgx * a) * (1 + sgy * b); w = hx * hy / 4
            dNx = (0.25 * sgx * (1 + sgy * b))[None] * (2 / hx)[:, None]
            dNy = (0.25 * sgy * (1 + sgx * a))[None] * (2 / hy)[:, None]
            xq, yq = xs[ex] + (a + 1) / 2 * hx, ys[ey] + (b + 1) / 2 * hy
            Me += w[:, None, None] * np.outer(Nn, Nn)[None]
            G = dNx[:, :, None] * dNx[:, None, :] + dNy[:, :, None] * dNy[:, None, :]
            for j, f in enumerate(funcs): Ke[j] += (w * f(xq, yq))[:, None, None] * G
    r, cc, n = np.repeat(nodes, 4, 1).ravel(), np.tile(nodes, (1, 4)).ravel(), nx * ny
    return sp.csr_matrix((Me.ravel(), (r, cc)), (n, n)), [sp.csr_matrix((K.ravel(), (r, cc)), (n, n)) for K in Ke]
M, Ks = assemble([lambda x, y: np.ones_like(x), lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)])
T = U.reshape(len(t), -1); W = V.reshape(len(t), -1); ts = np.arange(50, 300)
Utc = (T[ts + 1] - T[ts - 1]) / 20; Uc, Wc = T[ts], W[ts]
for name, MM in [("consistent", M), ("lumped", sp.diags(np.asarray(M.sum(1)).ravel()))]:
    b_ = -(MM @ (Utc + Uc * Wc**2 - F * (1 - Uc)).T).T.ravel()
    A = np.stack([(Kj @ Uc.T).T.ravel() for Kj in Ks], 1); c, *_ = np.linalg.lstsq(A, b_, rcond=None)
    print(f"[weak form, {name} mass] a={c[0]:.4e} b={c[1]:.4e}  L_paper {Lp(c[0] + c[1] * s):.2f}%")

# ---------------- 3. shooting misfit scan
N = args.N; hh = 1.0 / N; xc = (np.arange(N) + 0.5) * hh; Xc, Yc = np.meshgrid(xc, xc, indexing="ij")
def faces(a, b):
    return (a + b * np.sin(np.pi * (Xc[:-1] + hh / 2)) * np.sin(np.pi * Yc[:-1]),
            a + b * np.sin(np.pi * Xc[:, :-1]) * np.sin(np.pi * (Yc[:, :-1] + hh / 2)))
def divf(u, fx_, fy_):
    fx = fx_ * (u[1:] - u[:-1]) / hh; fy = fy_ * (u[:, 1:] - u[:, :-1]) / hh
    o = np.zeros_like(u); o[:-1] += fx; o[1:] -= fx; o[:, :-1] += fy; o[:, 1:] -= fy; return o / hh
dt = min(0.2, 0.2 * hh * hh / 4e-5)
def misfit(b, dv=1e-5, H=5):
    Du, Dv = faces(2e-5, b), faces(dv, 0.0); e = 0
    for s0 in (60, 100, 150, 200, 250):
        u, v = RBS(xs, ys, U[s0])(xc, xc), RBS(xs, ys, V[s0])(xc, xc)
        for _ in range(int(round(H * 10 / dt))):
            uv2 = u * v * v
            u, v = u + dt * (divf(u, *Du) - uv2 + F * (1 - u)), v + dt * (divf(v, *Dv) + uv2 - (F + k) * v)
        e += np.mean((RBS(xc, xc, u)(xs, ys) - U[s0 + H])**2) / U.var() + np.mean((RBS(xc, xc, v)(xs, ys) - V[s0 + H])**2) / V.var()
    return e / 5
f = [misfit(b) for b in (1.9e-5, 2.0e-5, 2.1e-5)]
bh = 2e-5 + 1e-6 * (f[0] - f[2]) / (2 * (f[0] - 2 * f[1] + f[2]))
print(f"[shooting N={N}] misfit(b=1.9,2.0,2.1 e-5) = {f[0]:.2e} {f[1]:.2e} {f[2]:.2e}  ->  b_hat={bh:.4e}  L_paper {Lp(2e-5 + bh * s):.2f}%")
