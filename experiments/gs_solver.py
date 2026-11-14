"""Shared numerics for the Gray-Scott experiments (pure numpy + torch, no scipy needed)."""
import numpy as np
import torch


def spline_matrix(src, dst):
    """Matrix S with S @ f(src) = clamped cubic spline (f'=0 at both ends, i.e. no-flux walls)
    evaluated at dst."""
    n, h = len(src), np.diff(src)
    A = np.zeros((n, n)); B = np.zeros((n, n))         # A @ M = B @ y, M = second derivatives
    A[0, 0], A[0, 1] = h[0] / 3, h[0] / 6; B[0, 0], B[0, 1] = -1 / h[0], 1 / h[0]
    A[-1, -1], A[-1, -2] = h[-1] / 3, h[-1] / 6; B[-1, -1], B[-1, -2] = -1 / h[-1], 1 / h[-1]
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1] / 6, (h[i - 1] + h[i]) / 3, h[i] / 6
        B[i, i - 1], B[i, i], B[i, i + 1] = 1 / h[i - 1], -1 / h[i - 1] - 1 / h[i], 1 / h[i]
    Mop = np.linalg.solve(A, B)                          # y -> M
    j = np.clip(np.searchsorted(src, dst) - 1, 0, n - 2)
    hj = h[j]; a = (src[j + 1] - dst) / hj; b = 1 - a
    S = np.zeros((len(dst), n)); r = np.arange(len(dst))
    S[r, j] += a; S[r, j + 1] += b
    S += ((a**3 - a) * hj**2 / 6)[:, None] * Mop[j] + ((b**3 - b) * hj**2 / 6)[:, None] * Mop[j + 1]
    return S                                             # (len(dst), len(src))


def divflux2(u, Dfx, Dfy, hx, hy):
    """2nd-order finite-volume div(D grad u), zero-flux walls. u: (B,N,N);
    Dfx: interior x-faces (N-1,N), Dfy: (N,N-1), or scalars."""
    fx = Dfx * (u[:, 1:] - u[:, :-1]) / hx
    fy = Dfy * (u[:, :, 1:] - u[:, :, :-1]) / hy
    zx = torch.zeros_like(u[:, :1]); zy = torch.zeros_like(u[:, :, :1])
    return (torch.cat([fx, zx], 1) - torch.cat([zx, fx], 1)) / hx + \
           (torch.cat([fy, zy], 2) - torch.cat([zy, fy], 2)) / hy


def div4_1d(u, D, h):
    """d/dx (D du/dx) along dim 1, 4th order. u: (B,N,M) cell centres; D: faces 0..N, shape (N+1,M)
    or scalar. Interior face gradients (1,-27,27,-1)/24h; first faces off each wall one-sided
    (-23,21,3,-1)/24h; wall flux = 0 (no-flux); one ghost flux beyond each wall from the cubic
    extrapolation f_-1 = -6 f_1 + 4 f_2 - f_3 (uses f_0 = 0); then the same 4-point stencil on fluxes."""
    gi = (u[:, :-3] - 27 * u[:, 1:-2] + 27 * u[:, 2:-1] - u[:, 3:]) / (24 * h)            # faces 2..N-2
    g1 = (-23 * u[:, 0:1] + 21 * u[:, 1:2] + 3 * u[:, 2:3] - u[:, 3:4]) / (24 * h)        # face 1
    gN = (23 * u[:, -1:] - 21 * u[:, -2:-1] - 3 * u[:, -3:-2] + u[:, -4:-3]) / (24 * h)   # face N-1
    z = torch.zeros_like(g1)
    f = D * torch.cat([z, g1, gi, gN, z], 1)                                              # faces 0..N
    fm = -6 * f[:, 1:2] + 4 * f[:, 2:3] - f[:, 3:4]
    fp = -6 * f[:, -2:-1] + 4 * f[:, -3:-2] - f[:, -4:-3]
    f = torch.cat([fm, f, fp], 1)                                                         # faces -1..N+1
    return (f[:, :-3] - 27 * f[:, 1:-2] + 27 * f[:, 2:-1] - f[:, 3:]) / (24 * h)


def divflux4(u, Dfx, Dfy, hx, hy):
    """4th-order conservative FD div(D grad u), zero-flux walls. u: (B,N,N);
    Dfx: all x-faces (N+1,N), Dfy: (N,N+1), or scalars."""
    Dy = Dfy.T if torch.is_tensor(Dfy) else Dfy
    return div4_1d(u, Dfx, hx) + div4_1d(u.transpose(1, 2), Dy, hy).transpose(1, 2)


STAB = {2: 4.0, 4: 49.0 / 9.0}     # max eigenvalue of the -d2/dx2 stencil, times h^2
