"""Shared access to the COMSOL datasets in data/<bc>/D<d>/fk<p>/ (see data/README.md).

    from gs_data import add_data_args, load, Du_true, case_tag
    add_data_args(ap)                 # adds --param (1-4), --bc (noflux|dirichlet), --dcase (1,3,4,6)
    d = load(args.param, args.bc, args.dcase)   # dict: xs, ys, t, U, V (float64, [time, x, y]), F, k, path
    Du_true(X, Y, args.dcase)         # true Du for REPORTING only
    case_tag(args.bc, args.dcase)     # "" for noflux/D1 (old result names unchanged), else "_dirichlet_D4" etc.

Reads the compact .npz next to each COMSOL .txt (data/make_npz.py); falls back to parsing the .txt.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

FK = {1: (0.037, 0.060), 2: (0.014, 0.045), 3: (0.018, 0.051), 4: (0.0367, 0.0649)}
BCS = ("noflux", "dirichlet")
D_U, D_V = 2e-5, 1e-5
_DU = {
    1: lambda x, y: D_U * (1 + np.sin(np.pi * x) * np.sin(np.pi * y)),
    3: lambda x, y: D_U * (1 + np.exp(-((x - 0.3) ** 2 + (y - 0.65) ** 2) / (2 * 0.15 ** 2))),
    4: lambda x, y: D_U * (1.5 + 0.5 * np.tanh((x - 0.5) / 0.05)),
    6: lambda x, y: D_U * (1.5 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)),
}
DCASES = tuple(_DU)


def add_data_args(ap, param=True):
    if param:
        ap.add_argument("--param", type=int, default=1, choices=sorted(FK), help="(F,k) pair: data/<bc>/D<d>/fk<param>")
    ap.add_argument("--bc", type=str, default="noflux", choices=BCS, help="boundary condition of the data and solver")
    ap.add_argument("--dcase", type=int, default=1, choices=DCASES, help="true Du field D<dcase> (data/README.md)")


def data_path(param, bc="noflux", dcase=1, ext=".npz"):
    name = f"data_{bc}_D{dcase}_fk{param}"
    return os.path.join(DATA, bc, f"D{dcase}", f"fk{param}", name + ext)


def Du_true(x, y, dcase=1):
    return _DU[int(dcase)](np.asarray(x, float), np.asarray(y, float)) * np.ones(np.broadcast(x, y).shape)


def case_tag(bc="noflux", dcase=1):
    return "" if (bc == "noflux" and int(dcase) == 1) else f"_{bc}_D{int(dcase)}"


def _parse_txt(path):
    import pandas as pd
    a = pd.read_csv(path, comment="%", sep=r"\s+", header=None).to_numpy()
    xr, yr = np.round(a[:, 0], 12), np.round(a[:, 1], 12)
    xs, ys = np.unique(xr), np.unique(yr)
    assert len(xs) * len(ys) == len(a), "data is not on a tensor-product grid"
    ix, iy = np.searchsorted(xs, xr), np.searchsorted(ys, yr)
    f = a[:, 2:]; nt = f.shape[1] // 2
    U = np.zeros((nt, len(xs), len(ys))); V = np.zeros_like(U)
    U[:, ix, iy] = f[:, 0::2].T; V[:, ix, iy] = f[:, 1::2].T
    return xs, ys, np.arange(nt) * 10.0, U, V


def load(param=1, bc="noflux", dcase=1, path=None):
    """Load one dataset. `path` (npz or COMSOL txt) overrides the (param, bc, dcase) lookup."""
    path = path or data_path(param, bc, dcase)
    if not os.path.exists(path) and path.endswith(".npz") and os.path.exists(path[:-4] + ".txt"):
        path = path[:-4] + ".txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (git pull / git lfs pull, or run data/make_npz.py)")
    if path.endswith(".npz"):
        g = np.load(path)
        xs, ys, t = g["xs"], g["ys"], g["t"]
        U, V = g["U"].astype(np.float64), g["V"].astype(np.float64)
    else:
        xs, ys, t, U, V = _parse_txt(path)
    F, k = FK[param]
    return dict(xs=xs, ys=ys, t=t, U=U, V=V, F=F, k=k, path=path)
