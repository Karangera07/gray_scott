# Importing Libraries

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import math

print("Libraries imported successfully.")
fine_tuning = False  # Set to True if you want to fine-tune from Stage A

# Step 1: Define problem

L = 1.0  # length of the domain
x_min, x_max = 0.0, L
y_min, y_max = 0.0, L
t_min, t_max = 0.0, 3000.0

# dimensionality
input_dim = 3   # (x, y, t)
output_dim = 1  # C only

param = 1
n_data = 10000
N_collocation = 20000

F1 = 0.037
k1 = 0.06

F2 = 0.014
k2 = 0.045

F3 = 0.018
k3 = 0.051

if param == 1:
    F = F1
    k = k1
elif param == 2:
    F = F2
    k = k2
elif param == 3:
    F = F3
    k = k3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

# Sampling collocation points

def sample_collocation_points(N_col, x_min, x_max, y_min, y_max, t_min, t_max,device="cuda"):
    x = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(N_col, 1, device=device) * (y_max - y_min) + y_min
    t = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min
    return torch.cat([x, y, t], dim=1)


# Defining gradients

def gradients(C, x):
    return torch.autograd.grad(
        C,
        x,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]

# Defining NN Architecture

D_SCALE = 1e-5   # characteristic magnitude of the diffusion coefficients

class Du_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, X):
        # network output is O(1); scale it and keep it positive
        return D_SCALE * torch.nn.functional.softplus(self.network(X))


class Dv_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Dv_raw = nn.Parameter(torch.tensor(1.0))   # O(1) parameter -> Dv about 1.3e-5 at init

    def value(self):
        return D_SCALE * torch.nn.functional.softplus(self.Dv_raw)

    def forward(self, X):
        return self.value() * torch.ones_like(X[:, 0:1])   # shape (N,1)


# Defining PDE Consistency Loss

def normalize(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def make_D_fns(Du_nn, Dv_nn):
    def Du_fn(x, y):
        return Du_nn(torch.cat([normalize(x, x_min, x_max), normalize(y, y_min, y_max)], dim=1))
    def Dv_fn(x, y):
        return Dv_nn(torch.cat([normalize(x, x_min, x_max), normalize(y, y_min, y_max)], dim=1))
    return Du_fn, Dv_fn

def div_flux(D, f, x, y):
    """div(D grad f) in physical coordinates."""
    f_x, f_y = torch.autograd.grad(f, [x, y], torch.ones_like(f), create_graph=True)
    qx, qy = D * f_x, D * f_y
    dqx = torch.autograd.grad(qx, x, torch.ones_like(qx), create_graph=True)[0]
    dqy = torch.autograd.grad(qy, y, torch.ones_like(qy), create_graph=True)[0]
    return dqx + dqy

def be_consistency_loss(u_nn, v_nn, Du_fn, Dv_fn, F, k, N, dt,
                        sigma_u, sigma_v, t_lo=None):
    t_lo = t_min if t_lo is None else t_lo
    x = (torch.rand(N, 1, device=device) * float(x_max - x_min) + float(x_min)).requires_grad_(True)
    y = (torch.rand(N, 1, device=device) * float(y_max - y_min) + float(y_min)).requires_grad_(True)
    t0 = torch.rand(N, 1, device=device) * float(t_max - dt - t_lo) + float(t_lo)
    t1 = t0 + dt

    def inp(t):
        return torch.cat([normalize(x, x_min, x_max),
                          normalize(y, y_min, y_max),
                          normalize(t, t_min, t_max)], dim=1)

    X0, X1 = inp(t0), inp(t1)
    u0, v0 = u_nn(X0), v_nn(X0)
    u1, v1 = u_nn(X1), v_nn(X1)

    Du, Dv = Du_fn(x, y), Dv_fn(x, y)

    # Gray-Scott right-hand sides, evaluated at t+dt (backward Euler)
    div_u, div_v = div_flux(Du, u1, x, y), div_flux(Dv, v1, x, y)
    R_u = div_u - u1 * v1**2 + F * (1 - u1)
    R_v = div_v + u1 * v1**2 - (F + k) * v1

    u_pi = u1 - dt * R_u          # predicted u at time t
    v_pi = v1 - dt * R_v

    loss_u = torch.mean(((u_pi - u0) / sigma_u) ** 2)
    loss_v = torch.mean(((v_pi - v0) / sigma_v) ** 2)
    return div_u.mean(),div_v.mean(),loss_u,loss_v,loss_u + loss_v

# Importing the true solution for normalization

file_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\data\\data_{param}.txt"

# Generate column names
times = range(0, 3001, 10)

columns = ["x", "y"]

for t in times:
    columns += [f"u_{t}", f"v_{t}"]

# Read the data
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="%",
    header=None,
    names=columns
)

print(f"Data loaded from {file_path}. Shape: {df.shape}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

print("GPU available:", torch.cuda.is_available())

# Sampling Data

# Separate u and v columns
u_df = df[["x", "y"] + [col for col in df.columns if col.startswith("u_")]]
v_df = df[["x", "y"] + [col for col in df.columns if col.startswith("v_")]]

# Randomly sample 1000 data points from each


u_sample = u_df.sample(n=n_data, random_state=42)
v_sample = v_df.sample(n=n_data, random_state=42)

times = np.arange(0, 3001, 10)

# u data
x_u = np.repeat(u_sample["x"].values, len(times))
y_u = np.repeat(u_sample["y"].values, len(times))
t_u = np.tile(times, len(u_sample))

u_true = u_sample[
    [f"u_{t}" for t in times]
].values.flatten()

# v data
x_v = np.repeat(v_sample["x"].values, len(times))
y_v = np.repeat(v_sample["y"].values, len(times))
t_v = np.tile(times, len(v_sample))

v_true = v_sample[
    [f"v_{t}" for t in times]
].values.flatten()

X_u = torch.tensor(
    np.column_stack([x_u, y_u, t_u]),
    dtype=torch.float32
)

U_true = torch.tensor(
    u_true.reshape(-1, 1),
    dtype=torch.float32
)

X_v = torch.tensor(
    np.column_stack([x_v, y_v, t_v]),
    dtype=torch.float32
)

V_true = torch.tensor(
    v_true.reshape(-1, 1),
    dtype=torch.float32
)

X_u, U_true = X_u.to(device), U_true.to(device)
X_v, V_true = X_v.to(device), V_true.to(device)


sigma_u = U_true.std().item()
sigma_v = V_true.std().item()
dt_be = 20.0        # physical time units; data spacing is 10
t_lo_be = 500.0     # skip early times where the fields are almost flat (no information on D)
# Loading Pretraining u and v

class u_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self, X):
        # xt = torch.cat([x, y, t], dim=-1)
        return self.network(X)

class v_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self, X):
        # xt = torch.cat([x, y, t], dim=-1)
        return self.network(X)


pretrain_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\models\\pretrained\\uv_{n_data}_spatial_points.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(pretrain_path, map_location=device)

u_nn = u_net().to(device)
v_nn = v_net().to(device)
u_nn.load_state_dict(ckpt["u_nn"])
v_nn.load_state_dict(ckpt["v_nn"])

# Sanity check: the normalization bounds must match the ones used in pre-training
current = dict(x_min=float(x_min), x_max=float(x_max),
               y_min=float(y_min), y_max=float(y_max),
               t_min=float(t_min), t_max=float(t_max))
assert all(abs(ckpt["bounds"][k] - current[k]) < 1e-9 for k in current), \
    f"Bounds mismatch! saved={ckpt['bounds']} current={current}"

print("Loaded pre-trained u_nn and v_nn.")

# Training

Du_nn = Du_net().to(device)
Dv_nn = Dv_net().to(device)

Du_fn, Dv_fn = make_D_fns(Du_nn, Dv_nn)

if fine_tuning==True:

    stage_a_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\models\\cons_pinn\\checkpoints_{param}\\trained_stage_a\\uv_{n_data}_spatial_pts_stage_a.pt"
    test_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\models\\test\\uv_{n_data}_pts_stage_b.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(test_path, map_location=device)

    # Check that the normalization bounds match
    current = dict(x_min=float(x_min), x_max=float(x_max),
                y_min=float(y_min), y_max=float(y_max),
                t_min=float(t_min), t_max=float(t_max))
    assert all(abs(ckpt["bounds"][k_] - current[k_]) < 1e-9 for k_ in current), \
        f"Bounds mismatch! saved={ckpt['bounds']} current={current}"

    # Build the networks (same architecture as when saved)
    u_nn  = u_net().to(device)          # or ScaledMLP().to(device) if you use that class
    v_nn  = v_net().to(device)          # or ScaledMLP().to(device)
    Du_nn = Du_net().to(device)
    Dv_nn = Dv_net().to(device)

    u_nn.load_state_dict(ckpt["u_nn"])
    v_nn.load_state_dict(ckpt["v_nn"])
    Du_nn.load_state_dict(ckpt["Du_nn"])
    Dv_nn.load_state_dict(ckpt["Dv_nn"])

    # Re-create the D function wrappers around the loaded networks
    Du_fn, Dv_fn = make_D_fns(Du_nn, Dv_nn)


    print(f"Loaded Stage A checkpoint. Last step:", ckpt["step"])

# Training

# ---------------------------------------------------------
# 0. Unfreeze u, v (they were frozen in Stage A) and switch to train mode
# ---------------------------------------------------------
for p in list(u_nn.parameters()) + list(v_nn.parameters()):
    p.requires_grad_(True)
for m in (u_nn, v_nn, Du_nn, Dv_nn):
    m.train()

# ---------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------
n_B_steps = 5000
data_bs = 8192            # data points per step
N_be = 10000              # consistency-loss points per step
lambda_data = 1.0
lambda_pde_max = 1.0      # tune this: lower if the data loss rises, higher if D stalls
ramp_steps = 100        # PDE weight rises linearly from 0 to lambda_pde_max

uv_params = list(u_nn.parameters()) + list(v_nn.parameters())
D_params = list(Du_nn.parameters()) + list(Dv_nn.parameters())

optimizer_B = torch.optim.Adam([
    {"params": uv_params, "lr": 5e-4},     # small: don't destroy the pre-trained fit
    {"params": D_params,  "lr": 5e-4},
])
sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_B, T_max=n_B_steps, eta_min=1e-6)

# ---------------------------------------------------------
# 2. Data loss, normalized by sigma (same units as the consistency loss)
# ---------------------------------------------------------
def data_loss_norm(idx):
    Xu_b, Xv_b = X_u[idx], X_v[idx]
    def norm_in(X):
        return torch.cat([normalize(X[:, 0:1], x_min, x_max),
                          normalize(X[:, 1:2], y_min, y_max),
                          normalize(X[:, 2:3], t_min, t_max)], dim=1)
    lu = torch.mean(((u_nn(norm_in(Xu_b)) - U_true[idx]) / sigma_u) ** 2)
    lv = torch.mean(((v_nn(norm_in(Xv_b)) - V_true[idx]) / sigma_v) ** 2)
    return lu + lv

# ---------------------------------------------------------
# 3. Monitoring of D error (uses your known true D, for monitoring only)
# ---------------------------------------------------------
mon_idx = np.random.RandomState(0).choice(len(df), 5000, replace=False)
x_mon = torch.tensor(df["x"].values[mon_idx], dtype=torch.float32, device=device).reshape(-1, 1)
y_mon = torch.tensor(df["y"].values[mon_idx], dtype=torch.float32, device=device).reshape(-1, 1)
Du_true_mon = (2e-5 * (1 + torch.sin(np.pi * x_mon) * torch.sin(np.pi * y_mon))).cpu().numpy().ravel()

@torch.no_grad()
def du_errors():
    pred = Du_nn(torch.cat([normalize(x_mon, x_min, x_max),
                            normalize(y_mon, y_min, y_max)], dim=1)).cpu().numpy().ravel()
    mse = np.mean((pred - Du_true_mon) ** 2)
    L_paper = np.sqrt(mse / np.mean((Du_true_mon - Du_true_mon.mean()) ** 2))
    return mse, L_paper

# ---------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------
hist = {"data": [], "pde": [], "Du_L": []}
N_rows = X_u.shape[0]

for i in range(3):
    print(f"Stage B training, run {i+1}/3")
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_B,
        T_max=n_B_steps,
        eta_min=1e-5
    )
    for step in range(n_B_steps):
        if i==0:
            w_pde = lambda_pde_max * min(1.0, (step + 1) / ramp_steps)

        idx = torch.randint(0, N_rows, (data_bs,), device=device)
        L_data = data_loss_norm(idx)

        div_u, div_v, loss_u, loss_v, L_pde = be_consistency_loss(u_nn, v_nn, Du_fn, Dv_fn, F, k,
                                    N_be, dt_be, sigma_u, sigma_v, t_lo=t_lo_be)

        L = lambda_data * L_data + w_pde * L_pde

        optimizer_B.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(uv_params + D_params, 1.0)
        optimizer_B.step()
        sched_B.step()

        if (step + 1) % 50 == 0:
            mse_D, L_D = du_errors()
            hist["data"].append(L_data.item()); hist["pde"].append(L_pde.item()); hist["Du_L"].append(L_D)
            if (step + 1) % 50 == 0:
                print(f"step {step+1}/{n_B_steps} | data {L_data.item():.3e} | pde {L_pde.item():.3e} "
                    f"| w_pde {w_pde:.2f} | Du MSE {mse_D:.2e} | Du L {L_D*100:.1f}% "
                    f"| Dv {Dv_nn.value().item():.3e}")




# Saving the trained model

import os

save_dir = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\models\\test"
os.makedirs(save_dir, exist_ok=True)
stageB_path = os.path.join(save_dir, f"uv_{n_data}_pts_stage_b.pt")

torch.save({
    "u_nn": u_nn.state_dict(),
    "v_nn": v_nn.state_dict(),
    "Du_nn": Du_nn.state_dict(),
    "Dv_nn": Dv_nn.state_dict(),      # contains Dv_raw
    "optimizer_B": optimizer_B.state_dict(),
    "sched_B": sched_B.state_dict(),
    "step": step + 1,                 # last finished Stage B step
    "dt_be": dt_be,
    "t_lo_be": t_lo_be,
    "sigma_u": sigma_u,
    "sigma_v": sigma_v,
    "bounds": dict(x_min=float(x_min), x_max=float(x_max),
                   y_min=float(y_min), y_max=float(y_max),
                   t_min=float(t_min), t_max=float(t_max)),
}, stageB_path)

print("Saved Stage B checkpoint to", stageB_path)
