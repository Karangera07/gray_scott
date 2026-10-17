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

fine_tuning = True

print("Libraries imported successfully.")

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


# Importing Data

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

# print("u_df shape:", u_df.shape)
# print("v_df shape:", v_df.shape)

# print("u_sample shape:", u_sample.shape)
# print("v_sample shape:", v_sample.shape)

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
    

# Sampling collocation points

def sample_collocation_points(N_col, x_min, x_max, y_min, y_max, t_min, t_max,device="cuda"):
    x = torch.rand(N_col, 1, device=device) * (x_max - x_min) + x_min
    y = torch.rand(N_col, 1, device=device) * (y_max - y_min) + y_min
    t = torch.rand(N_col, 1, device=device) * (t_max - t_min) + t_min
    return torch.cat([x, y, t], dim=1)

X_col = sample_collocation_points(N_collocation, x_min, x_max, y_min, y_max, t_min, t_max, device=device)

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


# Defining PDE Residuals

def Du_true(x, y):
    return 2e-5 * (1.0 + torch.sin(math.pi * x) * torch.sin(math.pi * y))

def Dv_true(x, y):
    return torch.full_like(x, 1e-5)

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

def pde_loss(
    u_net, v_net, Du_net,
    X_col,
    Dv_net, F, k,
    x_min, x_max,
    y_min, y_max,
    t_min, t_max,
    return_terms=False
):

    # --------------------------------------------------
    # Normalization factors
    # --------------------------------------------------

    sx = 2.0 / (x_max - x_min)
    sy = 2.0 / (y_max - y_min)
    st = 2.0 / (t_max - t_min)

    # --------------------------------------------------
    # Normalize collocation points
    # --------------------------------------------------

    X_raw = X_col.clone().requires_grad_(True)

    x = normalize(X_raw[:, 0:1], x_min, x_max)
    y = normalize(X_raw[:, 1:2], y_min, y_max)
    t = normalize(X_raw[:, 2:3], t_min, t_max)

    X = torch.cat([x, y, t], dim=1)

    # --------------------------------------------------
    # Network predictions
    # --------------------------------------------------

    u = u_net(X)
    v = v_net(X)

    # D = D(x,y)
    D = Du_net(X[:, 0:2])
    Dv = Dv_net(X[:, 0:2])

    # True diffusivities at physical coordinates
    Du_exact = Du_true(X_raw[:, 0:1], X_raw[:, 1:2])
    Dv_exact = Dv_true(X_raw[:, 0:1], X_raw[:, 1:2])

    # MSE of predicted diffusivities
    mse_Du = torch.mean((D - Du_exact)**2)
    mse_Dv = torch.mean((Dv - Dv_exact)**2)

    # --------------------------------------------------
    # First derivatives of u
    # --------------------------------------------------

    u_grad = torch.autograd.grad(
        u,
        X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xn = u_grad[:, 0:1]
    u_yn = u_grad[:, 1:2]
    u_tn = u_grad[:, 2:3]

    # Convert to physical derivatives
    u_x = sx * u_xn
    u_y = sy * u_yn
    u_t = st * u_tn

    # --------------------------------------------------
    # Second derivatives of u
    # --------------------------------------------------

    u_xn_xn = torch.autograd.grad(
        u_xn,
        X,
        grad_outputs=torch.ones_like(u_xn),
        create_graph=True
    )[0][:, 0:1]

    u_yn_yn = torch.autograd.grad(
        u_yn,
        X,
        grad_outputs=torch.ones_like(u_yn),
        create_graph=True
    )[0][:, 1:2]

    # Convert to physical second derivatives
    u_xx = sx**2 * u_xn_xn
    u_yy = sy**2 * u_yn_yn

    # --------------------------------------------------
    # Derivatives of D(x,y)
    # --------------------------------------------------

    D_grad = torch.autograd.grad(
        D,
        X,
        grad_outputs=torch.ones_like(D),
        create_graph=True
    )[0]

    D_xn = D_grad[:, 0:1]
    D_yn = D_grad[:, 1:2]

    # Convert to physical derivatives
    D_x = sx * D_xn
    D_y = sy * D_yn

    # --------------------------------------------------
    # First derivatives of v
    # --------------------------------------------------

    v_grad = torch.autograd.grad(
        v,
        X,
        grad_outputs=torch.ones_like(v),
        create_graph=True
    )[0]

    v_xn = v_grad[:, 0:1]
    v_yn = v_grad[:, 1:2]
    v_tn = v_grad[:, 2:3]

    # Physical derivatives
    v_x = sx * v_xn
    v_y = sy * v_yn
    v_t = st * v_tn

    # --------------------------------------------------
    # Second derivatives of v
    # --------------------------------------------------

    v_xn_xn = torch.autograd.grad(
        v_xn,
        X,
        grad_outputs=torch.ones_like(v_xn),
        create_graph=True
    )[0][:, 0:1]

    v_yn_yn = torch.autograd.grad(
        v_yn,
        X,
        grad_outputs=torch.ones_like(v_yn),
        create_graph=True
    )[0][:, 1:2]

    # Physical second derivatives
    v_xx = sx**2 * v_xn_xn
    v_yy = sy**2 * v_yn_yn

    # --------------------------------------------------
    # Diffusion terms
    # --------------------------------------------------

    # div(D grad(u))
    diffusion_u = (
        D * (u_xx + u_yy)
        + D_x * u_x
        + D_y * u_y
    )

    # Dv is constant
    diffusion_v = Dv * (v_xx + v_yy)

    # --------------------------------------------------
    # Gray-Scott reaction terms
    # --------------------------------------------------

    reaction_u = -u * v**2 + F * (1.0 - u)

    reaction_v = u * v**2 - (F + k) * v

    # --------------------------------------------------
    # PDE residuals
    # --------------------------------------------------

    f_u = u_t - diffusion_u - reaction_u

    f_v = v_t - diffusion_v - reaction_v

    # --------------------------------------------------
    # PDE loss
    # --------------------------------------------------

    T_scale = t_max - t_min          # measure the residual per full time span
    loss_u = torch.mean((T_scale * f_u)**2)
    loss_v = torch.mean((T_scale * f_v)**2)

    loss_pde = loss_u + loss_v

    if return_terms:
        terms = {
            "|u_t|": u_t.abs().mean().item(),
            "|diff_u|": diffusion_u.abs().mean().item(),
            "|react_u|": reaction_u.abs().mean().item(),
            "|resid_u|": f_u.abs().mean().item(),
            "|v_t|": v_t.abs().mean().item(),
            "|diff_v|": diffusion_v.abs().mean().item(),
            "|react_v|": reaction_v.abs().mean().item(),
            "|resid_v|": f_v.abs().mean().item(),
            "MSE_Du": mse_Du.item(),
            "MSE_Dv": mse_Dv.item()
        }
        return loss_pde, terms

    return loss_pde

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


pretrain_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\python_files\\vannila_pinn\\pretrained\\uv_{n_data}_spatial_points.pt"

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


u_nn  = u_net().to(device)
v_nn  = v_net().to(device)
Du_nn = Du_net().to(device)
Dv_nn = Dv_net().to(device)

Du_fn, Dv_fn = make_D_fns(Du_nn, Dv_nn)

if fine_tuning==True:
    pretrain_path = f"C:\\Users\\GPU2\\gray_scott_projects\\gray_scott\\python_files\\vannila_pinn\\pretrained\\uv_{n_data}_spatial_points.pt"

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
# 2. Data loss
# ---------------------------------------------------------
def data_loss_norm(idx):
    Xu_b, Xv_b = X_u[idx], X_v[idx]
    def norm_in(X):
        return torch.cat([normalize(X[:, 0:1], x_min, x_max),
                          normalize(X[:, 1:2], y_min, y_max),
                          normalize(X[:, 2:3], t_min, t_max)], dim=1)
    lu = torch.mean(((u_nn(norm_in(Xu_b)) - U_true[idx])) ** 2)
    lv = torch.mean(((v_nn(norm_in(Xv_b)) - V_true[idx])) ** 2)
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
# hist = {"data": [], "pde": [], "Du_L": []}
N_rows = X_u.shape[0]

for step in range(n_B_steps):

    w_pde = lambda_pde_max * min(1.0, (step + 1) / ramp_steps)

    idx = torch.randint(0, N_rows, (data_bs,), device=device)
    L_data = data_loss_norm(idx)

    L_pde,terms = pde_loss(
        u_net, v_net, Du_net,
        X_col,
        Dv_net, F, k,
        x_min, x_max,
        y_min, y_max,
        t_min, t_max,
        return_terms=True
    )

    L = lambda_data * L_data + w_pde * L_pde

    optimizer_B.zero_grad()
    L.backward()
    torch.nn.utils.clip_grad_norm_(uv_params + D_params, 1.0)
    optimizer_B.step()
    sched_B.step()

    if (step + 1) % 5 == 0:
        mse_D, L_D = du_errors()
        # hist["data"].append(L_data.item()); hist["pde"].append(L_pde.item()); hist["Du_L"].append(L_D)
        if (step + 1) % 5 == 0:
            print(f"step {step+1}/{n_B_steps} | data {L_data.item():.3e}",terms)


