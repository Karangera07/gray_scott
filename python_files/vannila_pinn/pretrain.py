# Importing Libraries

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd

print("Libraries imported successfully.")

# Step 1: Define problem

L = 1.0  # length of the domain
x_min, x_max = 0.0, L
y_min, y_max = 0.0, L
t_min, t_max = 0.0, 3000.0

# dimensionality
input_dim = 3   # (x, y, t)
output_dim = 1  # C only


# Select Case (Pair of Params) --> 

# Case 1 - F = 0.037 k = 0.06

# Case 2 - F = 0.014 k = 0.045

# Case 3 - F = 0.018 k = 0.051

param = 1
n_data = 10000

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

# Importing Data

file_path = f"data/data_{param}.txt"

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

def normalize(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

# Define the neural network architecture

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

u_nn  = u_net().to(device)
v_nn  = v_net().to(device)

import math

batch_size = 32768   # was 4096, see Fix 2
n_batches = math.ceil(X_u.shape[0] / batch_size)

def batch_iterator(X, U, V, batch_size):
    perm = torch.randperm(X.shape[0], device=X.device)
    for i in range(0, X.shape[0], batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], U[idx], V[idx]

def pretraining_loss_u(
    u_net,
    X_batch,
    U_true_batch,
    V_true_batch,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max
):

    # Normalize inputs
    x_n = normalize(X_batch[:, 0:1], x_min, x_max)
    y_n = normalize(X_batch[:, 1:2], y_min, y_max)
    t_n = normalize(X_batch[:, 2:3], t_min, t_max)

    X_norm = torch.cat([x_n, y_n, t_n], dim=1)

    # Predictions
    u_pred = u_net(X_norm)
    # v_pred = v_net(X_norm)

    # Data losses
    loss_u = torch.mean((u_pred - U_true_batch) ** 2)
    # loss_v = torch.mean((v_pred - V_true_batch) ** 2)

    # Total pre-training loss
    # loss = loss_u + loss_v

    return loss_u


def pretraining_loss_v(
    v_net,
    X_batch,
    U_true_batch,
    V_true_batch,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max
):

    # Normalize inputs
    x_n = normalize(X_batch[:, 0:1], x_min, x_max)
    y_n = normalize(X_batch[:, 1:2], y_min, y_max)
    t_n = normalize(X_batch[:, 2:3], t_min, t_max)

    X_norm = torch.cat([x_n, y_n, t_n], dim=1)

    # Predictions
    # u_pred = u_net(X_norm)
    v_pred = v_net(X_norm)

    # Data losses
    # loss_u = torch.mean((u_pred - U_true_batch) ** 2)
    loss_v = torch.mean((v_pred - V_true_batch) ** 2)

    # Total pre-training loss
    # loss = loss_u + loss_v

    return loss_v

pre_batch_size = 16384      # small batches = many steps (pre-training is cheap)
pre_epochs = 500
n_pre = math.ceil(X_u.shape[0] / pre_batch_size)

optimizer_pretrain_u = torch.optim.Adam(
    list(u_nn.parameters()), lr=2e-3
)
scheduler_u = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_pretrain_u, T_max=pre_epochs, eta_min=1e-5
)

for epoch in range(pre_epochs):
    tot_u = 0.0
    # tot_v = 0.0

    for X_batch, U_b, V_b in batch_iterator(X_u, U_true, V_true, pre_batch_size):
        loss_u = pretraining_loss_u(
            u_nn, X_batch, U_b, V_b,
            x_min, x_max, y_min, y_max, t_min, t_max
        )
        optimizer_pretrain_u.zero_grad()
        loss_u.backward()

        optimizer_pretrain_u.step()
        tot_u += loss_u.detach()


    scheduler_u.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{pre_epochs} | "
              f"u MSE {tot_u.item()/n_pre:.3e} (var {U_true.var().item():.2e}) ")


optimizer_pretrain_v = torch.optim.Adam(
    list(v_nn.parameters()), lr=2e-3
)
scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_pretrain_v, T_max=pre_epochs, eta_min=1e-5
)

for epoch in range(pre_epochs):

    tot_v = 0.0

    for X_batch, U_b, V_b in batch_iterator(X_u, U_true, V_true, pre_batch_size):
        loss_v = pretraining_loss_v(
            v_nn, X_batch, U_b, V_b,
            x_min, x_max, y_min, y_max, t_min, t_max
        )
        optimizer_pretrain_v.zero_grad()
        loss_v.backward()
        optimizer_pretrain_v.step()
        tot_v += loss_v.detach()

    scheduler_v.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{pre_epochs} | "
              f"v MSE {tot_v.item()/n_pre:.3e} (var {V_true.var().item():.2e})")


