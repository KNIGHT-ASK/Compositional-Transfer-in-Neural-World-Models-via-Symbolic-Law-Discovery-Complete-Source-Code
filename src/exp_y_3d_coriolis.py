import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Force GPU
assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# Constants
G_VAL = 9.81
OMEGA = torch.tensor([0.0, 0.0, 0.7], device=device) # Rotating frame (Z-axis)
CW = 0.1
V_WIND = torch.tensor([2.0, 0.0, 0.0], device=device)
DT = 0.01

# -----------------------------------------------
# Pure-GPU Data Generation
# -----------------------------------------------
def make_gravity_data(n=500000):
    print(f"Generating {n:,} Gravity samples...")
    torch.manual_seed(1)
    with torch.no_grad():
        pos = torch.empty(n, 3, device=device).uniform_(-10, 10)
        vel = torch.empty(n, 3, device=device).uniform_(-15, 15)
        S = torch.cat([pos, vel], dim=1)
        accel = torch.zeros(n, 3, device=device)
        accel[:, 2] = -G_VAL
        dV = accel * DT
        dX = (vel + 0.5 * accel * DT) * DT
        Y = torch.cat([dX, dV], dim=1)
    return S, Y

def make_coriolis_data(n=500000):
    print(f"Generating {n:,} Coriolis samples...")
    torch.manual_seed(2)
    with torch.no_grad():
        pos = torch.empty(n, 3, device=device).uniform_(-10, 10)
        vel = torch.empty(n, 3, device=device).uniform_(-15, 15)
        S = torch.cat([pos, vel], dim=1)
        # a = -2 * (Omega x v)
        # torch.cross requires dim=1
        accel = -2 * torch.cross(OMEGA.expand_as(vel), vel, dim=1)
        dV = accel * DT
        dX = (vel + 0.5 * accel * DT) * DT
        Y = torch.cat([dX, dV], dim=1)
    return S, Y

def make_wind_data(n=500000):
    print(f"Generating {n:,} Wind samples...")
    torch.manual_seed(3)
    with torch.no_grad():
        pos = torch.empty(n, 3, device=device).uniform_(-10, 10)
        vel = torch.empty(n, 3, device=device).uniform_(-15, 15)
        S = torch.cat([pos, vel], dim=1)
        # a = -Cw * (v - v_wind)
        accel = -CW * (vel - V_WIND)
        dV = accel * DT
        dX = (vel + 0.5 * accel * DT) * DT
        Y = torch.cat([dX, dV], dim=1)
    return S, Y

# -----------------------------------------------
# Elite Training (Pure GPU)
# -----------------------------------------------
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 6)
        )
    def forward(self, x):
        return self.net(x)

def train_elite(S, Y, seed=0, epochs=100):
    torch.manual_seed(seed)
    model = Brain().to(device)
    n, batch = S.shape[0], 32768
    Sm, Ss = S.mean(0), S.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Sn, Yn = (S - Sm) / Ss, (Y - Ym) / Ys
    opt = optim.Adam(model.parameters(), lr=2e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs+1):
        idx = torch.randperm(n, device=device)
        total_loss, nb = 0, 0
        for start in range(0, n, batch):
            end = min(start + batch, n)
            bx, by = Sn[idx[start:end]], Yn[idx[start:end]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = nn.MSELoss()(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); nb += 1
        sch.step()
        if epoch % 50 == 0: print(f"  Epoch {epoch:3d} | Loss: {total_loss/nb:.8f}")
    return model.cpu(), (Sm.cpu(), Ss.cpu(), Ym.cpu(), Ys.cpu())

class Ensemble(nn.Module):
    def __init__(self, models_data):
        super().__init__()
        self.models = nn.ModuleList([m[0] for m in models_data])
        norms = models_data[0][1]
        self.register_buffer('Sm', norms[0]); self.register_buffer('Ss', norms[1])
        self.register_buffer('Ym', norms[2]); self.register_buffer('Ys', norms[3])
    @torch.no_grad()
    def forward(self, s):
        sn = (s - self.Sm) / self.Ss
        preds = torch.stack([m(sn) for m in self.models])
        return preds.mean(0) * self.Ys + self.Ym

# -----------------------------------------------
# Triple Composition Evaluation
# -----------------------------------------------
def run_triple_composition(ens_g, ens_c, ens_w, steps=500):
    # Initial state: projectile fired upwards
    s = torch.tensor([[0.0, 0.0, 0.0, 5.0, 2.0, 20.0]], device=device)
    traj = [s.cpu().numpy()]
    print(f"Starting Rollout (Zero-Shot Triple Composition)...")
    for _ in range(steps):
        # res = [dx, dv]
        r_g = ens_g(s)
        r_c = ens_c(s)
        r_w = ens_w(s)
        
        # Compose: Sum force residuals (dv), take one kinematic residual (dx)
        ds = torch.zeros_like(s)
        ds[:, 0:3] = r_g[:, 0:3]
        ds[:, 3:6] = r_g[:, 3:6] + r_c[:, 3:6] + r_w[:, 3:6]
        
        s = s + ds
        traj.append(s.cpu().numpy())
    return np.array(traj).squeeze()

if __name__ == "__main__":
    print("=== ELITE 3D PROJECTILE TRIPLE COMPOSITION ===")
    
    # 1. Data
    S_g, Y_g = make_gravity_data()
    S_c, Y_c = make_coriolis_data()
    S_w, Y_w = make_wind_data()
    
    # 2. Train Ensembles
    print("\nTraining Gravity Ensemble...")
    ens_g = Ensemble([train_elite(S_g, Y_g, seed=i) for i in range(5)]).to(device)
    print("\nTraining Coriolis Ensemble...")
    ens_c = Ensemble([train_elite(S_c, Y_c, seed=i+10) for i in range(5)]).to(device)
    print("\nTraining Wind Ensemble...")
    ens_w = Ensemble([train_elite(S_w, Y_w, seed=i+20) for i in range(5)]).to(device)
    
    # 3. Rollout
    traj = run_triple_composition(ens_g, ens_c, ens_w)
    
    # 4. Viz
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='teal')
    ax1.set_title("3D Trajectory (Zero-Shot Triple)")
    ax1.set_xlabel("X (Wind Side)"); ax1.set_ylabel("Y (Coriolis Shift)"); ax1.set_zlabel("Z (Gravity)")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(traj[:, 2], label='Height')
    ax2.set_title("Vertical Profile")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("coriolis_triple_composition.png")
    print("\nTriple Composition Complete. Saved: coriolis_triple_composition.png")
