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

# Environment Constants
K_SPRING = 2.0
B_DAMPER = 0.1
M_MASS = 1.0
DT = 0.01

# -----------------------------------------------
# Pure-GPU Data Generation
# -----------------------------------------------
def make_spring_data(n=500000):
    print(f"Generating {n:,} Spring samples on GPU...")
    t0 = time.time()
    torch.manual_seed(42)
    with torch.no_grad():
        # [x, y, z, vx, vy, vz]
        pos = torch.empty(n, 3, device=device).uniform_(-2.0, 2.0)
        vel = torch.empty(n, 3, device=device).uniform_(-2.0, 2.0)
        S = torch.cat([pos, vel], dim=1)
        
        # Residual acceleration (Hooke's Law)
        # dv = (-k/m * x) * dt
        accel = -(K_SPRING / M_MASS) * pos
        dV = accel * DT
        # dx = (v + 0.5*a*dt)*dt -> Using simple Euler residual for training
        dX = (vel + 0.5 * accel * DT) * DT
        Y = torch.cat([dX, dV], dim=1)
    print(f"Done: {time.time()-t0:.1f}s")
    return S, Y

def make_damper_data(n=500000):
    print(f"Generating {n:,} Damper samples on GPU...")
    t0 = time.time()
    torch.manual_seed(43)
    with torch.no_grad():
        pos = torch.empty(n, 3, device=device).uniform_(-2.0, 2.0)
        vel = torch.empty(n, 3, device=device).uniform_(-2.0, 2.0)
        S = torch.cat([pos, vel], dim=1)
        
        # Linear Damping: a = -b/m * v
        accel = -(B_DAMPER / M_MASS) * vel
        dV = accel * DT
        dX = (vel + 0.5 * accel * DT) * DT
        Y = torch.cat([dX, dV], dim=1)
    print(f"Done: {time.time()-t0:.1f}s")
    return S, Y

# -----------------------------------------------
# Elite Architecture
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

# -----------------------------------------------
# Pure-GPU Training (No DataLoader)
# -----------------------------------------------
def train_elite(S, Y, seed=0, epochs=100):
    torch.manual_seed(seed)
    model = Brain().to(device)
    n = S.shape[0]
    batch = 32768
    
    # Pre-normalize on GPU
    Sm, Ss = S.mean(0), S.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Sn = (S - Sm) / Ss
    Yn = (Y - Ym) / Ys
    
    opt = optim.Adam(model.parameters(), lr=2e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    t0 = time.time()
    for epoch in range(epochs+1):
        idx = torch.randperm(n, device=device)
        total_loss = 0
        nb = 0
        for start in range(0, n, batch):
            end = min(start + batch, n)
            bx = Sn[idx[start:end]]
            by = Yn[idx[start:end]]
            
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = nn.MSELoss()(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            nb += 1
        sch.step()
        if epoch % 25 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss/nb:.8f} | GPU: {torch.cuda.utilization()}%")
            
    return model.cpu(), (Sm.cpu(), Ss.cpu(), Ym.cpu(), Ys.cpu())

class Ensemble(nn.Module):
    def __init__(self, models_data):
        super().__init__()
        self.models = nn.ModuleList([m[0] for m in models_data])
        # Use first model's norms (they should be identical for same data)
        norms = models_data[0][1]
        self.register_buffer('Sm', norms[0])
        self.register_buffer('Ss', norms[1])
        self.register_buffer('Ym', norms[2])
        self.register_buffer('Ys', norms[3])
        
    @torch.no_grad()
    def forward(self, s):
        sn = (s - self.Sm) / self.Ss
        preds = torch.stack([m(sn) for m in self.models])
        yn = preds.mean(0)
        return yn * self.Ys + self.Ym

# -----------------------------------------------
# Evaluation
# -----------------------------------------------
def run_composition(ens_s, ens_d, steps=500):
    # Initial state: stretched spring
    s = torch.tensor([[1.5, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
    traj = [s.cpu().numpy()]
    energies = []
    
    def get_energy(st):
        # E = 0.5*m*v^2 + 0.5*k*r^2
        r_sq = torch.sum(st[:, :3]**2, dim=1)
        v_sq = torch.sum(st[:, 3:6]**2, dim=1)
        return (0.5 * M_MASS * v_sq + 0.5 * K_SPRING * r_sq).item()

    print(f"Starting Rollout (Zero-Shot Composition)...")
    for _ in range(steps):
        e = get_energy(s)
        energies.append(e)
        
        # Zero-Shot composition of residuals
        # Both predict full 6D residual, we sum them
        # Note: position residual dx is mostly kinematics (v*dt), 
        # so we should be careful not to double count.
        # Physics residual is dv.
        res_s = ens_s(s)
        res_d = ens_d(s)
        
        # Sum velocity residuals (forces are additive)
        # Position residual: take one (since both should learn v*dt)
        # or average them. Taking one is safer for kinematic identity.
        ds = torch.zeros_like(s)
        ds[:, 0:3] = res_s[:, 0:3] # Kinematic dx
        ds[:, 3:6] = res_s[:, 3:6] + res_d[:, 3:6] # Force composition dv
        
        s = s + ds
        traj.append(s.cpu().numpy())
    
    drift = abs(energies[-1] - energies[0])
    return np.array(traj).squeeze(), np.array(energies), drift

if __name__ == "__main__":
    print("=== ELITE 3D SPRING-DAMPER COMPOSITION ===")
    
    # 1. Data
    S_s, Y_s = make_spring_data()
    S_d, Y_d = make_damper_data()
    
    # 2. Train Ensemble
    print("\nTraining Spring Ensemble (5 brains)...")
    ens_s_data = [train_elite(S_s, Y_s, seed=i, epochs=100) for i in range(5)]
    ens_s = Ensemble(ens_s_data).to(device)
    
    print("\nTraining Damper Ensemble (5 brains)...")
    ens_d_data = [train_elite(S_d, Y_d, seed=i+10, epochs=100) for i in range(5)]
    ens_d = Ensemble(ens_d_data).to(device)
    
    # 3. Rollout
    traj, energy, drift = run_composition(ens_s, ens_d)
    
    print(f"\n{'='*40}")
    print(f"Composition Success!")
    print(f"Energy Drift: {drift:.6f}")
    print(f"{'='*40}")
    
    # 4. Viz
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(traj[:, 0], label='X (Spring)')
    plt.title("3D Damped Oscillation (Zero-Shot)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(energy, color='red')
    plt.title(f"Hamiltonian (Drift: {drift:.4f})")
    
    plt.tight_layout()
    plt.savefig("spring_composition_elite.png")
    print("Result saved to spring_composition_elite.png")
