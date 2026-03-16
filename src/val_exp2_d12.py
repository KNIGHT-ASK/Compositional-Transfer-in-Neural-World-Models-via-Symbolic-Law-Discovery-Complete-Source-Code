import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------------------------
# Constants
# -----------------------------------------------
G        = 1.0
K        = 2.0
L0       = 1.0
EPS      = 0.05
DT       = 0.005
N        = 500000
N_MODELS = 5
EPOCHS   = 100
BATCH    = 65536
STEPS    = 1000

# -----------------------------------------------
# CORE TRANSFORM: Symmetric-Log (Singularity Handler)
# -----------------------------------------------
def symlog(x, k=0.1):
    return torch.sign(x) * torch.log(1 + torch.abs(x) / k)

def symexp(x, k=0.1):
    return torch.sign(x) * k * (torch.exp(torch.abs(x)) - 1)

# -----------------------------------------------
# DATA GENERATION (Batch-Optimized)
# -----------------------------------------------
def sample_bodies(n, seed):
    torch.manual_seed(seed)
    r1 = torch.empty(n, 3, device=device).uniform_(-5, 5)
    v1 = torch.empty(n, 3, device=device).uniform_(-3, 3)
    r2 = torch.empty(n, 3, device=device).uniform_(-5, 5)
    v2 = torch.empty(n, 3, device=device).uniform_(-3, 3)
    return r1, v1, r2, v2

def make_gravity_data(n=N):
    print(f"  Gravity: {n:,} samples...")
    with torch.no_grad():
        r1, v1, r2, v2 = sample_bodies(n, seed=10)
        # 6D relative input (trans/rot invariance)
        S = torch.cat([r2 - r1, v2 - v1], dim=1)
        
        r_rel = r2 - r1
        r_mag = torch.norm(r_rel, dim=1, keepdim=True)
        r_soft = torch.sqrt(r_mag**2 + EPS**2)
        ag1 = G * r_rel / r_soft**3
        ag2 = -ag1
        
        # 12D residuals
        dx1 = (v1 + 0.5*ag1*DT)*DT; dv1 = ag1*DT
        dx2 = (v2 + 0.5*ag2*DT)*DT; dv2 = ag2*DT
        Y = torch.cat([dx1, dv1, dx2, dv2], dim=1)
    return S, Y

def make_spring_data(n=N):
    print(f"  Spring: {n:,} samples...")
    with torch.no_grad():
        r1, v1, r2, v2 = sample_bodies(n, seed=20)
        S = torch.cat([r2 - r1, v2 - v1], dim=1)
        
        r_rel = r2 - r1
        r_mag = torch.norm(r_rel, dim=1, keepdim=True).clamp(min=1e-6)
        r_hat = r_rel / r_mag
        stretch = r_mag - L0
        as1 = K * stretch * r_hat
        as2 = -as1
        
        dx1 = (v1 + 0.5*as1*DT)*DT; dv1 = as1*DT
        dx2 = (v2 + 0.5*as2*DT)*DT; dv2 = as2*DT
        Y = torch.cat([dx1, dv1, dx2, dv2], dim=1)
    return S, Y

def make_combined_data(n=N):
    print(f"  Combined: {n:,} samples...")
    with torch.no_grad():
        r1, v1, r2, v2 = sample_bodies(n, seed=30)
        S = torch.cat([r2 - r1, v2 - v1], dim=1)
        r_rel = r2 - r1
        r_mag = torch.norm(r_rel, dim=1, keepdim=True)
        # Gravity
        r_soft = torch.sqrt(r_mag**2 + EPS**2)
        ag1 = G * r_rel / r_soft**3; ag2 = -ag1
        # Spring
        r_hat = r_rel / r_mag.clamp(min=1e-6)
        stretch = r_mag - L0
        as1 = K * stretch * r_hat; as2 = -as1
        # Sum
        a1 = ag1 + as1; a2 = ag2 + as2
        dx1 = (v1 + 0.5*a1*DT)*DT; dv1 = a1*DT
        dx2 = (v2 + 0.5*a2*DT)*DT; dv2 = a2*DT
        Y = torch.cat([dx1, dv1, dx2, dv2], dim=1)
    return S, Y

# -----------------------------------------------
# ARCHITECTURE & TRAINING
# -----------------------------------------------
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 12)
        )
    def forward(self, x): return self.net(x)

def train(S, Y, label='', seed=0):
    torch.manual_seed(seed)
    model = Brain().to(device)
    n = S.shape[0]

    # Target Linearization: SymLog handles the force singularities
    Y_sl = symlog(Y)
    
    Sm, Ss = S.mean(0), S.std(0) + 1e-8
    Ym, Ys = Y_sl.mean(0), Y_sl.std(0) + 1e-8
    Sn, Yn = (S - Sm) / Ss, (Y_sl - Ym) / Ys

    opt = optim.Adam(model.parameters(), lr=3e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    
    t0 = time.time()
    for epoch in range(EPOCHS + 1):
        idx = torch.randperm(n, device=device)
        total_loss = 0
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            bx, by = Sn[idx[s:e]], Yn[idx[s:e]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = nn.MSELoss()(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total_loss += loss.item() * (e - s)
        sch.step()
        if epoch % 50 == 0:
            print(f"  [{label}] Ep{epoch:3d} | Loss:{total_loss/n:.8f} | {time.time()-t0:.0f}s")
    
    return model.cpu(), (Sm.cpu(), Ss.cpu(), Ym.cpu(), Ys.cpu())

class Ensemble(nn.Module):
    def __init__(self, models_data):
        super().__init__()
        self.models = nn.ModuleList([m[0] for m in models_data])
        Sm, Ss, Ym, Ys = models_data[0][1]
        self.register_buffer('Sm', Sm); self.register_buffer('Ss', Ss)
        self.register_buffer('Ym', Ym); self.register_buffer('Ys', Ys)

    @torch.no_grad()
    def forward(self, s_rel):
        sn = (s_rel - self.Sm) / (self.Ss + 1e-8)
        p = torch.stack([m(sn) for m in self.models]).mean(0)
        # VERIFIED ORDER: 1. Denormalize (Invert Step 2) -> 2. SymExp (Invert Step 1)
        # This prevents the corrupted output seen in previous power-law attempts.
        return symexp(p * self.Ys + self.Ym)

# -----------------------------------------------
# EVALUATION & ROLLOUT
# -----------------------------------------------
def gt_step(s):
    r1, v1, r2, v2 = s[0:3], s[3:6], s[6:9], s[9:12]
    r_rel = r2 - r1
    r_mag = np.linalg.norm(r_rel)
    r_s = np.sqrt(r_mag**2 + EPS**2)
    # Combined Acceleration
    ag1 = G * r_rel / r_s**3
    r_hat = r_rel / max(r_mag, 1e-6)
    as1 = K * (r_mag - L0) * r_hat
    a1 = ag1 + as1; a2 = -a1
    # Residual Update
    v1n = v1 + a1 * DT
    v2n = v2 + a2 * DT
    r1n = r1 + 0.5 * (v1 + v1n) * DT
    r2n = r2 + 0.5 * (v2 + v2n) * DT
    return np.concatenate([r1n, v1n, r2n, v2n])

def run_zeroshot(ens_g, ens_sp, s_init, steps=STEPS):
    s = s_init.copy()
    traj = [s.copy()]
    for _ in range(steps):
        r1, v1, r2, v2 = s[0:3], s[3:6], s[6:9], s[9:12]
        s_rel = torch.tensor(np.concatenate([r2 - r1, v2 - v1]), dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get 12D residuals
        # [dx1, dv1, dx2, dv2]
        ds_g = ens_g(s_rel).squeeze(0).cpu().numpy()
        ds_sp = ens_sp(s_rel).squeeze(0).cpu().numpy()
        
        # APPROACH B: Kinematic Selection + Force Superposition
        # dx comes from a single module (kinematic drift remains consistent)
        # dv comes from the sum of both force-induced updates
        dx1 = ds_g[0:3]
        dv1 = ds_g[3:6] + ds_sp[3:6]
        dx2 = ds_g[6:9]
        dv2 = ds_g[9:12] + ds_sp[9:12]
        
        ds = np.concatenate([dx1, dv1, dx2, dv2])
        s += ds
        traj.append(s.copy())
    return np.array(traj)

def run_monolith(ens_m, s_init, steps=STEPS):
    s = s_init.copy()
    traj = [s.copy()]
    for _ in range(steps):
        r1, v1, r2, v2 = s[0:3], s[3:6], s[6:9], s[9:12]
        s_rel = torch.tensor(np.concatenate([r2 - r1, v2 - v1]), dtype=torch.float32, device=device).unsqueeze(0)
        ds = ens_m(s_rel).squeeze(0).cpu().numpy()
        s += ds
        traj.append(s.copy())
    return np.array(traj)

def energy(s):
    r1, v1, r2, v2 = s[0:3], s[3:6], s[6:9], s[9:12]
    KE = 0.5 * (np.sum(v1**2) + np.sum(v2**2))
    r12 = np.linalg.norm(r2 - r1)
    PE = -G/np.sqrt(r12**2 + EPS**2) + 0.5*K*(r12 - L0)**2
    return KE + PE

if __name__ == "__main__":
    print("=== 12_D ZERO-SHOT: FINAL VELOCITY CORRECTION ===")
    S_g,  Y_g  = make_gravity_data()
    S_sp, Y_sp = make_spring_data()
    S_m,  Y_m  = make_combined_data()

    ens_g  = Ensemble([train(S_g,  Y_g,  label='Gravity', seed=i) for i in range(N_MODELS)]).to(device)
    ens_sp = Ensemble([train(S_sp, Y_sp, label='Spring',  seed=i+10) for i in range(N_MODELS)]).to(device)
    ens_m  = Ensemble([train(S_m,  Y_m,  label='Monolith', seed=i+20) for i in range(N_MODELS)]).to(device)

    s_init = np.array([-1,0,0, 0,0.8,0.2, 1,0,0, 0,-0.8,-0.2], dtype=float)
    traj_gt = [s_init.copy()]
    s_curr = s_init.copy()
    for _ in range(STEPS):
        s_curr = gt_step(s_curr); traj_gt.append(s_curr.copy())
    traj_gt = np.array(traj_gt)
    
    traj_zs = run_zeroshot(ens_g, ens_sp, s_init)
    traj_mo = run_monolith(ens_m, s_init)

    mse_zs = np.mean((traj_zs - traj_gt)**2)
    mse_mo = np.mean((traj_mo - traj_gt)**2)
    
    print(f"\nZero-Shot MSE: {mse_zs:.8f}")
    print(f"Monolith  MSE: {mse_mo:.8f}")
    print(f"Ratio: {mse_mo/mse_zs:.2f}x improvement over monolith" if mse_zs < mse_mo else f"Ratio: {mse_zs/mse_mo:.2f}x monolith is better")

    # Plot & Save
    fig = plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    plt.plot(np.mean((traj_zs - traj_gt)**2, axis=1), color='#00d4ff', label='Zero-Shot (Ours)')
    plt.plot(np.mean((traj_mo - traj_gt)**2, axis=1), color='#ff4444', label='Monolith (Baseline)')
    plt.yscale('log')
    plt.title("12D Phase Space Composition - Scaling Convergence")
    plt.legend()
    # Removed absolute path for notebook compatibility
    save_path = 'd12_scaling_loss.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")