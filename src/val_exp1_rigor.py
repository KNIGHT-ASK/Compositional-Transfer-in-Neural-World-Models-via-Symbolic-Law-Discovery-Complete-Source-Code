import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Force GPU
assert torch.cuda.is_available(), "No GPU!"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

GM = 1.0
dt = 0.001
epsilon = 0.001
N_MODELS = 10

# -----------------------------------------------
# Generate and KEEP data on GPU permanently
# -----------------------------------------------
def make_gpu_data(n=500000):
    print(f"Making {n:,} samples on GPU...")
    t0 = time.time()
    torch.manual_seed(42)
    
    with torch.no_grad():
        r = torch.empty(n, device=device).uniform_(0.5, 2.0)
        angle = torch.empty(n, device=device).uniform_(0, 2*np.pi)
        
        x = r * torch.cos(angle)
        y = r * torch.sin(angle)
        z = torch.empty(n, device=device).uniform_(-0.1, 0.1)
        
        v_circ = torch.sqrt(
            torch.tensor(GM, device=device) / r
        )
        vx = -v_circ * torch.sin(angle)
        vy =  v_circ * torch.cos(angle)
        vz = torch.empty(n, device=device).uniform_(-0.05, 0.05)
        
        # States [N, 6] — STAYS ON GPU
        S = torch.stack([x,y,z,vx,vy,vz], dim=1)
        S += torch.randn_like(S) * 0.03
        
        # True acceleration [N, 3] — STAYS ON GPU
        pos = S[:, 0:3]
        r_n = torch.norm(pos, dim=1, keepdim=True)
        r_s = torch.sqrt(r_n**2 + epsilon**2)
        A = -GM * pos / (r_s**3)
    
    mem = torch.cuda.memory_allocated()/1e9
    print(f"Done: {time.time()-t0:.1f}s | "
          f"VRAM used: {mem:.3f}GB")
    print(f"S shape: {S.shape} on {S.device}")
    print(f"A shape: {A.shape} on {A.device}")
    return S, A

# -----------------------------------------------
# Normalize ON GPU
# -----------------------------------------------
def normalize_gpu(S, A):
    Sm = S.mean(0)
    Ss = S.std(0) + 1e-8
    Am = A.mean(0)
    As = A.std(0) + 1e-8
    Sn = (S - Sm) / Ss
    An = (A - Am) / As
    return Sn, An, Sm, Ss, Am, As

# -----------------------------------------------
# Small Fast Architecture
# -----------------------------------------------
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# Pure GPU Training — No DataLoader
# DataLoader causes CPU bottleneck
# Direct GPU tensor slicing instead
# -----------------------------------------------
def train_gpu(Sn, An, seed=0, epochs=80):
    torch.manual_seed(seed)
    model = Brain().to(device)
    
    # Verify on GPU
    assert next(model.parameters()).is_cuda, \
        "Model not on GPU!"
    
    n = Sn.shape[0]
    batch = 65536
    
    opt = optim.Adam(
        model.parameters(), lr=5e-3
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs
    )
    scaler = torch.amp.GradScaler('cuda')
    
    t0 = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs+1):
        # Shuffle indices ON GPU
        idx = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        # Manual batching — pure GPU
        for start in range(0, n, batch):
            end = min(start + batch, n)
            bx = Sn[idx[start:end]]
            by = An[idx[start:end]]
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        sch.step()
        avg_loss = epoch_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 20 == 0:
            mem = torch.cuda.memory_allocated()/1e9
            util = torch.cuda.utilization()
            print(
                f"  Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.6f} | "
                f"Time: {time.time()-t0:.1f}s | "
                f"VRAM: {mem:.2f}GB | "
                f"GPU: {util}%"
            )
    
    model.eval()
    return model

# -----------------------------------------------
# Train All Models
# -----------------------------------------------
def train_ensemble(S, A):
    print("\nNormalizing on GPU...")
    Sn, An, Sm, Ss, Am, As = normalize_gpu(S, A)
    norms = (Sm.cpu(), Ss.cpu(), 
             Am.cpu(), As.cpu())
    
    models = []
    print(f"\nTraining {N_MODELS} brains "
          f"(pure GPU)...")
    
    t0 = time.time()
    for i in range(N_MODELS):
        print(f"\nBrain {i+1}/{N_MODELS}:")
        m = train_gpu(Sn, An, seed=i)
        models.append(m.cpu())
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        remaining = elapsed/(i+1)*(N_MODELS-i-1)
        print(f"  Done. ETA: {remaining:.0f}s")
    
    print(f"\nAll done: {time.time()-t0:.1f}s")
    return models, norms

# -----------------------------------------------
# Ensemble Prediction On GPU
# -----------------------------------------------
class Ensemble(nn.Module):
    def __init__(self, models, norms):
        super().__init__()
        self.models = nn.ModuleList(models)
        Sm, Ss, Am, As = norms
        self.register_buffer('Sm', Sm)
        self.register_buffer('Ss', Ss)
        self.register_buffer('Am', Am)
        self.register_buffer('As', As)
    
    @torch.no_grad()
    def predict_accel(self, s):
        sn = (s - self.Sm) / self.Ss
        preds = torch.stack([
            m(sn) for m in self.models
        ])
        an = preds.mean(0)
        return an * self.As + self.Am

# -----------------------------------------------
# Metrics
# -----------------------------------------------
def metrics(s):
    if s.dim() == 1:
        s = s.unsqueeze(0)
    pos = s[:, 0:3]
    vel = s[:, 3:6]
    r = torch.norm(pos, dim=1)
    E = 0.5*torch.sum(vel**2,dim=1) - GM/r
    L = torch.norm(
        torch.linalg.cross(pos, vel), dim=1
    )
    return r.squeeze(), E.squeeze(), L.squeeze()

# -----------------------------------------------
# Symplectic Rollout On GPU
# -----------------------------------------------
@torch.no_grad()
def rollout(ens, s0, steps=10000):
    ens.to(device)
    s = torch.tensor(
        s0, dtype=torch.float32, device=device
    )
    
    all_s = torch.zeros(
        steps+1, 6, device=device
    )
    all_s[0] = s
    
    t0 = time.time()
    for i in range(steps):
        # Leapfrog step 1
        a1 = ens.predict_accel(s)
        vh = s[3:6] + 0.5*a1*dt
        pn = s[0:3] + vh*dt
        
        # Leapfrog step 2
        sm = torch.cat([pn, vh])
        a2 = ens.predict_accel(sm)
        vn = vh + 0.5*a2*dt
        
        s = torch.cat([pn, vn])
        all_s[i+1] = s
        
        if i % 2000 == 0 and i > 0:
            sp = i/(time.time()-t0)
            _, e, _ = metrics(s)
            print(f"  Step {i:5d} | "
                  f"E={e:.5f} | "
                  f"{sp:.0f} sps")
    
    t = time.time()-t0
    print(f"Rollout: {t:.1f}s "
          f"@ {steps/t:.0f} steps/s")
    return all_s.cpu()

# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    
    # Step 1 — GPU data
    S, A = make_gpu_data(500000)
    
    # Step 2 — Train
    models, norms = train_ensemble(S, A)
    
    # Step 3 — Ensemble
    ens = Ensemble(models, norms)
    
    # Step 4 — Initial orbit
    r0 = 1.0
    s0 = np.array([
        r0, 0.0, 0.0,
        0.0, np.sqrt(GM/r0), 0.0
    ])
    _, e0, l0 = metrics(
        torch.tensor(s0, dtype=torch.float32)
    )
    e0, l0 = e0.item(), l0.item()
    print(f"\nInitial: E={e0:.6f} L={l0:.6f}")
    
    # Step 5 — Rollout
    print("\n10,000-step rollout...")
    all_s = rollout(ens, s0, 10000)
    
    # Step 6 — Results
    _, e_arr, l_arr = metrics(all_s)
    r_arr, _, _ = metrics(all_s)
    
    energies = e_arr.numpy()
    momenta = l_arr.numpy()
    radii = r_arr.numpy()
    
    e_d = abs(energies[-1]-e0)
    l_d = abs(momenta[-1]-l0)
    
    print(f"\n{'='*45}")
    print(f"Energy Drift:   {e_d:.8f}")
    print(f"Momentum Drift: {l_d:.8f}")
    print(f"Total Time: "
          f"{time.time()-t_all:.1f}s")
    
    if e_d < 0.001:
        v = "EXCELLENT"
    elif e_d < 0.01:
        v = "SUCCESS"
    elif e_d < 0.05:
        v = "PARTIAL"
    else:
        v = "NEEDS WORK"
    print(f"RESULT: {v}")
    print(f"{'='*45}")
    
    # Plot
    fig, axs = plt.subplots(
        3, 1, figsize=(12,15)
    )
    sx = np.arange(len(energies))
    
    for ax, data, true, name in zip(
        axs,
        [radii, energies, momenta],
        [r0, e0, l0],
        ['Radius','Energy','Momentum']
    ):
        ax.plot(sx, data,
                color='royalblue',
                linewidth=0.7,
                label=f'Neural {name}')
        ax.axhline(true,
                   color='red',
                   linestyle='--',
                   linewidth=2,
                   label=f'True={true:.4f}')
        ax.set_title(
            f"{name} — 10,000 Steps"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axs[-1].set_xlabel("Step")
    plt.suptitle(
        f"P100 Manifold Persistence | "
        f"E_drift={e_d:.6f}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("manifold_final.png",
                dpi=150,
                bbox_inches='tight')
    print("Saved: manifold_final.png")
