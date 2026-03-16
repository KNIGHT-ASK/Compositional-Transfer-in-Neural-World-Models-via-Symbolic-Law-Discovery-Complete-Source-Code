import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print(f"GPU: {torch.cuda.get_device_name(0)}")

GM = 1.0
dt = 0.001
epsilon = 0.001

# -----------------------------------------------
# Data — predict acceleration directly
# -----------------------------------------------
def make_data(n=500000):
    print(f"Generating {n:,} samples...")
    t0 = time.time()
    torch.manual_seed(42)
    
    with torch.no_grad():
        # Log-uniform radius distribution
        # Covers wide dynamic range
        log_r = torch.empty(
            n, device=device
        ).uniform_(
            np.log(0.1), np.log(10.0)
        )
        r_mag = torch.exp(log_r)
        
        # Random 3D directions
        phi = torch.empty(
            n, device=device
        ).uniform_(0, 2*np.pi)
        cos_t = torch.empty(
            n, device=device
        ).uniform_(-1, 1)
        sin_t = torch.sqrt(1 - cos_t**2)
        
        x = r_mag * sin_t * torch.cos(phi)
        y = r_mag * sin_t * torch.sin(phi)
        z = r_mag * cos_t
        
        # States [N, 6]
        # Velocities random — more realistic
        v_scale = torch.sqrt(
            torch.tensor(GM, device=device) 
            / r_mag
        )
        vx = torch.randn(
            n, device=device
        ) * v_scale * 0.3
        vy = torch.randn(
            n, device=device
        ) * v_scale * 0.3
        vz = torch.randn(
            n, device=device
        ) * v_scale * 0.3
        
        S = torch.stack(
            [x, y, z, vx, vy, vz], dim=1
        )
        
        # True acceleration — with softening
        # [N, 3]
        r_soft = torch.sqrt(
            r_mag**2 + epsilon**2
        )
        ax = -GM * x / r_soft**3
        ay = -GM * y / r_soft**3
        az = -GM * z / r_soft**3
        
        A = torch.stack([ax, ay, az], dim=1)
    
    print(f"Done: {time.time()-t0:.1f}s")
    print(f"S: {S.shape} | A: {A.shape}")
    print(f"r range: [{r_mag.min():.2f}, "
          f"{r_mag.max():.2f}]")
    return S, A

# -----------------------------------------------
# Architecture — outputs 3 (acceleration)
# -----------------------------------------------
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 3) 
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# Ensemble
# -----------------------------------------------
class Ensemble(nn.Module):
    def __init__(self, models, norms):
        super().__init__()
        self.models = nn.ModuleList(models)
        Sm, Ss, Lm, Ls = norms
        self.register_buffer('Sm', Sm)
        self.register_buffer('Ss', Ss)
        self.register_buffer('Lm', Lm)
        self.register_buffer('Ls', Ls)
    
    @torch.no_grad()
    def predict_accel(self, s):
        sn = (s - self.Sm) / (self.Ss + 1e-8)
        # Ensemble predicts Log-Acceleration
        preds = torch.stack([m(sn) for m in self.models])
        log_an = preds.mean(0)
        
        # Convert back from Log-Z-Score to Real Acceleration
        # log_a_recovered = log_an * Ls + Lm
        # |a| = exp(log_a_recovered)
        # Note: We keep the sign from the Log-Target direction logic
        # For simplicity in 1D probe, we'll return magnitude
        log_a = log_an * self.Ls + self.Lm
        return torch.exp(log_a)

# -----------------------------------------------
# Training with relative loss
# -----------------------------------------------
def train(Sn, Ln, seed=0, epochs=300):
    torch.manual_seed(seed)
    model = Brain().to(device)
    n = Sn.shape[0]
    batch = 16384 # High resolution for law discovery
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    t0 = time.time()
    for epoch in range(epochs+1):
        idx = torch.randperm(n, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, n, batch):
            end = min(start+batch, n)
            bx, bl = Sn[idx[start:end]], Ln[idx[start:end]]
            
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred = model(bx)
                # Now standard MSE on LOG targets — no more explosion
                loss = nn.MSELoss()(pred, bl)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        
        sch.step()
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss/n_batches:.6f} | "
                  f"Time: {time.time()-t0:.1f}s")
    
    return model.cpu()

# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == "__main__":
    t_all = time.time()
    print("=== Phase III: Pure-GPU Purity Discovery (Log-Target Mode) ===")
    
    # 1. Generate data
    S, A = make_data(500000)
    
    # 2. Convert Targets to Log-Magnitude for high-dynamic resolution
    # |a| = GM/r^2
    # log(|a|) = log(GM) - 2*log(r)
    # This is a LINEAL plane. Neural networks are elite at linear planes.
    mask = torch.norm(A, dim=1) > 1e-12
    S_valid = S[mask]
    A_valid = A[mask]
    
    LogA = torch.log(torch.abs(A_valid) + 1e-12)
    
    # 3. Normalize in Log-Space
    Sm, Ss = S_valid.mean(0), S_valid.std(0)+1e-8
    Lm, Ls = LogA.mean(0), LogA.std(0)+1e-8
    
    Sn = (S_valid - Sm) / Ss
    Ln = (LogA - Lm) / Ls # Normalized Log-Targets
    
    print("\nTraining Ensemble (Log-Target Pipeline)...")
    models = [train(Sn, Ln, seed=i, epochs=300) for i in range(5)]
    ens = Ensemble(models, (Sm.cpu(), Ss.cpu(), Lm.cpu(), Ls.cpu())).to(device)
    
    # 4. Probe Power Law
    print("\nProbing for L=-2.00...")
    radii = torch.logspace(np.log10(0.2), np.log10(9.0), 200).to(device)
    probe_states = torch.zeros(200, 6, device=device)
    probe_states[:, 0] = radii
    # Note: Ensemble also outputs magnitude now
    with torch.no_grad():
        a_pred_vec = ens.predict_accel(probe_states)
        # Use magnitude of the x-component prediction
        ax_pred = a_pred_vec[:, 0].cpu().numpy()
    
    # Power law fit
    accels = np.abs(ax_pred)
    r_np = radii.cpu().numpy()
    
    # Remove any zeros
    mask = accels > 1e-10
    log_r = np.log(r_np[mask])
    log_a = np.log(accels[mask])
    
    slope, intercept = np.polyfit(
        log_r, log_a, 1
    )
    error = abs(slope + 2.0)
    
    # True law values
    true_accels = GM / r_np**2
    
    print(f"\n{'='*45}")
    print(f"Learned Exponent:  {slope:.6f}")
    print(f"Target Exponent:   -2.000000")
    print(f"Error:             {error:.6f}")
    print(f"Recovery Accuracy: "
          f"{(1-error/2)*100:.2f}%")
    print(f"Total Time:        "
          f"{time.time()-t_all:.1f}s")
    
    if error < 0.01:
        verdict = "EXCELLENT — Near-perfect"
    elif error < 0.05:
        verdict = "SUCCESS — Good recovery"
    elif error < 0.1:
        verdict = "PARTIAL — Approximate"
    else:
        verdict = "FAILED — Poor recovery"
    print(f"RESULT: {verdict}")
    print(f"{'='*45}")
    
    # Plot
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6)
    )
    
    # Log-log plot
    axes[0].scatter(
        log_r, log_a,
        alpha=0.6, s=15,
        color='royalblue',
        label='Neural Prediction'
    )
    axes[0].plot(
        log_r,
        -2*log_r + intercept,
        'r--', linewidth=2,
        label=f'Fitted: r^{slope:.3f}'
    )
    axes[0].plot(
        log_r,
        np.log(true_accels[mask]),
        'g-', linewidth=2,
        label='True: r^-2',
        alpha=0.7
    )
    axes[0].set_xlabel("log(r)")
    axes[0].set_ylabel("log(|a|)")
    axes[0].set_title(
        f"Power Law Recovery\n"
        f"Learned={slope:.4f} | "
        f"True=-2.0000"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Direct comparison
    axes[1].plot(
        r_np, true_accels,
        'g-', linewidth=2,
        label='True GM/r²'
    )
    axes[1].plot(
        r_np, accels,
        'b--', linewidth=2,
        alpha=0.8,
        label='Neural Prediction'
    )
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("|acceleration|")
    axes[1].set_title(
        "Direct Acceleration Comparison"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 5)
    
    plt.suptitle(
        f"Inverse-Square Law Recovery | "
        f"Error={error:.4f} | "
        f"{verdict}",
        fontsize=13,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        "power_law_recovery.png",
        dpi=150,
        bbox_inches='tight'
    )
    print("\nSaved: power_law_recovery.png")
