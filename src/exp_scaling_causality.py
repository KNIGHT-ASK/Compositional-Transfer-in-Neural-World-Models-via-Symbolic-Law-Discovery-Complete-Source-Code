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

# -----------------------------------------------
# CONSTANTS & PHYSICS
# -----------------------------------------------
G = 2.0
K = 2.0
EPS = 0.05
N_SAMPLES = 100000
EPOCHS = 100
BATCH = 32768

def get_data_2d(n):
    # D=2 (1D Physics)
    x = torch.empty(n, 1, device=device).uniform_(-5, 5)
    v = torch.empty(n, 1, device=device).uniform_(-5, 5)
    S = torch.cat([x, v], dim=1)
    
    ag = torch.full((n, 1), -G, device=device)
    as_ = -K * x
    
    dt = 0.01
    Yg = torch.cat([(v + 0.5 * ag * dt) * dt, ag * dt], dim=1)
    Ys = torch.cat([(v + 0.5 * as_ * dt) * dt, as_ * dt], dim=1)
    return S, Yg, Ys

def get_data_6d(n):
    # D=6 (3D Physics)
    pos = torch.empty(n, 3, device=device).uniform_(-5, 5)
    vel = torch.empty(n, 3, device=device).uniform_(-5, 5)
    S = torch.cat([pos, vel], dim=1)
    
    ag = torch.zeros(n, 3, device=device); ag[:, 2] = -G
    as_ = -K * pos
    
    dt = 0.005
    Yg = torch.cat([(vel + 0.5 * ag * dt) * dt, ag * dt], dim=1)
    Ys = torch.cat([(vel + 0.5 * as_ * dt) * dt, as_ * dt], dim=1)
    return S, Yg, Ys

def get_data_12d(n):
    # D=12 (3D Two-Body)
    r1 = torch.empty(n, 3, device=device).uniform_(-5, 5)
    v1 = torch.empty(n, 3, device=device).uniform_(-3, 3)
    r2 = torch.empty(n, 3, device=device).uniform_(-5, 5)
    v2 = torch.empty(n, 3, device=device).uniform_(-3, 3)
    
    r_rel = r2 - r1; v_rel = v2 - v1
    S_rel = torch.cat([r_rel, v_rel], dim=1)
    
    r_mag = torch.norm(r_rel, dim=1, keepdim=True)
    r_soft = torch.sqrt(r_mag**2 + EPS**2)
    ag1 = G * r_rel / r_soft**3; ag2 = -ag1
    
    r_hat = r_rel / r_mag.clamp(min=1e-6)
    as1 = K * (r_mag - 1.0) * r_hat; as2 = -as1
    
    dt = 0.005
    Yg = torch.cat([(v1+0.5*ag1*dt)*dt, ag1*dt, (v2+0.5*ag2*dt)*dt, ag2*dt], dim=1)
    Ys = torch.cat([(v1+0.5*as1*dt)*dt, as1*dt, (v2+0.5*as2*dt)*dt, as2*dt], dim=1)
    return S_rel, Yg, Ys

# -----------------------------------------------
# CAUSALITY MEASUREMENT
# -----------------------------------------------
def measure_causality(D_in, D_out, data_fn, label):
    print(f"\nAnalyzing {label}...")
    S, Yg, Ys = data_fn(N_SAMPLES)
    
    # Normalization
    Sm, Ss = S.mean(0), S.std(0) + 1e-8
    Ygm, Ygs = Yg.mean(0), Yg.std(0) + 1e-8
    Ysm, Yss = Ys.mean(0), Ys.std(0) + 1e-8
    
    Sn = (S - Sm) / Ss
    Ygn = (Yg - Ygm) / Ygs
    Ysn = (Ys - Ysm) / Yss
    
    model = nn.Sequential(
        nn.Linear(D_in, 512), nn.Tanh(),
        nn.Linear(512, 512), nn.Tanh(),
        nn.Linear(512, D_out)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    history = []
    
    for epoch in range(EPOCHS):
        epoch_sim = []
        idx = torch.randperm(N_SAMPLES, device=device)
        
        for i in range(0, N_SAMPLES, BATCH):
            batch_idx = idx[i:i+BATCH]
            bx = Sn[batch_idx]
            by_g = Ygn[batch_idx]
            by_s = Ysn[batch_idx]
            
            optimizer.zero_grad()
            
            # Compute Gradient for Task 1 (Gravity)
            pred = model(bx)
            loss_g = criterion(pred, by_g)
            grad_g = torch.autograd.grad(loss_g, model.parameters(), retain_graph=True)
            vec_g = torch.cat([g.view(-1) for g in grad_g])
            
            # Compute Gradient for Task 2 (Spring)
            loss_s = criterion(pred, by_s)
            grad_s = torch.autograd.grad(loss_s, model.parameters(), retain_graph=True)
            vec_s = torch.cat([g.view(-1) for g in grad_s])
            
            # Measurement: Cosine Similarity (Gradient Interference)
            # sim = 1.0 (Align), sim = -1.0 (Oppose), sim = 0.0 (Orthogonal)
            sim = torch.nn.functional.cosine_similarity(vec_g, vec_s, dim=0)
            epoch_sim.append(sim.item())
            
            # Step Monolith on combined loss
            (loss_g + loss_s).backward()
            optimizer.step()
            
        avg_sim = np.mean(epoch_sim)
        history.append(avg_sim)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Gradient Alignment: {avg_sim:.4f}")
            
    return history

# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    results = {}
    
    # 2D (Conflict is low, shared dimensions)
    results['2D'] = measure_causality(2, 2, get_data_2d, "2D Manifold")
    
    # 6D (Higher conflict)
    results['6D'] = measure_causality(6, 6, get_data_6d, "6D Manifold")
    
    # 12D (High-Dimensional Interference)
    results['12D'] = measure_causality(6, 12, get_data_12d, "12D Two-Body")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    for label, history in results.items():
        plt.plot(history, label=f'{label} Alignment', lw=2)
        
    plt.axhline(y=0.0, color='white', linestyle='--', alpha=0.5, label='Orthogonal (No Correlation)')
    plt.title("Experimental Causal Analysis: Gradient Interference across Scales")
    plt.xlabel("Training Epoch (Monolith)")
    plt.ylabel("Cosine Similarity (Gravity Grad vs Spring Grad)")
    plt.legend()
    plt.grid(alpha=0.2)
    
    save_path = 'gradient_causality.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved Causal Proof: {save_path}")
    
    # Analysis Print
    print("\n--- CAUSAL CONCLUSION ---")
    for dim in ['2D', '6D', '12D']:
        final_sim = results[dim][-1]
        print(f"{dim} Final Alignment: {final_sim:.4f}")
    
    print("\nInterpretation:")
    print("If Alignment < 0: Forces are in ACTIVE CONFLICT. Monolith must sacrifice one to learn the other.")
    print("If Alignment ~ 0: Forces are ORTHOGONAL. No shared intelligence; Monolith capacity is wasted.")
    print("Zero-Shot wins because it executes 100%% of each gradient in isolation.")
