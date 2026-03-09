import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# A trick to run in environments missing _utils
if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

# ---------------------------------------------------------
# Environment Constants
# ---------------------------------------------------------
GM = 1.0       # Central Gravity parameter
Cd = 0.05      # Coefficient of Drag
dt = 0.05      # Timestep

# State vector is [x, y, vx, vy]. We drop external F to focus purely on the non-linear state mapping.

class CentralGravity2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        r = np.sqrt(x**2 + y**2)
        r = max(r, 0.1) # prevent singularity
        ax = -GM * x / (r**3)
        ay = -GM * y / (r**3)
        self.s[2] += ax * dt
        self.s[3] += ay * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

class Drag2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        v = np.sqrt(vx**2 + vy**2)
        ax = -Cd * v * vx
        ay = -Cd * v * vy
        self.s[2] += ax * dt
        self.s[3] += ay * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

class CombinedOrbit2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        r = np.sqrt(x**2 + y**2)
        r = max(r, 0.1)
        v = np.sqrt(vx**2 + vy**2)
        
        ax = -GM * x / (r**3) - Cd * v * vx
        ay = -GM * y / (r**3) - Cd * v * vy
        
        self.s[2] += ax * dt
        self.s[3] += ay * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

# ---------------------------------------------------------
# Neural Architecture
# ---------------------------------------------------------
class Brain2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4) # Predicts [dx, dy, dvx, dvy]
        )
    def forward(self, x):
        return self.net(x)

def collect_data(sim_class, n=40000, r_bounds=(0.5, 3.0), v_bounds=(-2.0, 2.0)):
    X, Y = [], []
    def reset():
        # Random initial position in a ring
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(*r_bounds)
        x, y = r*np.cos(theta), r*np.sin(theta)
        vx, vy = np.random.uniform(*v_bounds), np.random.uniform(*v_bounds)
        return sim_class(x, y, vx, vy)
        
    sim = reset()
    for i in range(n):
        if i % 50 == 0:
            sim = reset()
            
        s_old = sim.s.copy()
        s_new = sim.step()
        ds = s_new - s_old
        
        X.append(s_old)
        Y.append(ds)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def train_model(model, X, Y, epochs=150, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def eval_composition(g_ensemble, d_ensemble, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    errors = []
    # Test on exactly Circular Orbit velocities to see if it decays properly
    for _ in range(2000):
        theta = np.random.uniform(0, 2*np.pi)
        r = 1.0
        x, y = r*np.cos(theta), r*np.sin(theta)
        # Circular orbit velocity is sqrt(GM/r) tangent to radius
        v_mag = np.sqrt(GM/r)
        vx, vy = -v_mag*np.sin(theta), v_mag*np.cos(theta)
        
        sim = CombinedOrbit2D(x, y, vx, vy)
        s_old = sim.s.copy()
        s_new = sim.step()
        true_ds = s_new - s_old
        
        inp = torch.tensor(np.array([s_old]), dtype=torch.float32)
        with torch.no_grad():
            rg_list = [g(inp).numpy()[0] for g in g_ensemble]
            rd_list = [d(inp).numpy()[0] for d in d_ensemble]
            
            rg = np.mean(rg_list, axis=0)
            rd = np.mean(rd_list, axis=0)
            
            # Compose: sum velocities, take position update from one (since dx=vx*dt is invariant)
            pred_ds = np.zeros(4)
            pred_ds[0] = rg[0]  # dx from gravity
            pred_ds[1] = rg[1]  # dy from gravity
            pred_ds[2] = rg[2] + rd[2] # sum dvx
            pred_ds[3] = rg[3] + rd[3] # sum dvy
            
        err = np.mean((pred_ds - true_ds)**2)
        errors.append(err)
    return np.mean(errors)

if __name__ == "__main__":
    print("=== JMLR 2D ORBITAL MECHANICS (NON-LINEAR) ===")
    ensemble_size = 5
    ens_mses = []
    
    # Running 10 Meta-Seeds for strict JMLR statistical rigor
    for seed in range(10):
        print(f"\n--- META-SEED {seed+1}/10 ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        g_ens, d_ens = [], []
        for i in range(ensemble_size):
            Xg, Yg = collect_data(CentralGravity2D)
            m_g = train_model(Brain2D(), Xg, Yg)
            g_ens.append(m_g)
            
            Xd, Yd = collect_data(Drag2D)
            m_d = train_model(Brain2D(), Xd, Yd)
            d_ens.append(m_d)
            
        mse = eval_composition(g_ens, d_ens, seed=seed)
        ens_mses.append(mse)
        print(f"Zero-Shot 2D Ensemble MSE: {mse * 1e4:.4f} x 10^-4")
        
    print(f"\nFINAL 2D ENSEMBLE ZERO-SHOT: {np.mean(ens_mses)*1e4:.4f} ± {np.std(ens_mses)*1e4:.4f} x 10^-4")
    
    # ---------------------------------------------------------
    # Now run Monolithic 100K Target Baseline for Comparison
    # ---------------------------------------------------------
    print("\n--- Training 100K 2D Baseline (10 Seeds) ---")
    base_errors_all = []
    
    # Generate unified 2D Test Set
    np.random.seed(999)
    X_test, Y_test = [], []
    for _ in range(2000):
        theta = np.random.uniform(0, 2*np.pi)
        r = 1.0
        x, y = r*np.cos(theta), r*np.sin(theta)
        v_mag = np.sqrt(GM/r)
        vx, vy = -v_mag*np.sin(theta), v_mag*np.cos(theta)
        
        sim = CombinedOrbit2D(x, y, vx, vy)
        s_old = sim.s.copy()
        true_ds = sim.step() - s_old
        
        X_test.append(s_old)
        Y_test.append(true_ds)
        
    X_test_t = torch.tensor(np.array(X_test), dtype=torch.float32)
    Y_test_t = torch.tensor(np.array(Y_test), dtype=torch.float32)
    
    for seed in range(10):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        print(f"\n--- BASELINE SEED {seed+1}/10 ---")
        
        X_base, Y_base = collect_data(CombinedOrbit2D, n=100000)
        baseline = train_model(Brain2D(), X_base, Y_base, epochs=200)
        
        with torch.no_grad():
            preds = baseline(X_test_t)
            mse = nn.MSELoss()(preds, Y_test_t).item()
            
        base_errors_all.append(mse)
        print(f"Monolithic 2D Baseline (100K) Seed {seed+1}: {mse*1e4:.4f} x 10^-4")
        
    print("\n" + "=" * 70)
    print("FINAL 2D BASELINE RESULTS (10 Seeds)")
    print("=" * 70)
    print(f"FINAL 2D BASELINE (100K): {np.mean(base_errors_all)*1e4:.4f} \u00B1 {np.std(base_errors_all)*1e4:.4f} x 10^-4")

    # ---------------------------------------------------------
    # Generate 2D Trajectory Plot for Paper
    # ---------------------------------------------------------
    print("\nGenerating 2D Orbital Trajectory Plot...")
    import matplotlib.pyplot as plt
    import os
    
    # Run a 200-step rollout for visualization
    theta = np.random.uniform(0, 2*np.pi)
    r = 1.0
    x, y = r*np.cos(theta), r*np.sin(theta)
    v_mag = np.sqrt(GM/r)
    vx, vy = -v_mag*np.sin(theta), v_mag*np.cos(theta)
    
    sim = CombinedOrbit2D(x, y, vx, vy)
    true_traj, ens_traj, base_traj = [sim.s.copy()], [sim.s.copy()], [sim.s.copy()]
    
    curr_ens = sim.s.copy()
    curr_base = sim.s.copy()
    
    for _ in range(300):
        # Ground Truth Step
        true_traj.append(sim.step())
        
        # Ensemble Zero-Shot Step
        inp_ens = torch.tensor(np.array([curr_ens]), dtype=torch.float32)
        with torch.no_grad():
            rg_list = [g(inp_ens).numpy()[0] for g in g_ens]
            rd_list = [d(inp_ens).numpy()[0] for d in d_ens]
            rg, rd = np.mean(rg_list, axis=0), np.mean(rd_list, axis=0)
            
        pred_ds_ens = np.zeros(4)
        pred_ds_ens[0] = rg[0]
        pred_ds_ens[1] = rg[1]
        pred_ds_ens[2] = rg[2] + rd[2]
        pred_ds_ens[3] = rg[3] + rd[3]
        curr_ens = curr_ens + pred_ds_ens
        ens_traj.append(curr_ens.copy())
        
        # Monolithic Baseline Step
        inp_base = torch.tensor(np.array([curr_base]), dtype=torch.float32)
        with torch.no_grad():
            pred_ds_base = baseline(inp_base).numpy()[0]
        curr_base = curr_base + pred_ds_base
        base_traj.append(curr_base.copy())
        
    true_traj = np.array(true_traj)
    ens_traj = np.array(ens_traj)
    base_traj = np.array(base_traj)
    
    plt.figure(figsize=(10, 8))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=2.5, label='Ground Truth (Gravity + Drag)')
    plt.plot(ens_traj[:, 0], ens_traj[:, 1], 'b--', linewidth=2, label='Zero-Shot Ensemble')
    plt.plot(base_traj[:, 0], base_traj[:, 1], 'r:', linewidth=2, label='100K Monolithic Baseline')
    
    plt.scatter(true_traj[0, 0], true_traj[0, 1], color='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(0, 0, color='gold', marker='*', s=300, label='Central Mass', zorder=5)
    
    plt.title("2D Orbital Mechanics: Zero-Shot Composition vs Monolithic Baseline", fontsize=14)
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axis('equal')
    
    # Save the plot
    plt.savefig('2D_Orbital_Trajectory.png', dpi=300, bbox_inches='tight')
    print("Plot saved as '2D_Orbital_Trajectory.png'")
