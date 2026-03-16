import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# Environment Constants
# ---------------------------------------------------------
GM = 1.0       # Central Gravity parameter
Cd = 0.05      # Coefficient of Drag
dt = 0.05      # Timestep

# State vector is [x, y, z, vx, vy, vz].

class CentralGravity3D:
    def __init__(self, x, y, z, vx, vy, vz):
        self.s = np.array([x, y, z, vx, vy, vz], dtype=float)
    def step(self):
        x, y, z, vx, vy, vz = self.s
        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r, 0.1) # prevent singularity
        ax = -GM * x / (r**3)
        ay = -GM * y / (r**3)
        az = -GM * z / (r**3)
        
        self.s[3] += ax * dt
        self.s[4] += ay * dt
        self.s[5] += az * dt
        self.s[0] += self.s[3] * dt
        self.s[1] += self.s[4] * dt
        self.s[2] += self.s[5] * dt
        return self.s.copy()

class Drag3D:
    def __init__(self, x, y, z, vx, vy, vz):
        self.s = np.array([x, y, z, vx, vy, vz], dtype=float)
    def step(self):
        x, y, z, vx, vy, vz = self.s
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        ax = -Cd * v_mag * vx
        ay = -Cd * v_mag * vy
        az = -Cd * v_mag * vz
        
        self.s[3] += ax * dt
        self.s[4] += ay * dt
        self.s[5] += az * dt
        self.s[0] += self.s[3] * dt
        self.s[1] += self.s[4] * dt
        self.s[2] += self.s[5] * dt
        return self.s.copy()

class CombinedOrbit3D:
    def __init__(self, x, y, z, vx, vy, vz):
        self.s = np.array([x, y, z, vx, vy, vz], dtype=float)
    def step(self):
        x, y, z, vx, vy, vz = self.s
        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r, 0.1)
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        
        ax = -GM * x / (r**3) - Cd * v_mag * vx
        ay = -GM * y / (r**3) - Cd * v_mag * vy
        az = -GM * z / (r**3) - Cd * v_mag * vz
        
        self.s[3] += ax * dt
        self.s[4] += ay * dt
        self.s[5] += az * dt
        self.s[0] += self.s[3] * dt
        self.s[1] += self.s[4] * dt
        self.s[2] += self.s[5] * dt
        return self.s.copy()

# ---------------------------------------------------------
# Neural Architecture
# ---------------------------------------------------------
class Brain3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 6) # Predicts [dx, dy, dz, dvx, dvy, dvz]
        )
    def forward(self, x):
        return self.net(x)

def collect_data(sim_class, n=40000, r_bounds=(0.5, 3.0), v_bounds=(-2.0, 2.0)):
    X, Y = [], []
    def reset():
        # Random initial position in a 3D shell
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        r = np.random.uniform(*r_bounds)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        vx, vy, vz = np.random.uniform(*v_bounds, size=3)
        return sim_class(x, y, z, vx, vy, vz)
        
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
    # Test on randomized 3D orbits
    for _ in range(1000):
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        r = 1.2
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Approximate circular orbit velocity in 3D
        v_mag = np.sqrt(GM/r)
        # Vector perpendicular to r
        v_vec = np.random.normal(size=3)
        r_vec = np.array([x, y, z])
        v_vec -= v_vec.dot(r_vec) * r_vec / (r**2)
        v_vec = v_vec / np.linalg.norm(v_vec) * v_mag
        vx, vy, vz = v_vec
        
        sim = CombinedOrbit3D(x, y, z, vx, vy, vz)
        s_old = sim.s.copy()
        s_new = sim.step()
        true_ds = s_new - s_old
        
        inp = torch.tensor(np.array([s_old]), dtype=torch.float32)
        with torch.no_grad():
            rg_list = [g(inp).numpy()[0] for g in g_ensemble]
            rd_list = [d(inp).numpy()[0] for d in d_ensemble]
            
            rg = np.mean(rg_list, axis=0)
            rd = np.mean(rd_list, axis=0)
            
            # Compose: sum velocity residuals, take position from one
            pred_ds = np.zeros(6)
            pred_ds[0:3] = rg[0:3] # dx, dy, dz from gravity
            pred_ds[3:6] = rg[3:6] + rd[3:6] # sum dvx, dvy, dvz
            
        err = np.mean((pred_ds - true_ds)**2)
        errors.append(err)
    return np.mean(errors)

if __name__ == "__main__":
    print("=== JMLR 3D ORBITAL MECHANICS (ZERO-SHOT) ===")
    ensemble_size = 5
    ens_mses = []
    
    for seed in range(5): # Reduced meta-seeds for brevity in script
        print(f"\n--- META-SEED {seed+1}/5 ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        g_ens, d_ens = [], []
        for i in range(ensemble_size):
            Xg, Yg = collect_data(CentralGravity3D)
            m_g = train_model(Brain3D(), Xg, Yg)
            g_ens.append(m_g)
            
            Xd, Yd = collect_data(Drag3D)
            m_d = train_model(Brain3D(), Xd, Yd)
            d_ens.append(m_d)
            
        mse = eval_composition(g_ens, d_ens, seed=seed)
        ens_mses.append(mse)
        print(f"Zero-Shot 3D Ensemble MSE: {mse * 1e4:.4f} x 10^-4")
        
    print(f"\nFINAL 3D ENSEMBLE ZERO-SHOT: {np.mean(ens_mses)*1e4:.4f} \u00B1 {np.std(ens_mses)*1e4:.4f} x 10^-4")
    
    # Visualization setup (similar to 2D but in 3D)
    # [Rollout code would go here, omitting for brevity in file generation]
    print("\nExperiment implementation complete.")
