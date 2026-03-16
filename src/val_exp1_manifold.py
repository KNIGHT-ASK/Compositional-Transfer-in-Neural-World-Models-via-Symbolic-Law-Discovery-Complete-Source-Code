import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Environment Constants
# ---------------------------------------------------------
GM = 1.0
dt = 0.05

class CentralGravity3D:
    def __init__(self, x, y, z, vx, vy, vz):
        self.s = np.array([x, y, z, vx, vy, vz], dtype=float)
    def step(self):
        x, y, z, vx, vy, vz = self.s
        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r, 0.1)
        ax, ay, az = -GM*x/r**3, -GM*y/r**3, -GM*z/r**3
        self.s[3:6] += np.array([ax, ay, az]) * dt
        self.s[0:3] += self.s[3:6] * dt
        return self.s.copy()

def get_energy(s):
    x, y, z, vx, vy, vz = s
    r = np.sqrt(x**2 + y**2 + z**2)
    v2 = vx**2 + vy**2 + vz**2
    return v2/2 - GM/r

class Brain3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 6)
        )
    def forward(self, x): return self.net(x)

def collect_data(sim_class, n=60000):
    X, Y = [], []
    for _ in range(n // 50):
        phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
        r = np.random.uniform(0.7, 3.5) # Wider range for better stability
        s = [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta),
             np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        sim = sim_class(*s)
        for _ in range(50):
            s_old = sim.s.copy()
            s_new = sim.step()
            X.append(s_old); Y.append(s_new - s_old)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def train(model, X, Y):
    opt = optim.Adam(model.parameters(), lr=5e-4) # Slower LR for Tanh stability
    for _ in range(250): # More epochs for Tanh convergence
        opt.zero_grad()
        nn.MSELoss()(model(X), Y).backward()
        opt.step()
    return model

if __name__ == "__main__":
    print("--- Phase I: Manifold Persistence Validation (Ensemble V2) ---")
    ensemble_size = 10
    ensemble = []
    
    print(f"Training Ensemble of {ensemble_size} Brains...")
    for i in range(ensemble_size):
        torch.manual_seed(i)
        Xg, Yg = collect_data(CentralGravity3D)
        m = train(Brain3D(), Xg, Yg)
        ensemble.append(m)
        print(f"  - Model {i+1}/{ensemble_size} trained.")
    
    # Initial condition: Circular orbit
    r = 1.2; v_mag = np.sqrt(GM/r)
    s_init = np.array([r, 0, 0, 0, v_mag, 0])
    E_init = get_energy(s_init)
    
    steps = 500
    curr_s = s_init.copy()
    energies = [E_init]
    
    print(f"Starting Rollout (Steps={steps})...")
    for _ in range(steps):
        inp = torch.tensor(np.array([curr_s]), dtype=torch.float32)
        with torch.no_grad():
            preds = [m(inp).numpy()[0] for m in ensemble]
            ds = np.mean(preds, axis=0) # Marginalize error via Ensembling
        curr_s += ds
        energies.append(get_energy(curr_s))
        
    final_drift = abs(energies[-1] - E_init)
    print(f"Initial Energy: {E_init:.6f}")
    print(f"Final Energy: {energies[-1]:.6f}")
    print(f"Energy Drift: {final_drift:.6f}")
    
    if final_drift < 0.1: # Goal for JMLR: Stability over long horizon
        print("RESULT: SUCCESS - Manifold Persistence Achieved via Ensembling.")
    else:
        print("RESULT: WEAK PERSISTENCE - Increase data density or layers.")
    
    plt.figure(figsize=(10,6))
    plt.plot(energies, label='Ensemble Energy')
    plt.axhline(y=E_init, color='r', linestyle='--', label='Theoretical Manifold')
    plt.title("Manifold Persistence in 3D Orbit: Ensemble Stability")
    plt.xlabel("Step"); plt.ylabel("Specific Orbital Energy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("manifold_ensemble_stability.png")
    print("Result: Plot saved to manifold_ensemble_stability.png")
