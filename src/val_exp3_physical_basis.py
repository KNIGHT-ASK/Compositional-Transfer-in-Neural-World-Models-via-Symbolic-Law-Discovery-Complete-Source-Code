import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Lasso

# ---------------------------------------------------------
# Environment: 3D Inverse-Square Gravity (Tanh Network)
# ---------------------------------------------------------
GM = 1.0
dt = 0.001

class LeapfrogSim:
    def __init__(self, s): self.s = np.array(s, dtype=float)
    def step(self):
        pos, vel = self.s[0:3], self.s[3:6]
        r = np.linalg.norm(pos)
        a = -GM * pos / r**3
        v_half = vel + 0.5 * a * dt
        p_new = pos + v_half * dt
        a_new = -GM * p_new / np.linalg.norm(p_new)**3
        v_new = v_half + 0.5 * a_new * dt
        self.s = np.concatenate([p_new, v_new])
        return self.s.copy()

class Brain3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 6)
        )
    def forward(self, x): return self.net(x)

def collect_data(n=100000):
    X, Y = [], []
    for _ in range(n // 50):
        # Sampling specifically around the curvature
        r = np.exp(np.random.uniform(np.log(0.6), np.log(4.0)))
        phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
        pos = [r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]
        sim = LeapfrogSim(np.concatenate([pos, [0,0,0]]))
        for _ in range(50):
            s_old = sim.s.copy()
            Y.append(sim.step() - s_old); X.append(s_old)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

if __name__ == "__main__":
    print("--- Phase III: Physical Basis Symbolic Discovery ---")
    X, Y = collect_data()
    m = Brain3D()
    opt = optim.Adam(m.parameters(), lr=1e-4) # High precision training
    
    print("Training Tanh Gravity Brain...")
    for i in range(400):
        opt.zero_grad(); nn.MSELoss()(m(X), Y).backward(); opt.step()
        if i % 100 == 0: print(f"  Epoch {i}")
        
    print("\nProbing Network with Physical Basis: [x1/r^3, x2/r^3, x3/r^3]...")
    X_probe, Y_probe = [], []
    # Probe on a dense 3D grid
    for _ in range(1000):
        r = np.random.uniform(0.7, 3.5)
        phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
        x, y, z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
        
        with torch.no_grad():
            dv = m(torch.tensor([[x, y, z, 0, 0, 0]], dtype=torch.float32)).numpy()[0, 3:6]
        
        accel = dv / dt
        # Physical basis: [x/r^3, y/r^3, z/r^3, x, y, z, 1]
        X_probe.append([x/r**3, y/r**3, z/r**3, x, y, z, 1])
        Y_probe.append(accel)
        
    X_p = np.array(X_probe)
    Y_p = np.array(Y_probe)
    
    features = ["x/r^3", "y/r^3", "z/r^3", "x", "y", "z", "1"]
    
    print("-" * 50)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        lasso = Lasso(alpha=0.001)
        lasso.fit(X_p, Y_p[:, i])
        coefs = lasso.coef_
        eq = f"accel_{axis} = "
        for c, f in zip(coefs, features):
            if abs(c) > 0.01: eq += f"{c:.4f}*{f} + "
        print(eq.strip(" + "))
    print("-" * 50)
    
    # Target: accel_X = -1.0 * x/r^3, etc.
    print(f"Target: accel_X = -1.0000*x/r^3")
    print("Result: Check if non-physical terms (x, y, z, 1) are zero.")
