"""
EXPERIMENT V: 2D Neural ODE Baseline
====================================
Evaluates a Fair Structural Comparator on the 2D Orbital Mechanics system.
The Neural ODE learns the vector field ds/dt = f_theta(s) rather than 
discrete states, making it mathematically isomorphic to our residual framing
without assuming independent physical components.

Run in Google Colab. Paste output back.
"""
import torch

if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------
# Environment Constants
# ---------------------------------------------------------
GM = 1.0       # Central Gravity parameter
Cd = 0.05      # Coefficient of Drag
dt = 0.05      # Timestep

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

def collect_data(n=40000, r_bounds=(0.5, 3.0), v_bounds=(-2.0, 2.0)):
    X, Y = [], []
    def reset():
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(*r_bounds)
        x, y = r*np.cos(theta), r*np.sin(theta)
        vx, vy = np.random.uniform(*v_bounds), np.random.uniform(*v_bounds)
        return CombinedOrbit2D(x, y, vx, vy)
        
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

# ---------------------------------------------------------
# Neural ODE Architecture
# ---------------------------------------------------------
class NeuralODE2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_field = nn.Sequential(
            nn.Linear(4, 128),   # Input: [x, y, vx, vy]
            nn.Tanh(),           # Smooth for continuous flow modeling
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4)    # Output: ds/dt
        )
    
    def forward(self, x):
        dsdt = self.vector_field(x)
        return dsdt * dt

def train_neural_ode(X, Y, model, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = nn.MSELoss()(out, by)
            loss.backward()
            optimizer.step()
    return model

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT V: 2D Neural ODE Baseline (10 Seeds)")
    print("=" * 70)
    
    n_seeds = 10
    base_errors_all = []
    
    # Unified test set: circular orbits to test orbital decay prediction
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

    for seed in range(n_seeds):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        print(f"\n--- SEED {seed+1}/{n_seeds} ---")
        X_train, Y_train = collect_data(100000)
        
        model = NeuralODE2D()
        model = train_neural_ode(X_train, Y_train, model, epochs=200)
        
        with torch.no_grad():
            preds = model(X_test_t)
            mse = nn.MSELoss()(preds, Y_test_t).item()
            
        base_errors_all.append(mse)
        print(f"2D Neural ODE (100K) MSE: {mse * 1e4:.4f} x 10^-4")
        
    print("\n" + "=" * 70)
    print("FINAL 2D NEURAL ODE BASELINE RESULTS (10 Seeds)")
    print("=" * 70)
    print(f"Mean MSE: {np.mean(base_errors_all) * 1e4:.4f} \u00B1 {np.std(base_errors_all) * 1e4:.4f} x 10^-4")
    
    # Save the array for statistical comparison (optional)
    np.save("mse_2d_neural_ode.npy", np.array(base_errors_all))

    # ---------------------------------------------------------
    # Generate 2D Trajectory Plot
    # ---------------------------------------------------------
    print("\nGenerating 2D Neural ODE Trajectory Plot...")
    import matplotlib.pyplot as plt
    import os
    
    # Run a 300-step rollout for visualization
    theta = np.random.uniform(0, 2*np.pi)
    r = 1.0
    x, y = r*np.cos(theta), r*np.sin(theta)
    v_mag = np.sqrt(GM/r)
    vx, vy = -v_mag*np.sin(theta), v_mag*np.cos(theta)
    
    sim = CombinedOrbit2D(x, y, vx, vy)
    true_traj, ode_traj = [sim.s.copy()], [sim.s.copy()]
    
    curr_ode = sim.s.copy()
    
    for _ in range(300):
        # Ground Truth
        true_traj.append(sim.step())
        
        # Neural ODE
        inp_ode = torch.tensor(np.array([curr_ode]), dtype=torch.float32)
        with torch.no_grad():
            pred_ds_ode = model(inp_ode).numpy()[0]
        curr_ode = curr_ode + pred_ds_ode
        ode_traj.append(curr_ode.copy())
        
    true_traj = np.array(true_traj)
    ode_traj = np.array(ode_traj)
    
    plt.figure(figsize=(10, 8))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'k-', linewidth=2.5, label='Ground Truth (Gravity + Drag)')
    plt.plot(ode_traj[:, 0], ode_traj[:, 1], 'm--', linewidth=2, label='100K Neural ODE Baseline')
    
    plt.scatter(true_traj[0, 0], true_traj[0, 1], color='green', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(0, 0, color='gold', marker='*', s=300, label='Central Mass', zorder=5)
    
    plt.title("2D Orbital Mechanics: Neural ODE Baseline", fontsize=14)
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.axis('equal')
    
    plt.savefig('2D_Orbital_Neural_ODE.png', dpi=300, bbox_inches='tight')
    print("Plot saved as '2D_Orbital_Neural_ODE.png'")
