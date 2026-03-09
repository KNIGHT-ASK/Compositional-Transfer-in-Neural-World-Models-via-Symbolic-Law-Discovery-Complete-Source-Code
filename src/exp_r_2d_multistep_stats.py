import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# --- PHYSICS ---
class Orbit2D:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.G = 100.0  # Gravitational constant scale
        self.M = 1.0    # Central mass
        self.k_drag = 0.05
        
    def get_accel(self, state):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Gravity
        ax_g = -self.G * self.M * x / (r**3)
        ay_g = -self.G * self.M * y / (r**3)
        
        # Drag
        ax_d = -self.k_drag * v_mag * vx
        ay_d = -self.k_drag * v_mag * vy
        
        return np.array([ax_g + ax_d, ay_g + ay_d])

    def step(self, state):
        x, y, vx, vy = state
        accel = self.get_accel(state)
        # First-order Euler
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        vx_new = vx + accel[0] * self.dt
        vy_new = vy + accel[1] * self.dt
        return np.array([x_new, y_new, vx_new, vy_new])

# --- MODELS ---
class ResidualMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4)
        )
    def forward(self, x): return self.net(x)

def train_component(env_type="gravity", seeds=5):
    # env_type: 'gravity' or 'drag'
    dt = 0.01
    ensemble = []
    for s in range(seeds):
        torch.manual_seed(s)
        model = ResidualMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Data Generator
        # Gravity: Central force. Drag: v^2 force.
        # (Simulation is just for training data collection - shuffled)
        X = np.random.uniform(-5, 5, (50000, 4))
        # Zero out velocity for gravity training or position for drag if needed, 
        # but here we just use the true isolated forces.
        G, K_DRAG = 100.0, 0.05
        
        targets = []
        for x_in in X:
            r = np.sqrt(x_in[0]**2 + x_in[1]**2)
            v_mag = np.sqrt(x_in[2]**2 + x_in[3]**2)
            if env_type == "gravity":
                ax = -G * x_in[0] / (r**3)
                ay = -G * x_in[1] / (r**3)
            else:
                ax = -K_DRAG * v_mag * x_in[2]
                ay = -K_DRAG * v_mag * x_in[3]
            targets.append([x_in[2]*dt, x_in[3]*dt, ax*dt, ay*dt])
            
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(targets))
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        
        for e in range(10): # Fast training
            for bx, by in loader:
                optimizer.zero_grad(); nn.MSELoss()(model(bx), by).backward(); optimizer.step()
        ensemble.append(model)
    return ensemble

# --- EVALUATION ---
def rollout_eval(ensemble_g, ensemble_d, steps=300):
    env = Orbit2D()
    # Initial state: stable orbit with decay
    state = np.array([2.0, 0.0, 0.0, 7.0]) 
    
    gt_traj = [state]
    ens_traj = [state]
    
    curr_gt = state.copy()
    curr_ens = state.copy()
    
    for _ in range(steps):
        # Ground Truth
        curr_gt = env.step(curr_gt)
        gt_traj.append(curr_gt)
        
        # Ensemble Prediction
        with torch.no_grad():
            s_t = torch.FloatTensor(curr_ens.reshape(1, -1))
            # Average residuals
            rg = torch.stack([m(s_t) for m in ensemble_g]).mean(0).numpy()[0]
            rd = torch.stack([m(s_t) for m in ensemble_d]).mean(0).numpy()[0]
            
            # Composition (Sum - Drift)
            ds = rg + rd - np.array([curr_ens[2]*0.01, curr_ens[3]*0.01, 0, 0])
            curr_ens = curr_ens + ds
            ens_traj.append(curr_ens)
            
    mse = np.mean((np.array(gt_traj) - np.array(ens_traj))**2)
    return mse

if __name__ == "__main__":
    print("Training 2D Ensembles (10 Meta-Seeds)...")
    results = []
    for meta_seed in range(10):
        print(f"Meta-Seed {meta_seed+1}/10...")
        ens_g = train_component("gravity", seeds=5)
        ens_d = train_component("drag", seeds=5)
        mse = rollout_eval(ens_g, ens_d)
        results.append(mse)
        
    print("\n--- JMLR 2D MULTI-STEP RIGOR ---")
    print(f"Mean Rollout MSE (300 steps): {np.mean(results):.4e} +- {np.std(results):.4e}")
    # Note: Comparison against 100K baseline would go here in full run.
    # For now, this provides the missing 'Zero-Shot Ensemble' Multi-step stat.
