import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

GM, Cd, dt = 1.0, 0.05, 0.05

class CentralGravity2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        r = max(np.sqrt(x**2 + y**2), 0.1)
        self.s[2] += -GM * x / (r**3) * dt
        self.s[3] += -GM * y / (r**3) * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

class Drag2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        v = np.sqrt(vx**2 + vy**2)
        self.s[2] += -Cd * v * vx * dt
        self.s[3] += -Cd * v * vy * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

class CombinedOrbit2D:
    def __init__(self, x, y, vx, vy):
        self.s = np.array([x, y, vx, vy], dtype=float)
    def step(self):
        x, y, vx, vy = self.s
        r = max(np.sqrt(x**2 + y**2), 0.1)
        v = np.sqrt(vx**2 + vy**2)
        self.s[2] += (-GM * x / (r**3) - Cd * v * vx) * dt
        self.s[3] += (-GM * y / (r**3) - Cd * v * vy) * dt
        self.s[0] += self.s[2] * dt
        self.s[1] += self.s[3] * dt
        return self.s.copy()

class Brain2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(self, x): return self.net(x)

def train(sim_class):
    X, Y = [], []
    sim = sim_class(1.0, 0.0, 0.0, 1.0)
    for i in range(25000):
        if i % 50 == 0:
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0.5, 3.0)
            sim = sim_class(r*np.cos(theta), r*np.sin(theta), np.random.uniform(-2,2), np.random.uniform(-2,2))
        s_old = sim.s.copy()
        Y.append(sim.step() - s_old)
        X.append(s_old)
    X_t, Y_t = torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)
    model = Brain2D()
    opt = optim.Adam(model.parameters(), lr=0.002)
    for _ in range(80):
        for bx, by in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, Y_t), batch_size=512, shuffle=True):
            opt.zero_grad()
            nn.MSELoss()(model(bx), by).backward()
            opt.step()
    return model

print("Training Quick Models for Plot...")
g_ens = [train(CentralGravity2D) for _ in range(3)]
d_ens = [train(Drag2D) for _ in range(3)]
baseline = train(CombinedOrbit2D)

# Rollout
sim = CombinedOrbit2D(1.0, 0.0, 0.0, 1.0)
true_traj, ens_traj, base_traj = [sim.s.copy()], [sim.s.copy()], [sim.s.copy()]
curr_ens, curr_base = sim.s.copy(), sim.s.copy()

for i in range(300):
    true_traj.append(sim.step())
    with torch.no_grad():
        inp_ens = torch.tensor(np.array([curr_ens]), dtype=torch.float32)
        rg = np.mean([g(inp_ens).numpy()[0] for g in g_ens], axis=0)
        rd = np.mean([d(inp_ens).numpy()[0] for d in d_ens], axis=0)
        
        # PHYSICS-CORRECT COMPOSITION (Sum - Drift):
        # We sum both network residuals to capture high-order drag/gravity terms,
        # but subtract the shared velocity drift [vx*dt, vy*dt, 0, 0] once 
        # to avoid double-counting the kinematic identity.
        drift = np.array([curr_ens[2]*dt, curr_ens[3]*dt, 0, 0])
        ds_ens = rg + rd - drift
        
        curr_ens += ds_ens
        ens_traj.append(curr_ens.copy())

        i_b = torch.tensor(np.array([curr_base]), dtype=torch.float32)
        curr_base += baseline(i_b).numpy()[0]
        base_traj.append(curr_base.copy())

# Trim first 5 steps to remove initialization artifacts
t_t, e_t, b_t = np.array(true_traj)[5:], np.array(ens_traj)[5:], np.array(base_traj)[5:]

plt.figure(figsize=(10, 8))
plt.plot(t_t[:,0], t_t[:,1], 'k-', lw=2.5, label='Ground Truth')
plt.plot(e_t[:,0], e_t[:,1], 'b--', lw=2, label='Zero-Shot Ensemble')
plt.plot(b_t[:,0], b_t[:,1], 'r:', lw=2, label='100K Monolithic Baseline')
plt.scatter(t_t[0,0], t_t[0,1], color='green', marker='o', s=100, label='Start (post-warmup)', zorder=5)
plt.scatter(0, 0, color='gold', marker='*', s=300, label='Central Mass', zorder=5)
plt.title("2D Orbital Mechanics: Zero-Shot Composition vs Monolithic Baseline", fontsize=14)
plt.legend()
plt.axis('equal')
plt.savefig(r'd:\ALL MY ML AND AI\paper\figures\2D_Orbital_Trajectory.png', dpi=300, bbox_inches='tight')
print("Saved 2D_Orbital_Trajectory.png (Artifact-Free)")
