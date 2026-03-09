import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- GOOGLE COLAB PYTORCH BUG FIX ---
if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

# Shared Constants
dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

class BouncingBallSim:
    def __init__(self, y0=0.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g + (F / mass)
        self.v += a * dt
        self.y += self.v * dt
        # If we expand domain, maybe ignore floor to ensure pure smooth physics,
        # or keep the floor at y=0 if y>0? Let's just do pure smooth gravity here
        # for a clean diagnostic, or keep the old logic. 
        # Actually, let's keep it simple: no floor logic in raw physics for the test,
        # because the combined env has no floor. We just want to learn gravity vector.
        return self.y, self.v

class SpringSim:
    def __init__(self, x0=0.0, v0=0.0):
        self.x, self.v = x0, v0
    def step(self, F=0.0):
        a = -(k_spring * self.x / mass) + (F / mass)
        self.v += a * dt
        self.x += self.v * dt
        return self.x, self.v

class CombinedSim:
    def __init__(self, y0=0.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g - (k_spring * self.y / mass) + (F / mass)
        self.v += a * dt
        self.y += self.v * dt
        return self.y, self.v

class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def collect_gravity_data(n, bounds=(-15, 15)):
    X, Y = [], []
    sim = BouncingBallSim(y0=np.random.uniform(*bounds))
    for i in range(n):
        if i % 50 == 0:
            sim = BouncingBallSim(y0=np.random.uniform(*bounds), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def collect_spring_data(n, bounds=(-15, 15)):
    X, Y = [], []
    sim = SpringSim(x0=np.random.uniform(*bounds))
    for i in range(n):
        if i % 50 == 0:
            sim = SpringSim(x0=np.random.uniform(*bounds), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        x_old, v_old = sim.x, sim.v
        x_new, v_new = sim.step(F)
        X.append([x_old, v_old, F])
        Y.append([x_new - x_old, v_new - v_old])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def train_model(model, X, Y, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def eval_composition(brain_g, brain_s, n_test=5000):
    errors = []
    # Test bounds are specific to combined env: [-10, 0]
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        # Neural Composition
        x_g = torch.tensor([[y, v, 0.0]], dtype=torch.float32)
        x_s = torch.tensor([[y, v, 0.0]], dtype=torch.float32)
        
        with torch.no_grad():
            rg = brain_g(x_g).numpy()[0]
            rs = brain_s(x_s).numpy()[0]
            
        dv_pred = rg[1] + rs[1] + F * dt
        dy_pred = rg[0] + rs[0] - v * dt
        
        err = (dy_pred - true_dy)**2 + (dv_pred - true_dv)**2
        errors.append(err)
        
    return np.mean(errors)

if __name__ == "__main__":
    n_seeds = 3
    
    print("=== HYPOTHESIS TEST: FIXING DISTRIBUTION MISMATCH ===")
    print("Training component models on y in [-15, 15] to cover the composition domain [-10, 0].")
    
    mses = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1. Train Gravity (20k data, full bounds [-15, 15])
        Xg, Yg = collect_gravity_data(20000, bounds=(-15, 15))
        bg = Brain()
        train_model(bg, Xg, Yg, epochs=50)
        
        # 2. Train Spring (20k data, full bounds [-15, 15])
        Xs, Ys = collect_spring_data(20000, bounds=(-15, 15))
        bs = Brain()
        train_model(bs, Xs, Ys, epochs=50)
        
        # 3. Evaluate Composition
        mse = eval_composition(bg, bs)
        mses.append(mse)
        print(f"Seed {seed}: MSE = {mse*1e4:.4f} x 10^-4")
        
    print(f"\nFINAL COMPOSITION MSE WITH FIXED DISTRIBUTION: {np.mean(mses)*1e4:.4f} ± {np.std(mses)*1e4:.4f} x 10^-4")
