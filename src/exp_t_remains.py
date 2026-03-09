import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

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
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
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

def train_model(model, X, Y, epochs=100, lr=0.001):
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

def eval_composition(gravity_models, spring_models, seed=None):
    if seed is not None:
        np.random.seed(seed)
    errors = []
    for _ in range(5000):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        x = torch.tensor([[y, v, 0.0]], dtype=torch.float32)
        
        with torch.no_grad():
            rg_list = [g(x).numpy()[0] for g in gravity_models]
            rs_list = [s(x).numpy()[0] for s in spring_models]
            rg = np.mean(rg_list, axis=0)
            rs = np.mean(rs_list, axis=0)
            dv_pred = rg[1] + rs[1] + F * dt
            dy_pred = rg[0] + rs[0] - v * dt
                
        err = (dy_pred - true_dy)**2 + (dv_pred - true_dv)**2
        errors.append(err)
    return np.mean(errors)

if __name__ == "__main__":
    ensemble_size = 5
    ens_mses = []
    
    for seed in [7, 8, 9]:
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        g_ensemble = []
        s_ensemble = []
        
        print(f"\n--- SEED {seed+1}/10 ---")
        for i in range(ensemble_size):
            Xg, Yg = collect_gravity_data(20000, bounds=(-15, 15))
            bg = Brain()
            train_model(bg, Xg, Yg, epochs=100)
            g_ensemble.append(bg)
            
            Xs, Ys = collect_spring_data(20000, bounds=(-15, 15))
            bs = Brain()
            train_model(bs, Xs, Ys, epochs=100)
            s_ensemble.append(bs)
            
        mse_ens = eval_composition(g_ensemble, s_ensemble, seed=seed)
        ens_mses.append(mse_ens)
        print(f"  Ensemble MSE (0 target data): {mse_ens*1e4:.4f} x 10^-4")
