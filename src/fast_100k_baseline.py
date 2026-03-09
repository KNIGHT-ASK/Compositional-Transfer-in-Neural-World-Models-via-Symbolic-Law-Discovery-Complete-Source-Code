import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

class CombinedSim:
    def __init__(self, y0=0.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g - (k_spring * self.y / mass) + (F / mass)
        self.v += a * dt
        self.y += self.v * dt
        return self.y, self.v

class BigBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def collect_combined_data(n):
    X, Y = np.zeros((n, 3)), np.zeros((n, 2))
    sim = CombinedSim(y0=np.random.uniform(-10, 0))
    for i in range(n):
        if i % 50 == 0:
            sim = CombinedSim(y0=np.random.uniform(-10, 0), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X[i] = [y_old, v_old, F]
        Y[i] = [y_new - y_old, v_new - v_old]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_big(model, X, Y, epochs=300, lr=0.002, batch_size=2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.cpu()

def eval_baseline(model, n_test=5000):
    errors = []
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        x_t = torch.tensor([[y, v, F]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_t).numpy()[0]
        
        err = (pred[0] - true_dy)**2 + (pred[1] - true_dv)**2
        errors.append(err)
    return np.mean(errors)

if __name__ == "__main__":
    mses = []
    for seed in range(10):
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        Xc, Yc = collect_combined_data(100000)
        bl = BigBrain()
        train_big(bl, Xc, Yc, epochs=300) # 300 epochs is enough for 100K data to hit near-zero
        
        mse = eval_baseline(bl)
        mses.append(mse)
        print(f"Seed {seed}: Baseline 100K MSE = {mse*1e4:.4f} x 10^-4 (Took {time.time()-t0:.1f}s)")
    
    mses = np.array(mses)
    print(f"\nFINAL 100K BASELINE (10 seeds): {np.mean(mses)*1e4:.4f} ± {np.std(mses)*1e4:.4f} x 10^-4")
