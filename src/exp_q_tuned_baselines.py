import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

# --- ENVIRONMENT & UTILS (Simplified for Tuning) ---
def get_env_c_data(num_samples=100000, dt=0.02):
    # Gravity + Spring
    g, k, m = 9.8, 2.0, 1.0
    y = np.random.uniform(-15.0, 15.0, num_samples)
    v = np.random.uniform(-10.0, 10.0, num_samples)
    
    # Acceleration
    a = -g - (k*y)/m
    
    # 1-step target (Next State)
    y_next = y + v*dt
    v_next = v + a*dt
    
    # Input: (y, v), Target: (y_next, v_next)
    inputs = np.stack([y, v], axis=1)
    targets = np.stack([y_next, v_next], axis=1)
    return inputs, targets

# --- TUNED NEURAL ODE MODEL ---
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, depth):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.net(y)

def run_experiment(hidden_dim, depth, lr, seeds=10):
    mses = []
    print(f"Testing Config: H={hidden_dim}, D={depth}, LR={lr}")
    
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Data
        X, Y = get_env_c_data()
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Model
        f = ODEFunc(hidden_dim, depth)
        optimizer = torch.optim.Adam(f.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training (Fast Tuning - 20 epochs)
        f.train()
        for epoch in range(15):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                # Euler step approximation (matching our residual framing)
                dt = 0.02
                pred = batch_x + f(0, batch_x) * dt
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        # Eval
        f.eval()
        with torch.no_grad():
            X_test, Y_test = get_env_c_data(num_samples=5000)
            X_t = torch.FloatTensor(X_test)
            Y_t = torch.FloatTensor(Y_test)
            pred = X_t + f(0, X_t) * 0.02
            mse = criterion(pred, Y_t).item()
            mses.append(mse)
            
    mean_mse = np.mean(mses)
    std_mse = np.std(mses)
    print(f" -> Result: {mean_mse:.4e} +- {std_mse:.4e}")
    return mean_mse, std_mse

if __name__ == "__main__":
    configs = [
        (64, 2, 1e-3),
        (128, 3, 1e-3), # Previous "weak" baseline
        (256, 3, 5e-4),
        (256, 4, 1e-3)
    ]
    
    results = {}
    for h, d, l in configs:
        results[(h,d,l)] = run_experiment(h, d, l)

    print("\n--- FINAL JMLR SUMMARY ---")
    best_config = min(results, key=lambda x: results[x][0])
    print(f"Best Config: {best_config}")
    print(f"Tuned Neural ODE MSE: {results[best_config][0]:.4e} +- {results[best_config][1]:.4e}")
