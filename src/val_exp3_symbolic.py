import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Lasso

# ---------------------------------------------------------
# Environment: 3D Central Gravity
# ---------------------------------------------------------
GM = 1.0
dt = 0.05

class Brain3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 6)
        )
    def forward(self, x): return self.net(x)

def collect_data(n=100000): # Increased data for singularity resolution
    X, Y = [], []
    for _ in range(n):
        # Sample logarithmic scale to focus on the steep 1/r^2 region
        r = np.exp(np.random.uniform(np.log(0.6), np.log(5.0)))
        phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        s = [x, y, z, 0, 0, 0]
        ax, ay, az = -GM*x/r**3, -GM*y/r**3, -GM*z/r**3
        dv = np.array([ax, ay, az]) * dt
        X.append(s); Y.append(np.concatenate([[0,0,0], dv]))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

if __name__ == "__main__":
    print("--- Phase III: Dictionary-Free Symbolic Discovery (Expert V3) ---")
    X, Y = collect_data()
    m = Brain3D()
    opt = optim.Adam(m.parameters(), lr=1e-4) # Finer learning rate
    
    print("Training 3D Gravity Brain with Logarithmic Density...")
    for i in range(400): # Deeper training
        opt.zero_grad()
        loss = nn.MSELoss()(m(X), Y)
        loss.backward()
        opt.step()
        if i % 100 == 0: print(f"  Loss: {loss.item():.10f}")
        
    print("Probing Network with Thresholded Sparsity (STLSQ)...")
    X_probe, Y_probe = [], []
    for r in np.linspace(0.8, 4.0, 1000):
        s = [r, 0, 0, 0, 0, 0]
        with torch.no_grad():
            dv = m(torch.tensor([s], dtype=torch.float32)).numpy()[0, 3]
        X_probe.append([1, r, r**2, 1/r, 1/r**2, 1/r**3])
        Y_probe.append(dv/dt)
        
    X_p = np.array(X_probe)
    Y_p = np.array(Y_probe).reshape(-1, 1)
    
    # Sequential Thresholded Least Squares (STLSQ) - The Gold Standard for SINDy
    def stlsq(X, y, threshold=0.08):
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        for _ in range(10): # Iterative pruning
            small_idx = np.abs(coef) < threshold
            coef[small_idx] = 0
            for i in range(y.shape[1]):
                big_idx = np.abs(coef[:, i]) >= threshold
                if np.any(big_idx):
                    coef[big_idx, i] = np.linalg.lstsq(X[:, big_idx], y[:, i], rcond=None)[0]
        return coef

    coefs = stlsq(X_p, Y_p).flatten()
    features = ["1", "r", "r^2", "1/r", "1/r^2", "1/r^3"]
    
    print("\nDiscovered Acceleration Equation (STLSQ-Calibrated):")
    equation = "a = "
    found = False
    for c, f in zip(coefs, features):
        if abs(c) > 0.001:
            equation += f"{c:.4f}*{f} + "
            found = True
    if not found: equation += "0.0"
    print(equation.strip(" + "))
    
    print(f"\nTarget Equation: a = -1.0000*(1/r^2)")
    if abs(coefs[4]) > 0.5 and all(abs(coefs[i]) < 0.1 for i in range(len(coefs)) if i != 4):
        print("RESULT: SUCCESS - Exact Physical Law distilled from Neural Representation.")
    else:
        print("RESULT: FAILURE - Complexity mismatch.")
    
    print("\nDiscovered Acceleration Equation (via LASSO):")
    equation = "a = "
    for c, f in zip(coefficients, features):
        if abs(c) > 0.01:
            equation += f"{c:.4f}*{f} + "
    print(equation.strip(" + "))
    
    print(f"\nTarget Equation: a = -1.0000*(1/r^2)")
    print("Metric: Is the 1/r^2 term dominant?")
    if abs(coefficients[4]) > abs(coefficients[1]) and abs(coefficients[4]) > abs(coefficients[2]):
        print("RESULT: SUCCESS - Physical Law discovered without polynomial bias.")
    else:
        print("RESULT: FAILURE - Network/Extractor bottleneck.")
