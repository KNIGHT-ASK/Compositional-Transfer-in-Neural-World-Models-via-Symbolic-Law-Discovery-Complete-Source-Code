import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import os

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleMLP(nn.Module):
    """Same architecture used in the paper."""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def load_or_train_friction_net():
    save_path = "Brain_friction.pth"
    net = SimpleMLP().to(device)
    
    if os.path.exists(save_path):
        print("Loading existing friction network...")
        net.load_state_dict(torch.load(save_path, map_location=device))
        return net
    
    print("Training new friction network...")
    # True physics: a = -b/m * v
    b = 0.5
    m = 1.0
    dt = 0.02
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # Generate shuffled dataset
    N = 50000
    y = np.random.uniform(-15, 15, (N, 1))
    v = np.random.uniform(-10, 10, (N, 1))
    F = np.random.uniform(-2, 2, (N, 1))
    
    a = -b/m * v + F/m
    dy = v * dt
    dv = a * dt
    
    X = torch.FloatTensor(np.hstack([y, v, F])).to(device)
    Y = torch.FloatTensor(np.hstack([dy, dv])).to(device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred = net(X)
        loss = nn.MSELoss()(pred, Y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
    torch.save(net.state_dict(), save_path)
    return net

def diagnose_friction(net):
    print("\n--- DIAGNOSING FRICTION SINDy EXTRACTION ---")
    b = 0.5
    m = 1.0
    dt = 0.02
    true_coeff = -b/m * dt # -0.010
    
    # 1. Generate clean probe grid
    y_probe = np.linspace(-10, 10, 20)
    v_probe = np.linspace(-10, 10, 20)
    yy, vv = np.meshgrid(y_probe, v_probe)
    
    X_probe = np.zeros((400, 3))
    X_probe[:, 0] = yy.flatten()
    X_probe[:, 1] = vv.flatten()
    X_probe[:, 2] = 0.0 # Force = 0
    
    X_tensor = torch.FloatTensor(X_probe).to(device)
    
    with torch.no_grad():
         Y_pred = net(X_tensor).cpu().numpy()
         
    dv_pred = Y_pred[:, 1]
    
    # 2. SINDy Feature Library
    lib = np.zeros((400, 6))
    lib[:, 0] = 1 # constant
    lib[:, 1] = X_probe[:, 0] # y
    lib[:, 2] = X_probe[:, 1] # v
    lib[:, 3] = X_probe[:, 0]**2 # y^2
    lib[:, 4] = X_probe[:, 0] * X_probe[:, 1] # yv
    lib[:, 5] = X_probe[:, 1]**2 # v^2
    
    feature_names = ['1', 'y', 'v', 'y^2', 'yv', 'v^2']
    
    # 3. Fit LASSO across different alphas to see stability
    alphas = [1e-5, 1e-4, 1e-3]
    
    for alpha in alphas:
        print(f"\nLASSO Alpha = {alpha}:")
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000)
        model.fit(lib, dv_pred)
        
        recovered_coeff = model.coef_[2] # Coeff for 'v'
        accuracy = (recovered_coeff / true_coeff) * 100
        
        print(f"True Delta V eq: {true_coeff:.5f} * v")
        print("Discovered Eq: ", end="")
        terms = []
        for name, coef in zip(feature_names, model.coef_):
            if abs(coef) > 1e-6:
                terms.append(f"{coef:.6f} * {name}")
        print(" + ".join(terms))
        print(f"Recovered b_eff = {recovered_coeff/(-dt/m):.4f} (True: {b}) --> {accuracy:.2f}% accuracy")
        
    # 4. Analyze Neural V-Space prediction
    plt.figure(figsize=(10, 5))
    
    # Fix y=0, sweep v
    v_sweep = np.linspace(-10, 10, 100)
    X_v = np.zeros((100, 3))
    X_v[:, 1] = v_sweep
    with torch.no_grad():
        dv_sweep = net(torch.FloatTensor(X_v).to(device)).cpu().numpy()[:, 1]
        
    dv_true = -b/m * dt * v_sweep
    
    plt.plot(v_sweep, dv_true, 'k--', label='True Physics (Linear)')
    plt.plot(v_sweep, dv_sweep, 'r-', label='Neural Network Prediction')
    plt.xlabel('Velocity (v)')
    plt.ylabel('Delta V (dv)')
    plt.title('Friction Network: True vs Predicted Delta V')
    plt.legend()
    plt.grid(True)
    plt.savefig('friction_diagnosis.png')
    print("\nSaved plot to friction_diagnosis.png")

if __name__ == "__main__":
    net = load_or_train_friction_net()
    diagnose_friction(net)
