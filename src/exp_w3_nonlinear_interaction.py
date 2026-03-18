import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------
# COUPLED PHYSICS SIMULATOR
# -----------------------------------------------
def get_coupled_data(n, coupling=True):
    # State: [x, v]
    x = torch.empty(n, 1, device=device).uniform_(-3, 3)
    v = torch.empty(n, 1, device=device).uniform_(-3, 3)
    S = torch.cat([x, v], dim=1)
    
    # Base Forces
    a_g = -2.0 # Gravity
    a_s = -2.0 * x # Spring
    
    # Non-Linear Coupling: Magnetic-Viscous interaction Phi(x, v) = alpha * cos(x*v)
    alpha = 5.0
    phi = alpha * torch.cos(x * v) if coupling else torch.zeros_like(x)
    
    a_total = a_g + a_s + phi
    
    dt = 0.01
    dy = torch.cat([(v + 0.5 * a_total * dt) * dt, a_total * dt], dim=1)
    
    return S, dy

class ResidualMLP(nn.Module):
    def __init__(self, in_d=2, out_d=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, out_d)
        )
    def forward(self, x): return self.net(x)

def train_module(S, Y, label):
    print(f"Learning {label}...")
    model = ResidualMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model(S), Y)
        loss.backward()
        optimizer.step()
    return model

# -----------------------------------------------
# EXECUTION
# -----------------------------------------------
if __name__ == "__main__":
    print("--- NON-LINEAR INTERACTION DISCOVERY ---")
    
    # 1. Train Base Modules on isolated clean data
    S_g, Y_g = get_coupled_data(20000, coupling=False) # Isolated Gravity (Spring=0, Phi=0)
    # Re-simulating for pure isolation
    S_g = torch.cat([torch.empty(20000,1,device=device).uniform_(-5,5), torch.empty(20000,1,device=device).uniform_(-5,5)], dim=1)
    Y_g = torch.cat([(S_g[:,1:2] + 0.5*(-2.0)*0.01)*0.01, torch.full((20000,1), -2.0*0.01, device=device)], dim=1)
    m_grav = train_module(S_g, Y_g, "Base Gravity")
    
    S_s = torch.cat([torch.empty(20000,1,device=device).uniform_(-5,5), torch.empty(20000,1,device=device).uniform_(-5,5)], dim=1)
    Y_s = torch.cat([(S_s[:,1:2] + 0.5*(-2.0*S_s[:,0:1])*0.01)*0.01, (-2.0*S_s[:,0:1])*0.01], dim=1)
    m_spring = train_module(S_s, Y_s, "Base Spring")
    
    # 2. Compose and measure failure on Coupled System
    print("\nAttempting Zero-Shot Composition on Coupled System...")
    S_c, Y_c = get_coupled_data(20000, coupling=True)
    with torch.no_grad():
        Y_composed = m_grav(S_c) + m_spring(S_c)
        failure_mse = nn.MSELoss()(Y_composed, Y_c).item()
    print(f"Zero-Shot MSE (Superposition only): {failure_mse:.8f}")
    
    # 3. Learn the "Interaction Residual Module"
    # Target: The error of the superposition (which should be Phi)
    Y_residual = Y_c - Y_composed
    m_interaction = train_module(S_c, Y_residual, "Interaction Module")
    
    # 4. Final Finalized Composition
    with torch.no_grad():
        Y_final = Y_composed + m_interaction(S_c)
        final_mse = nn.MSELoss()(Y_final, Y_c).item()
        
    print(f"\nFinal Composed MSE (Base + Interaction): {final_mse:.8f}")
    
    if final_mse < failure_mse * 0.01:
        print("RESULT: SUCCESS. The framework discovered and corrected for non-linear interactions.")
    else:
        print("RESULT: FAILURE to resolve coupling.")
