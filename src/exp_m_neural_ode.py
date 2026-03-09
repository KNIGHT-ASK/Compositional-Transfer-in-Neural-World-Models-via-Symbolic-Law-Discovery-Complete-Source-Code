"""
EXPERIMENT M: Neural ODE Baseline for Environment C
====================================================
Replaces the unfair HNN baseline with a Neural ODE that does NOT 
assume energy conservation, making it a fair structural comparator.

The Neural ODE learns ds/dt = f_theta(s,F) and integrates using 
a simple Euler step (matching our MLP's residual framing exactly).

Unlike HNNs, Neural ODEs can represent non-conservative forces
(friction, inelastic collisions) without architectural contradiction.

Run in Google Colab. Paste output back.
"""
import torch

# --- GOOGLE COLAB PYTORCH BUG FIX ---
if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# 1. Physics Engine (Environment C: Combined Gravity + Spring)
# ============================================================
dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

def step_combined(y, v, F_ext):
    acc = -g - (k_spring / mass) * y + (F_ext / mass)
    v_next = v + acc * dt
    y_next = y + v_next * dt
    if y_next < 0:
        y_next = -y_next
        v_next = -0.8 * v_next
    return y_next, v_next

def collect_data(n_transitions):
    states, next_states, actions = [], [], []
    y, v = np.random.uniform(2, 15), 0.0
    
    for _ in range(n_transitions):
        F_ext = np.random.uniform(-2.0, 2.0)
        y_next, v_next = step_combined(y, v, F_ext)
        states.append([y, v])
        actions.append([F_ext])
        next_states.append([y_next, v_next])
        y, v = y_next, v_next
        if np.random.rand() < 0.02:
            y, v = np.random.uniform(2, 15), np.random.uniform(-5, 5)

    S = np.array(states)
    A = np.array(actions)
    S_next = np.array(next_states)
    X = np.hstack((S, A))
    Y = S_next - S  # Residual target
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ============================================================
# 2. Neural ODE Architecture
# ============================================================
# The Neural ODE learns the time derivative: ds/dt = f_theta(s, F)
# Then predicts: Delta_s = f_theta(s, F) * dt
# This is structurally identical to residual prediction but the
# architecture EXPLICITLY separates the learned vector field from
# the integration step, making it a principled structural baseline.

class NeuralODE(nn.Module):
    """
    Neural ODE that learns the continuous-time vector field ds/dt = f(s, F).
    
    Key difference from standard MLP residual prediction:
    - The network outputs the TIME DERIVATIVE (ds/dt), not the residual (Delta s)
    - The residual is computed as: Delta s = f(s, F) * dt
    - This disentangles the learned physics from the integration scheme
    """
    def __init__(self):
        super(NeuralODE, self).__init__()
        # Fair capacity: 3-layer 128-unit, matching the 100k baseline MLP
        self.vector_field = nn.Sequential(
            nn.Linear(3, 128),   # Input: [y, v, F]
            nn.Tanh(),           # Smooth activations for ODE vector fields
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)    # Output: [dy/dt, dv/dt]
        )
    
    def forward(self, x):
        # x = [y, v, F]
        # Output: ds/dt = [dy/dt, dv/dt]
        dsdt = self.vector_field(x)
        # Integrate with single Euler step: Delta s = ds/dt * dt
        return dsdt * dt

def train_neural_ode(X, Y, model, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    batch_size = 1024
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.8f}")
    
    return model

def evaluate_mse(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        mse = nn.MSELoss()(preds, Y_test).item()
    return mse

# ============================================================
# 3. Main Experiment
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT M: Neural ODE Baseline (Fair Structural Comparator)")
    print("=" * 70)
    print()
    print("Unlike HNNs, Neural ODEs do NOT assume energy conservation.")
    print("This makes them a fair baseline for non-conservative systems.")
    print("Architecture: 3-layer 128-unit Tanh, 100K transitions, 1000 epochs")
    print()
    
    n_seeds = 10
    mses = []
    
    # Generate unified test set
    np.random.seed(999)
    X_test, Y_test = collect_data(2000)
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        X_train, Y_train = collect_data(100000)
        
        model = NeuralODE()
        model = train_neural_ode(X_train, Y_train, model, epochs=1000)
        
        mse = evaluate_mse(model, X_test, Y_test)
        mses.append(mse)
        print(f"  Seed {seed+1} MSE: {mse * 1e4:.4f} x 10^-4")
        print()
    
    mses = np.array(mses)
    mean_mse = np.mean(mses) * 1e4
    std_mse = np.std(mses) * 1e4
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Neural ODE Baseline MSE ({n_seeds} seeds): {mean_mse:.2f} +/- {std_mse:.2f} (x 10^-4)")
    print(f"Per-seed MSEs (x 10^-4): {[f'{m*1e4:.4f}' for m in mses]}")
    print()
    print("COMPARISON TARGETS:")
    print(f"  Zero-Shot Neural Composition: 8.3 +/- 3.5  x 10^-4")
    print(f"  Zero-Shot SINDy Composition:  2.4 +/- 0.2  x 10^-4")
    print(f"  Baseline MLP (100K):          0.35 +/- 0.05 x 10^-4")
    print(f"  Neural ODE (100K):            {mean_mse:.2f} +/- {std_mse:.2f} x 10^-4")
    print()
    
    # Save for statistical testing
    np.save("mse_neural_ode_100k.npy", mses)
    print("Saved MSE array to mse_neural_ode_100k.npy")
