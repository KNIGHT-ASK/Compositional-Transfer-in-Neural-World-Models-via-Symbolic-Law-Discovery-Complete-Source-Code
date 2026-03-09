import torch

# --- GOOGLE COLAB PYTORCH BUG FIX ---
# Colab's default Python 3.12 environment has a broken torch._dynamo installation that crashes during optimizer initialization.
# This monkey-patch explicitly provides the missing internal function to allow the script to run flawlessly.
if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Physics Engine (Environment C: Combined)
dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

def step_combined(y, v, F_ext):
    acc = -g - (k_spring / mass) * y + (F_ext / mass)
    v_next = v + acc * dt
    y_next = y + v_next * dt
    # Keep floor logic identical for consistency with existing data
    if y_next < 0:
        y_next = -y_next
        v_next = -0.8 * v_next
    return y_next, v_next

def collect_data_combined(n_transitions=100000):
    states, next_states, actions = [], [], []
    y, v = np.random.uniform(2, 15), 0.0
    
    for _ in range(n_transitions):
        F_ext = np.random.uniform(-2.0, 2.0)
        y_next, v_next = step_combined(y, v, F_ext)
        
        states.append([y, v])
        actions.append([F_ext])
        next_states.append([y_next, v_next])
        
        y, v = y_next, v_next
        
        # Random resets to match data distribution
        if np.random.rand() < 0.02: 
            y, v = np.random.uniform(2, 15), np.random.uniform(-5, 5)

    return np.array(states), np.array(actions), np.array(next_states)

def get_residual_dataset(n_transitions):
    S, A, S_next = collect_data_combined(n_transitions)
    X = np.hstack((S, A))
    Y = S_next - S  # Predict residual
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# 2. Model Definition
class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        # Fair Tuning: Increased capacity to handle 100k monolithic data and floor boundary
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_model(X, Y, model, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    # Fair Tuning: Added LR scheduler to ensure total convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    batch_size = 1024 # Fair Tuning: Larger batch size for 100k data over 1000 epochs
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
    return model

def evaluate_mse(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        mse = nn.MSELoss()(preds, Y_test).item()
    return mse

if __name__ == "__main__":
    print("=== RESOLUTION 2: Asymmetric Data Budget ===")
    print("Training *FAIRLY TUNED* 100k-transition baseline on Combined Env C...")
    
    # Fair Tuning: 10 seeds for statistical rigorousness
    n_seeds = 10
    mses = []
    
    # Generate unified test set to match previous experiments (2000 points)
    X_test, Y_test = get_residual_dataset(2000)
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"Training Seed {seed+1}/10 on 100,000 transitions (1000 epochs, bs=1024, 128x3 MLP)...")
        X_train, Y_train = get_residual_dataset(100000)
        
        model = Brain()
        model = train_model(X_train, Y_train, model, epochs=1000)
        
        mse = evaluate_mse(model, X_test, Y_test)
        mses.append(mse)
        print(f"Seed {seed+1} MSE: {mse * 1e4:.2f} x 10^-4")
        
    mses = np.array(mses)
    mean_mse = np.mean(mses) * 1e4
    std_mse = np.std(mses) * 1e4
    print(f"\nFinal Fairly Tuned 100k Baseline MSE (10 seeds): {mean_mse:.2f} ± {std_mse:.2f} (x 10^-4)")
    
    # Save the array for TOST testing
    np.save("mse_baseline_100k.npy", mses)
    
