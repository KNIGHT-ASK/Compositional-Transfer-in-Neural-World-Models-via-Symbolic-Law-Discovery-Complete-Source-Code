import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------
# CORRECTED PHYSICS SIMULATOR
# -----------------------------------------------
def get_data(n, noise_level=0.05, force_type='gravity'):
    """
    Correct Euler residuals: delta_x = v*dt, delta_v = a*dt
    Noise is absolute (fixed sigma), not relative to std,
    to avoid the zero-std collapse on constant forces.
    """
    dt = 0.01
    
    x = torch.empty(n, 1, device=device).uniform_(-5, 5)
    v = torch.empty(n, 1, device=device).uniform_(-5, 5)
    S = torch.cat([x, v], dim=1)

    if force_type == 'gravity':
        a = torch.full((n, 1), -2.0, device=device)
    elif force_type == 'spring':
        a = -2.0 * x
    elif force_type == 'combined':
        a = -2.0 + (-2.0 * x)

    # CORRECT residuals
    delta_x = v * dt
    delta_v = a * dt
    dy_clean = torch.cat([delta_x, delta_v], dim=1)

    # Fixed absolute noise (avoids zero-std collapse)
    if noise_level > 0:
        S_noisy  = S  + torch.randn_like(S)  * noise_level
        dy_noisy = dy_clean + torch.randn_like(dy_clean) * noise_level * 0.1
    else:
        S_noisy  = S.clone()
        dy_noisy = dy_clean.clone()

    return S_noisy, dy_noisy, S, dy_clean


# -----------------------------------------------
# MODEL
# -----------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)


# -----------------------------------------------
# TRAINING
# -----------------------------------------------
def train_module(force_type, noise=0.05, epochs=200, hidden=64, label=None):
    label = label or force_type
    print(f"\nTraining [{label}] | noise={noise*100:.0f}% | hidden={hidden}")
    
    model = ResidualMLP(hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    S_n, Y_n, _, _ = get_data(50000, noise_level=noise, force_type=force_type)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(S_n), Y_n)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  Ep {epoch:3d} | Loss: {loss.item():.7f}")

    return model


# -----------------------------------------------
# EVALUATION
# -----------------------------------------------
def evaluate(models_dict, label):
    """
    Evaluate zero-shot composition vs monolith on clean combined ground truth.
    models_dict: {'composed': [m1, m2], 'monolith': m3}
    """
    _, _, S_clean, Y_clean = get_data(10000, noise_level=0.0, force_type='combined')
    criterion = nn.MSELoss()
    results = {}

    with torch.no_grad():
        # Zero-shot composition
        composed_modules = models_dict.get('composed', [])
        if composed_modules:
            pred_composed = sum(m(S_clean) for m in composed_modules)
            mse_composed = criterion(pred_composed, Y_clean).item()
            results['Zero-Shot Composed'] = mse_composed

        # Monolithic baseline
        mono = models_dict.get('monolith')
        if mono:
            pred_mono = mono(S_clean)
            mse_mono = criterion(pred_mono, Y_clean).item()
            results['Monolithic Baseline'] = mse_mono

    print(f"\n{'='*50}")
    print(f"  RESULTS — {label}")
    print(f"{'='*50}")
    for name, mse in results.items():
        print(f"  {name:<28} MSE = {mse:.8f}")

    if 'Zero-Shot Composed' in results and 'Monolithic Baseline' in results:
        ratio = results['Monolithic Baseline'] / results['Zero-Shot Composed']
        winner = "COMPOSITIONAL" if ratio > 1 else "MONOLITHIC"
        print(f"\n  Advantage ratio : {ratio:.2f}x  →  {winner} WINS")

    return results


# -----------------------------------------------
# EXPERIMENT 1: Noise Robustness
# -----------------------------------------------
print("\n" + "="*50)
print("  EXPERIMENT 1: Noise Robustness (5% noise)")
print("="*50)

m_grav   = train_module('gravity',  noise=0.05)
m_spring = train_module('spring',   noise=0.05)
m_mono   = train_module('combined', noise=0.05, hidden=128, label='monolith-combined')

evaluate(
    {'composed': [m_grav, m_spring], 'monolith': m_mono},
    label="5% Noise Robustness"
)

# -----------------------------------------------
# EXPERIMENT 2: Fair Zero-Shot Comparison
# The real W1 fix — monolith sees ISOLATED data only
# -----------------------------------------------
print("\n" + "="*50)
print("  EXPERIMENT 2: Fair Isolated-Data Comparison")
print("  Monolith trained on ISOLATED data (no combined)")
print("="*50)

# Collect isolated data for both forces, concatenate for monolith
S_g, Y_g, _, _ = get_data(50000, noise_level=0.0, force_type='gravity')
S_s, Y_s, _, _ = get_data(50000, noise_level=0.0, force_type='spring')

S_iso = torch.cat([S_g, S_s], dim=0)
Y_iso = torch.cat([Y_g, Y_s], dim=0)

# Shuffle
idx = torch.randperm(S_iso.shape[0])
S_iso, Y_iso = S_iso[idx], Y_iso[idx]

# Train monolith on isolated data
m_mono_iso = ResidualMLP(hidden=128).to(device)
optimizer  = optim.Adam(m_mono_iso.parameters(), lr=1e-3)
criterion  = nn.MSELoss()

print("\nTraining [monolith-isolated] | noise=0% | hidden=128")
for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(m_mono_iso(S_iso), Y_iso)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"  Ep {epoch:3d} | Loss: {loss.item():.7f}")

m_grav_clean   = train_module('gravity', noise=0.0)
m_spring_clean = train_module('spring',  noise=0.0)

evaluate(
    {'composed': [m_grav_clean, m_spring_clean], 'monolith': m_mono_iso},
    label="Fair Isolated-Data Comparison (W1 Fix)"
)