"""
PRE-PAPER STRENGTHENING: ALL 6 EXPERIMENTS
==========================================
EXP 1: 5-Seed Statistical Significance
EXP 2: Ablation — Absolute vs Residual Prediction  
EXP 3: Retrain Gravity for Higher Accuracy
EXP 4: GRU vs MLP Comparison
EXP 5: Training Loss Curves (with logging)
EXP 6: 3rd Force — Friction/Damping Composition
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

# ============================================================
# SHARED ARCHITECTURE & ENVIRONMENTS
# ============================================================
class Brain1_v2(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

class GRUBrain(nn.Module):
    """GRU-based model for comparison with MLP."""
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features) for GRU
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class BouncingBallSim:
    def __init__(self, y0=10.0, v0=0.0, g=9.8, m=1.0, dt=0.02, epsilon=0.9):
        self.y, self.v, self.g, self.m, self.dt, self.epsilon = y0, v0, g, m, dt, epsilon
    def step(self, F=0.0):
        a = -self.g + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        if self.y < 0:
            self.y = 0
            self.v = -self.epsilon * self.v
        return np.array([self.y, self.v], dtype=np.float32)

class SpringSim:
    def __init__(self, x0=5.0, v0=0.0, k=2.0, m=1.0, dt=0.02):
        self.x, self.v, self.k, self.m, self.dt = x0, v0, k, m, dt
    def step(self, F=0.0):
        a = -(self.k * self.x / self.m) + (F / self.m)
        self.v += a * self.dt
        self.x += self.v * self.dt
        return np.array([self.x, self.v], dtype=np.float32)

class CombinedSim:
    def __init__(self, y0=0.0, v0=0.0, g=9.8, k=2.0, m=1.0, dt=0.02):
        self.y, self.v, self.g, self.k, self.m, self.dt = y0, v0, g, k, m, dt
    def step(self, F=0.0):
        a = -self.g - (self.k * self.y / self.m) + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        return np.array([self.y, self.v], dtype=np.float32)

class FrictionSim:
    """Environment D: Linear damping only (F_drag = -b * v)."""
    def __init__(self, x0=0.0, v0=5.0, b=0.5, m=1.0, dt=0.02):
        self.x, self.v, self.b, self.m, self.dt = x0, v0, b, m, dt
    def step(self, F=0.0):
        a = -(self.b * self.v / self.m) + (F / self.m)
        self.v += a * self.dt
        self.x += self.v * self.dt
        return np.array([self.x, self.v], dtype=np.float32)

class TripleCombinedSim:
    """Environment E: Gravity + Spring + Friction."""
    def __init__(self, y0=0.0, v0=0.0, g=9.8, k=2.0, b=0.5, m=1.0, dt=0.02):
        self.y, self.v = y0, v0
        self.g, self.k, self.b, self.m, self.dt = g, k, b, m, dt
    def step(self, F=0.0):
        a = -self.g - (self.k * self.y / self.m) - (self.b * self.v / self.m) + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        return np.array([self.y, self.v], dtype=np.float32)

# ============================================================
# HELPER: Collect transitions
# ============================================================
def collect_data(sim_class, num_steps, sim_kwargs=None, reset_every=50, residual=True):
    if sim_kwargs is None:
        sim_kwargs = {}
    states, actions, targets = [], [], []
    sim = sim_class(**sim_kwargs)
    curr = np.array([sim.y if hasattr(sim, 'y') else sim.x, sim.v], dtype=np.float32)
    
    for i in range(num_steps):
        if i % reset_every == 0:
            sim = sim_class(**sim_kwargs)
            # Randomize initial conditions 
            if hasattr(sim, 'y'):
                sim.y = np.random.uniform(-10.0, 10.0) if not isinstance(sim, BouncingBallSim) else np.random.uniform(1.0, 15.0)
            else:
                sim.x = np.random.uniform(-10.0, 10.0)
            sim.v = np.random.uniform(-5.0, 5.0)
            curr = np.array([sim.y if hasattr(sim, 'y') else sim.x, sim.v], dtype=np.float32)
        
        F = np.random.uniform(-2.0, 2.0)
        states.append(curr.copy())
        actions.append([F])
        next_s = sim.step(F)
        if residual:
            targets.append(next_s - curr)
        else:
            targets.append(next_s.copy())
        curr = next_s.copy()
    
    X = np.concatenate([np.array(states), np.array(actions)], axis=-1).astype(np.float32)
    Y = np.array(targets, dtype=np.float32)
    return X, Y

def train_model(model, X, Y, epochs=100, lr=0.001):
    hx, hy = torch.tensor(X), torch.tensor(Y)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for ep in range(epochs):
        # Shuffle every epoch
        perm = torch.randperm(hx.size(0))
        hx_s, hy_s = hx[perm], hy[perm]
        optimizer.zero_grad()
        preds = model(hx_s)
        loss = criterion(preds, hy_s)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def measure_composition_error(brain_g, brain_s, num_tests=2000, brain_f=None):
    """Measure zero-shot composition error on Environment C (or E if brain_f given)."""
    errors = []
    sim_class = TripleCombinedSim if brain_f else CombinedSim
    dt = 0.02
    
    for _ in range(num_tests):
        y = np.random.uniform(-10.0, 0.0)
        v = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        
        sim = sim_class()
        sim.y = y; sim.v = v
        next_s = sim.step(F)
        true_res = next_s - np.array([y, v])
        
        x_t = torch.tensor([[y, v, F]], dtype=torch.float32)
        zeros = torch.zeros((1, 1))
        safe_y = torch.full((1, 1), 10.0)
        
        grav_in = torch.cat([safe_y, x_t[:, 1:2], zeros], dim=-1)
        spr_in = torch.cat([x_t[:, 0:2], zeros], dim=-1)
        
        with torch.no_grad():
            res_g = brain_g(grav_in).numpy()[0]
            res_s = brain_s(spr_in).numpy()[0]
        
        dv_composed = res_g[1] + res_s[1] + F * dt
        dy_composed = res_g[0] + res_s[0] - v * dt
        
        if brain_f is not None:
            fric_in = torch.cat([zeros, x_t[:, 1:2], zeros], dim=-1)
            with torch.no_grad():
                res_f = brain_f(fric_in).numpy()[0]
            dv_composed += res_f[1]
        
        pred_res = np.array([dy_composed, dv_composed])
        err = np.sum((pred_res - true_res) ** 2)
        errors.append(err)
    
    return np.mean(errors)

def measure_baseline_error(model, num_tests=2000, sim_class=CombinedSim):
    errors = []
    dt = 0.02
    for _ in range(num_tests):
        y = np.random.uniform(-10.0, 0.0)
        v = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        sim = sim_class()
        sim.y = y; sim.v = v
        next_s = sim.step(F)
        true_res = next_s - np.array([y, v])
        
        x_t = torch.tensor([[y, v, F]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x_t).numpy()[0]
        err = np.sum((pred - true_res) ** 2)
        errors.append(err)
    return np.mean(errors)

all_results = {}

# ============================================================
# EXP 3: RETRAIN GRAVITY FOR HIGHER ACCURACY (do first!)
# ============================================================
print("=" * 70)
print("EXP 3: RETRAINING GRAVITY FOR HIGHER ACCURACY")
print("=" * 70)

X_grav, Y_grav = collect_data(
    BouncingBallSim, 100000,  # 100k transitions (was 50k)
    sim_kwargs={'y0': 10.0}, reset_every=50, residual=True
)

best_gravity = None
best_grav_loss = float('inf')

for attempt in range(3):
    model = Brain1_v2()
    losses = train_model(model, X_grav, Y_grav, epochs=300, lr=0.001)
    final_loss = losses[-1]
    
    # Test gravity invariance
    test_h = [2.0, 5.0, 8.0, 12.0]
    dvs = []
    for h in test_h:
        state = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(state).numpy()[0]
        dvs.append(pred[1])
    mean_dv = np.mean(dvs)
    std_dv = np.std(dvs)
    
    print(f"  Attempt {attempt+1}: Loss={final_loss:.6f}, Mean Δv={mean_dv:.4f}, Std={std_dv:.4f}")
    
    if final_loss < best_grav_loss and std_dv < 0.01:
        best_grav_loss = final_loss
        best_gravity = model

if best_gravity is None:
    best_gravity = model  # Take last one

# Save the improved gravity model
torch.save(best_gravity.state_dict(), '../models/brain1_gravity_model_v2.pth')

# Verify
test_heights = [1.0, 2.0, 5.0, 8.0, 12.0, 15.0]
print("\n  Improved gravity Δv across heights:")
for h in test_heights:
    state = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        pred = best_gravity(state).numpy()[0]
    print(f"    h={h:5.1f} -> Δv = {pred[1]:.6f}")

all_results['exp3'] = {'status': 'complete', 'model_saved': '../models/brain1_gravity_model_v2.pth'}
print("\n  ✓ EXP 3 COMPLETE\n")

# ============================================================
# EXP 5: TRAINING LOSS CURVES (with logging)
# ============================================================
print("=" * 70)
print("EXP 5: TRAINING LOSS CURVES")
print("=" * 70)

# Re-train gravity with loss logging
print("  Training Gravity (logging losses)...")
grav_model_for_loss = Brain1_v2()
grav_losses = train_model(grav_model_for_loss, X_grav, Y_grav, epochs=200, lr=0.001)

# Re-train spring with loss logging
print("  Training Spring (logging losses)...")
X_spr, Y_spr = collect_data(
    SpringSim, 50000,
    sim_kwargs={'x0': 5.0}, reset_every=50, residual=True
)
spr_model_for_loss = Brain1_v2()
spr_losses = train_model(spr_model_for_loss, X_spr, Y_spr, epochs=200, lr=0.001)

# Plot Figure 1
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(grav_losses, 'b-', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=13)
axes[0].set_ylabel('MSE Loss', fontsize=13)
axes[0].set_title('(a) Brain1_gravity Training', fontsize=13, fontweight='bold')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(spr_losses, 'r-', linewidth=1.5, alpha=0.8)
axes[1].set_xlabel('Epoch', fontsize=13)
axes[1].set_ylabel('MSE Loss', fontsize=13)
axes[1].set_title('(b) Brain1_spring Training', fontsize=13, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 1: Training Loss Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_training_loss.png', dpi=200, bbox_inches='tight')
print("  Saved: fig1_training_loss.png")

all_results['exp5'] = {
    'grav_final_loss': grav_losses[-1],
    'spr_final_loss': spr_losses[-1]
}
print("  ✓ EXP 5 COMPLETE\n")

# ============================================================
# EXP 1: 5-SEED STATISTICAL SIGNIFICANCE
# ============================================================
print("=" * 70)
print("EXP 1: 5-SEED STATISTICAL SIGNIFICANCE")
print("=" * 70)

seed_results = {'composed_mse': [], 'baseline_0_mse': [], 'baseline_500_mse': [],
                'baseline_2000_mse': [], 'baseline_10000_mse': []}

for seed in range(5):
    print(f"\n  --- Seed {seed+1}/5 ---")
    torch.manual_seed(seed * 42)
    np.random.seed(seed * 42)
    
    # Train gravity
    X_g, Y_g = collect_data(BouncingBallSim, 100000, sim_kwargs={'y0': 10.0}, reset_every=50)
    bg = Brain1_v2()
    train_model(bg, X_g, Y_g, epochs=200, lr=0.001)
    
    # Train spring
    X_s, Y_s = collect_data(SpringSim, 50000, sim_kwargs={'x0': 5.0}, reset_every=50)
    bs = Brain1_v2()
    train_model(bs, X_s, Y_s, epochs=200, lr=0.001)
    
    # Measure composed error
    comp_err = measure_composition_error(bg, bs)
    seed_results['composed_mse'].append(comp_err)
    print(f"    Composed MSE: {comp_err:.6f}")
    
    # Measure baseline at different training sizes
    for n_steps, key in [(0, 'baseline_0_mse'), (500, 'baseline_500_mse'),
                          (2000, 'baseline_2000_mse'), (10000, 'baseline_10000_mse')]:
        if n_steps == 0:
            bl = Brain1_v2()  # untrained
        else:
            X_c, Y_c = collect_data(CombinedSim, n_steps, reset_every=50)
            bl = Brain1_v2()
            train_model(bl, X_c, Y_c, epochs=100, lr=0.001)
        bl_err = measure_baseline_error(bl)
        seed_results[key].append(bl_err)
        print(f"    Baseline ({n_steps:5d} steps) MSE: {bl_err:.6f}")

# Compute statistics
print("\n  === STATISTICAL RESULTS (5 seeds) ===")
for key, vals in seed_results.items():
    mean = np.mean(vals)
    std = np.std(vals)
    print(f"  {key:25s}: {mean:.6f} ± {std:.6f}")

all_results['exp1'] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                       for k, v in seed_results.items()}
print("\n  ✓ EXP 1 COMPLETE\n")

# ============================================================
# EXP 2: ABLATION — ABSOLUTE vs RESIDUAL PREDICTION
# ============================================================
print("=" * 70)
print("EXP 2: ABLATION — ABSOLUTE vs RESIDUAL PREDICTION")
print("=" * 70)

torch.manual_seed(0)
np.random.seed(0)

# Train with RESIDUAL targets (our method)
print("  Training with RESIDUAL targets...")
X_g_res, Y_g_res = collect_data(BouncingBallSim, 50000, sim_kwargs={'y0': 10.0}, residual=True)
X_s_res, Y_s_res = collect_data(SpringSim, 50000, sim_kwargs={'x0': 5.0}, residual=True)

bg_res = Brain1_v2()
train_model(bg_res, X_g_res, Y_g_res, epochs=200)
bs_res = Brain1_v2()
train_model(bs_res, X_s_res, Y_s_res, epochs=200)
res_err = measure_composition_error(bg_res, bs_res)
print(f"  Residual composition MSE: {res_err:.6f}")

# Train with ABSOLUTE targets (ablation)
print("  Training with ABSOLUTE targets...")
X_g_abs, Y_g_abs = collect_data(BouncingBallSim, 50000, sim_kwargs={'y0': 10.0}, residual=False)
X_s_abs, Y_s_abs = collect_data(SpringSim, 50000, sim_kwargs={'x0': 5.0}, residual=False)

bg_abs = Brain1_v2()
train_model(bg_abs, X_g_abs, Y_g_abs, epochs=200)
bs_abs = Brain1_v2()
train_model(bs_abs, X_s_abs, Y_s_abs, epochs=200)

# Try composition with absolute models (subtract current state to get residual)
abs_errors = []
for _ in range(2000):
    y = np.random.uniform(-10.0, 0.0)
    v = np.random.uniform(-5.0, 5.0)
    F = np.random.uniform(-2.0, 2.0)
    
    sim = CombinedSim(); sim.y = y; sim.v = v
    next_s = sim.step(F)
    true_res = next_s - np.array([y, v])
    
    x_t = torch.tensor([[y, v, F]], dtype=torch.float32)
    safe_y = torch.full((1, 1), 10.0)
    zeros = torch.zeros((1, 1))
    
    grav_in = torch.cat([safe_y, x_t[:, 1:2], zeros], dim=-1)
    spr_in = torch.cat([x_t[:, 0:2], zeros], dim=-1)
    
    with torch.no_grad():
        pred_g = bg_abs(grav_in).numpy()[0]  # predicts absolute next state
        pred_s = bs_abs(spr_in).numpy()[0]
    
    # Try to compose absolute: subtract current state from each, then add
    res_g = pred_g - np.array([10.0, v])  # gravity's state was [10, v]
    res_s = pred_s - np.array([y, v])      # spring's state was [y, v]
    pred_res = np.array([
        res_g[0] + res_s[0] - (-v * 0.02),  # approximate
        res_g[1] + res_s[1] + F * 0.02
    ])
    err = np.sum((pred_res - true_res) ** 2)
    abs_errors.append(err)

abs_err = np.mean(abs_errors)
print(f"  Absolute composition MSE: {abs_err:.6f}")
print(f"\n  ABLATION RESULT:")
print(f"    Residual framing MSE: {res_err:.6f}")
print(f"    Absolute framing MSE: {abs_err:.6f}")
print(f"    Ratio: {abs_err/res_err:.1f}x worse with absolute targets")

all_results['exp2'] = {
    'residual_mse': float(res_err),
    'absolute_mse': float(abs_err),
    'ratio': float(abs_err / max(res_err, 1e-10))
}
print("\n  ✓ EXP 2 COMPLETE\n")

# ============================================================
# EXP 4: GRU vs MLP COMPARISON
# ============================================================
print("=" * 70)
print("EXP 4: GRU vs MLP COMPARISON")
print("=" * 70)

torch.manual_seed(0); np.random.seed(0)

# MLP trained on SHUFFLED data (our method)
print("  Training MLP on shuffled data...")
X_g_mlp, Y_g_mlp = collect_data(BouncingBallSim, 50000, sim_kwargs={'y0': 10.0})
mlp_model = Brain1_v2()
mlp_losses = train_model(mlp_model, X_g_mlp, Y_g_mlp, epochs=200)

# Test gravity invariance for MLP
mlp_dvs = []
for h in [2.0, 5.0, 8.0, 12.0, 15.0]:
    state = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        pred = mlp_model(state).numpy()[0]
    mlp_dvs.append(pred[1])

# GRU trained on SEQUENTIAL data (the failure case)
print("  Training GRU on sequential data...")
# Collect data WITHOUT reshuffling between resets for GRU
X_g_gru, Y_g_gru = collect_data(BouncingBallSim, 50000, sim_kwargs={'y0': 10.0}, reset_every=200)
gru_model = GRUBrain()
gru_losses = train_model(gru_model, X_g_gru, Y_g_gru, epochs=200)

# Test gravity invariance for GRU
gru_dvs = []
for h in [2.0, 5.0, 8.0, 12.0, 15.0]:
    state = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        pred = gru_model(state).numpy()[0]
    gru_dvs.append(pred[1])

# MLP trained on SEQUENTIAL (non-shuffled) data
print("  Training MLP on sequential (non-shuffled) data...")
mlp_seq = Brain1_v2()
hx_seq = torch.tensor(X_g_gru)
hy_seq = torch.tensor(Y_g_gru)
opt_seq = optim.Adam(mlp_seq.parameters(), lr=0.001)
crit = nn.MSELoss()
mlp_seq.train()
for ep in range(200):
    opt_seq.zero_grad()
    preds = mlp_seq(hx_seq)  # NO SHUFFLING
    loss = crit(preds, hy_seq)
    loss.backward()
    opt_seq.step()

seq_dvs = []
for h in [2.0, 5.0, 8.0, 12.0, 15.0]:
    state = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        pred = mlp_seq(state).numpy()[0]
    seq_dvs.append(pred[1])

heights = [2.0, 5.0, 8.0, 12.0, 15.0]
true_dv = -9.8 * 0.02

print(f"\n  GRAVITY INVARIANCE TEST:")
print(f"  {'Height':>10} | {'MLP(shuffled)':>14} | {'MLP(sequential)':>16} | {'GRU':>10} | {'True':>10}")
print(f"  {'-'*10} | {'-'*14} | {'-'*16} | {'-'*10} | {'-'*10}")
for i, h in enumerate(heights):
    print(f"  {h:10.1f} | {mlp_dvs[i]:14.6f} | {seq_dvs[i]:16.6f} | {gru_dvs[i]:10.6f} | {true_dv:10.6f}")

mlp_std = np.std(mlp_dvs)
gru_std = np.std(gru_dvs)
seq_std = np.std(seq_dvs)
print(f"\n  Std across heights (lower = more invariant):")
print(f"    MLP (shuffled):    {mlp_std:.6f}")
print(f"    MLP (sequential):  {seq_std:.6f}")
print(f"    GRU:               {gru_std:.6f}")

# Plot comparison
fig4, ax4 = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(heights))
width = 0.25

bars1 = ax4.bar(x_pos - width, mlp_dvs, width, label='MLP (shuffled)', color='#2ecc71', edgecolor='black')
bars2 = ax4.bar(x_pos, seq_dvs, width, label='MLP (sequential)', color='#f39c12', edgecolor='black')
bars3 = ax4.bar(x_pos + width, gru_dvs, width, label='GRU', color='#e74c3c', edgecolor='black')
ax4.axhline(y=true_dv, color='black', linestyle='--', linewidth=1.5, label=f'True Δv = {true_dv:.3f}')
ax4.set_xlabel('Height (y)', fontsize=13)
ax4.set_ylabel('Predicted Δv', fontsize=13)
ax4.set_title('Architecture Comparison: Gravity Invariance', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([str(h) for h in heights])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig_exp4_architecture_comparison.png', dpi=200, bbox_inches='tight')
print("\n  Saved: fig_exp4_architecture_comparison.png")

all_results['exp4'] = {
    'mlp_shuffled_std': float(mlp_std),
    'mlp_sequential_std': float(seq_std),
    'gru_std': float(gru_std)
}
print("  ✓ EXP 4 COMPLETE\n")

# ============================================================
# EXP 6: 3RD FORCE — FRICTION/DAMPING COMPOSITION
# ============================================================
print("=" * 70)
print("EXP 6: 3RD FORCE — FRICTION/DAMPING COMPOSITION")
print("=" * 70)

torch.manual_seed(0); np.random.seed(0)

# Train friction network
print("  Training Brain1_friction on Environment D (damping)...")
X_fric, Y_fric = collect_data(
    FrictionSim, 50000, sim_kwargs={'x0': 0.0, 'v0': 5.0}, reset_every=50
)
brain_friction = Brain1_v2()
fric_losses = train_model(brain_friction, X_fric, Y_fric, epochs=200)

# Verify friction learned: Δv should depend on v  
print("\n  Friction Δv at different velocities (y=0):")
for v_test in [-5.0, -2.0, 0.0, 2.0, 5.0]:
    state = torch.tensor([[0.0, v_test, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        pred = brain_friction(state).numpy()[0]
    expected_dv = -0.5 * v_test * 0.02  # -b*v*dt
    print(f"    v={v_test:5.1f} -> Δv={pred[1]:.6f} (expected: {expected_dv:.6f})")

torch.save(brain_friction.state_dict(), '../models/brain1_friction_model.pth')

# Load best gravity and spring
brain_g = Brain1_v2()
brain_g.load_state_dict(torch.load('../models/brain1_gravity_model.pth', weights_only=True))
brain_g.eval()

brain_s = Brain1_v2()
brain_s.load_state_dict(torch.load('../models/brain1_spring_model.pth', weights_only=True))
brain_s.eval()

brain_friction.eval()

# Measure 3-force composition on Environment E
triple_err = measure_composition_error(brain_g, brain_s, brain_f=brain_friction)
print(f"\n  3-Force Composition MSE (Env E): {triple_err:.6f}")

# Compare with baseline trained from scratch on Env E
print("  Training baseline from scratch on Env E...")
X_e, Y_e = collect_data(TripleCombinedSim, 10000, reset_every=50)
baseline_e = Brain1_v2()
train_model(baseline_e, X_e, Y_e, epochs=100)
baseline_e_err = measure_baseline_error(baseline_e, sim_class=TripleCombinedSim)
print(f"  Baseline (10k steps) MSE on Env E: {baseline_e_err:.6f}")

# Multi-step rollout for triple composition
num_steps = 500
y0, v0 = -2.0, 3.0
dt = 0.02

# Ground truth
sim = TripleCombinedSim(y0=y0, v0=v0)
true_ys, true_vs = [y0], [v0]
for _ in range(num_steps):
    s = sim.step(0.0)
    true_ys.append(s[0]); true_vs.append(s[1])

# Composed 3-network
comp_y, comp_v = y0, v0
comp_ys, comp_vs = [y0], [v0]
for _ in range(num_steps):
    x_t = torch.tensor([[comp_y, comp_v, 0.0]], dtype=torch.float32)
    zeros = torch.zeros((1, 1))
    safe_y = torch.full((1, 1), 10.0)
    g_in = torch.cat([safe_y, x_t[:, 1:2], zeros], dim=-1)
    s_in = torch.cat([x_t[:, 0:2], zeros], dim=-1)
    f_in = torch.cat([zeros, x_t[:, 1:2], zeros], dim=-1)
    with torch.no_grad():
        rg = brain_g(g_in).numpy()[0]
        rs = brain_s(s_in).numpy()[0]
        rf = brain_friction(f_in).numpy()[0]
    dy = rg[0] + rs[0] + rf[0] - 2 * comp_v * dt  # subtract 2 extra v*dt (3 models, take once)
    dv = rg[1] + rs[1] + rf[1]
    comp_y += dy; comp_v += dv
    comp_ys.append(comp_y); comp_vs.append(comp_v)

# Plot triple rollout
t_arr = np.arange(num_steps + 1) * dt
fig6, axes6 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes6[0].plot(t_arr, true_ys, 'k-', lw=2, label='Ground Truth (Gravity+Spring+Friction)')
axes6[0].plot(t_arr, comp_ys, 'r--', lw=1.8, label='3-Network Composition (zero-shot)')
axes6[0].set_ylabel('Position', fontsize=13)
axes6[0].set_title('3-Force Composition: Gravity + Spring + Friction', fontsize=14, fontweight='bold')
axes6[0].legend(fontsize=11)
axes6[0].grid(True, alpha=0.3)

axes6[1].plot(t_arr, true_vs, 'k-', lw=2, label='Ground Truth')
axes6[1].plot(t_arr, comp_vs, 'r--', lw=1.8, label='3-Network Composition')
axes6[1].set_ylabel('Velocity', fontsize=13)
axes6[1].set_xlabel('Time (s)', fontsize=13)
axes6[1].legend(fontsize=11)
axes6[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_exp6_triple_composition.png', dpi=200, bbox_inches='tight')
print("  Saved: fig_exp6_triple_composition.png")

all_results['exp6'] = {
    'triple_composition_mse': float(triple_err),
    'baseline_10k_mse': float(baseline_e_err)
}
print("  ✓ EXP 6 COMPLETE\n")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
with open('experiment_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("=" * 70)
print("ALL 6 EXPERIMENTS COMPLETE")
print("=" * 70)
print(f"""
  Results saved to: experiment_results.json
  
  Generated Figures:
    • fig1_training_loss.png        (EXP 5)
    • fig_exp4_architecture_comparison.png (EXP 4)
    • fig_exp6_triple_composition.png      (EXP 6)
  
  Summary:
    EXP 1: 5-seed stats — Composed MSE: {all_results['exp1']['composed_mse']['mean']:.6f} ± {all_results['exp1']['composed_mse']['std']:.6f}
    EXP 2: Ablation — Residual {all_results['exp2']['residual_mse']:.6f} vs Absolute {all_results['exp2']['absolute_mse']:.6f} ({all_results['exp2']['ratio']:.1f}x)
    EXP 3: Improved gravity model saved
    EXP 4: MLP(shuffled) std={all_results['exp4']['mlp_shuffled_std']:.6f} vs GRU std={all_results['exp4']['gru_std']:.6f}
    EXP 5: Loss curves generated
    EXP 6: Triple MSE={all_results['exp6']['triple_composition_mse']:.6f}
""")
