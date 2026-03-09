"""
EXPERIMENT P: Complete Pairwise Statistical Testing (SELF-CONTAINED)
====================================================================
This script computes ALL pairwise comparisons for Table 2.

It does NOT require .npy files. Instead, it RE-RUNS the core experiments
with 10 seeds each to generate fresh, consistent MSE arrays.

It computes:
- Welch's t-test (two-tailed, unequal variance)
- Cohen's d (effect size)
- Per-seed MSE breakdown

NOTE: This is computationally expensive (~2-3 hours on Colab GPU).
      It trains 10 seeds x 3 model types x 100K+ transitions each.

Run in Google Colab WITH GPU. Paste output back.
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
from scipy import stats

# ============================================================
# 1. Shared Physics Engines
# ============================================================
dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

class BouncingBallSim:
    def __init__(self, y0=10.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g + (F / mass)
        self.v += a * dt
        self.y += self.v * dt
        if self.y < 0:
            self.y = 0
            self.v = -0.9 * self.v
        return self.y, self.v

class SpringSim:
    def __init__(self, x0=5.0, v0=0.0):
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

# ============================================================
# 2. Architecture
# ============================================================
class Brain(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        mods = [nn.Linear(3, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.ReLU()]
        mods.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*mods)
    def forward(self, x):
        return self.net(x)

class BigBrain(nn.Module):
    """Tuned 100K baseline: 3-layer 128-unit"""
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

# ============================================================
# 3. Data Collection Helpers
# ============================================================
def collect_gravity_data(n):
    X, Y = [], []
    sim = BouncingBallSim(y0=np.random.uniform(2, 15))
    for i in range(n):
        if i % 50 == 0:
            sim = BouncingBallSim(y0=np.random.uniform(2, 15), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def collect_spring_data(n):
    X, Y = [], []
    sim = SpringSim(x0=np.random.uniform(-10, 10))
    for i in range(n):
        if i % 50 == 0:
            sim = SpringSim(x0=np.random.uniform(-10, 10), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        x_old, v_old = sim.x, sim.v
        x_new, v_new = sim.step(F)
        X.append([x_old, v_old, F])
        Y.append([x_new - x_old, v_new - v_old])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def collect_combined_data(n):
    X, Y = [], []
    sim = CombinedSim(y0=np.random.uniform(-10, 0))
    for i in range(n):
        if i % 50 == 0:
            sim = CombinedSim(y0=np.random.uniform(-10, 0), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ============================================================
# 4. Training Helper
# ============================================================
def train(model, X, Y, epochs=200, lr=0.001, batch_size=512):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def train_big(model, X, Y, epochs=1000, lr=0.002, batch_size=1024):
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
    return model

# ============================================================
# 5. Evaluation: Composition MSE on Env C
# ============================================================
def eval_composition(brain_g, brain_s, n_test=5000):
    """Evaluate zero-shot neural composition on Environment C."""
    errors = []
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        x_g = torch.tensor([[10.0, v, 0.0]])  # safe_y for gravity
        x_s = torch.tensor([[y, v, 0.0]])
        with torch.no_grad():
            rg = brain_g(x_g).numpy()[0]
            rs = brain_s(x_s).numpy()[0]
        
        dv_pred = rg[1] + rs[1] + F * dt
        dy_pred = rg[0] + rs[0] - v * dt
        
        err = (dy_pred - true_dy)**2 + (dv_pred - true_dv)**2
        errors.append(err)
    return np.mean(errors)

def eval_sindy_composition(n_test=5000):
    """Evaluate symbolic (SINDy) composition on Environment C."""
    # SINDy discovered: dv = -0.181 - 0.040*y, dy = 0.020*v
    errors = []
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        dv_pred = -0.181 + (-0.040 * y) + F * dt
        dy_pred = 0.020 * v
        
        err = (dy_pred - true_dy)**2 + (dv_pred - true_dv)**2
        errors.append(err)
    return np.mean(errors)

def eval_baseline(model, n_test=5000):
    """Evaluate a monolithic baseline on Environment C."""
    errors = []
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        x_t = torch.tensor([[y, v, F]])
        with torch.no_grad():
            pred = model(x_t).numpy()[0]
        
        err = (pred[0] - true_dy)**2 + (pred[1] - true_dv)**2
        errors.append(err)
    return np.mean(errors)

# ============================================================
# 6. Statistical Functions
# ============================================================
def welch_t(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / max(sp, 1e-15)

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."

# ============================================================
# 7. MAIN EXPERIMENT: 10 Seeds Each
# ============================================================
if __name__ == "__main__":
    N_SEEDS = 10
    
    mse_composed_neural = []
    mse_composed_sindy = []
    mse_baseline_10k = []
    mse_baseline_100k = []
    
    print("=" * 70)
    print("EXPERIMENT P: FULL STATISTICAL TESTING (10 SEEDS)")
    print("=" * 70)
    
    # --- Part A: Composed Neural (10 seeds) ---
    print("\n--- PART A: Zero-Shot Neural Composition (10 seeds) ---")
    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        Xg, Yg = collect_gravity_data(100000)
        bg = Brain()
        train(bg, Xg, Yg, epochs=200)
        
        Xs, Ys = collect_spring_data(50000)
        bs = Brain()
        train(bs, Xs, Ys, epochs=200)
        
        mse = eval_composition(bg, bs)
        mse_composed_neural.append(mse)
        print(f"  Seed {seed}: Composed Neural MSE = {mse*1e4:.4f} x 10^-4")
    
    # --- Part B: Composed SINDy (10 seeds) ---
    print("\n--- PART B: Zero-Shot SINDy Composition (10 seeds) ---")
    for seed in range(N_SEEDS):
        np.random.seed(seed * 42)
        mse = eval_sindy_composition()
        mse_composed_sindy.append(mse)
        print(f"  Seed {seed}: Composed SINDy MSE = {mse*1e4:.4f} x 10^-4")
    
    # --- Part C: Baseline 10K (10 seeds) ---
    print("\n--- PART C: Baseline MLP 10K (10 seeds) ---")
    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        Xc, Yc = collect_combined_data(10000)
        bl = Brain()
        train(bl, Xc, Yc, epochs=100)
        
        mse = eval_baseline(bl)
        mse_baseline_10k.append(mse)
        print(f"  Seed {seed}: Baseline 10K MSE = {mse*1e4:.4f} x 10^-4")
    
    # --- Part D: Baseline 100K Tuned (10 seeds) ---
    print("\n--- PART D: Baseline MLP 100K Tuned (10 seeds) ---")
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        Xc, Yc = collect_combined_data(100000)
        bl = BigBrain()
        train_big(bl, Xc, Yc, epochs=1000)
        
        mse = eval_baseline(bl)
        mse_baseline_100k.append(mse)
        print(f"  Seed {seed}: Baseline 100K MSE = {mse*1e4:.4f} x 10^-4")
    
    # Convert to arrays
    mse_composed_neural = np.array(mse_composed_neural)
    mse_composed_sindy = np.array(mse_composed_sindy)
    mse_baseline_10k = np.array(mse_baseline_10k)
    mse_baseline_100k = np.array(mse_baseline_100k)
    
    # ============================================================
    # 8. RESULTS
    # ============================================================
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    models = {
        "Composed Neural": mse_composed_neural,
        "Composed SINDy": mse_composed_sindy,
        "Baseline 10K": mse_baseline_10k,
        "Baseline 100K": mse_baseline_100k,
    }
    
    print(f"\n{'Model':<22} {'N':>4} {'Mean (x10^-4)':>14} {'Std (x10^-4)':>13}")
    print("-" * 58)
    for name, arr in models.items():
        print(f"{name:<22} {len(arr):>4} {np.mean(arr)*1e4:>14.4f} {np.std(arr)*1e4:>13.4f}")
    
    # ============================================================
    # 9. PAIRWISE COMPARISONS
    # ============================================================
    print()
    print("=" * 70)
    print("PAIRWISE COMPARISONS (Welch's t-test, two-tailed)")
    print("=" * 70)
    
    names = list(models.keys())
    arrays = list(models.values())
    
    print(f"\n{'Comparison':<42} {'t':>8} {'p':>10} {'Sig':>5} {'d':>8} {'Effect':>10}")
    print("-" * 88)
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            t, p = welch_t(arrays[i], arrays[j])
            d = cohens_d(arrays[i], arrays[j])
            comp = f"{names[i]} vs {names[j]}"
            eff = "Large" if abs(d)>0.8 else ("Medium" if abs(d)>0.5 else ("Small" if abs(d)>0.2 else "Negligible"))
            print(f"{comp:<42} {t:>8.3f} {p:>10.6f} {sig_stars(p):>5} {d:>8.3f} {eff:>10}")
    
    # ============================================================
    # 10. PER-SEED ARRAYS (for paper and verification)
    # ============================================================
    print()
    print("=" * 70)
    print("PER-SEED MSE ARRAYS (x 10^-4) — PASTE INTO PAPER")
    print("=" * 70)
    for name, arr in models.items():
        vals = ", ".join([f"{v*1e4:.4f}" for v in arr])
        print(f"\n{name}:\n  [{vals}]")
    
    print()
    print("=" * 70)
    print("PASTE ALL OUTPUT ABOVE BACK TO THE AGENT")
    print("=" * 70)
