"""
Experiment Q: Hierarchical Decomposition of Gravity
Decomposes Environment A (gravity+floor) into:
  - Brain_gravity_pure: trained on gravity WITHOUT floor
  - Brain_collision: trained on floor bounce WITHOUT gravity
Then composes them and extracts constants via SINDy.
Expected recoveries: g ~99%+, epsilon (restitution) as new constant.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

dt = 0.02
g = 9.8
mass = 1.0
epsilon = 0.8  # coefficient of restitution (was 0.9 in BouncingBallSim — verify your code)

# --- Pure Gravity Environment (no floor) ---
class PureGravitySim:
    def __init__(self, y0=10.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g + F / mass
        self.v += a * dt
        self.y += self.v * dt
        # NO floor bounce — y can go negative freely
        return self.y, self.v

# --- Collision-Only Environment (floor at y=0, no gravity) ---
class CollisionOnlySim:
    def __init__(self, y0=2.0, v0=-3.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = F / mass  # no gravity, only external force
        self.v += a * dt
        self.y += self.v * dt
        if self.y < 0:
            self.y = -self.y
            self.v = -epsilon * self.v
        return self.y, self.v

# --- Original Full Environment (gravity + floor, for verification) ---
class FullGravitySim:
    def __init__(self, y0=10.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g + F / mass
        self.v += a * dt
        self.y += self.v * dt
        if self.y < 0:
            self.y = -self.y
            self.v = -epsilon * self.v
        return self.y, self.v

# --- Network ---
class Brain(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

# --- Data collection ---
def collect_data(sim_class, n, sim_kwargs=None, reset_every=50):
    if sim_kwargs is None:
        sim_kwargs = {}
    X, Y = [], []
    sim = sim_class(**sim_kwargs)
    for i in range(n):
        if i % reset_every == 0:
            sim = sim_class(**sim_kwargs)
            if hasattr(sim, 'y'):
                sim.y = np.random.uniform(0.5, 15.0)
            sim.v = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def collect_collision_data(n, reset_every=30):
    """Collision data needs careful collection: frequent resets near floor with downward velocity."""
    X, Y = [], []
    sim = CollisionOnlySim()
    for i in range(n):
        if i % reset_every == 0:
            # Reset near floor with mix of velocities to get both collision and non-collision transitions
            sim = CollisionOnlySim(
                y0=np.random.uniform(0.1, 3.0),
                v0=np.random.uniform(-8.0, 5.0)  # bias toward negative v to hit floor
            )
        F = np.random.uniform(-2.0, 2.0)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# --- Training ---
def train(model, X, Y, epochs=300, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                total_loss = criterion(model(X), Y).item()
            print(f"  Epoch {ep+1}/{epochs}, Loss: {total_loss:.8f}")
    return model

# --- SINDy extraction ---
def sindy_extract(model, name, y_range=(-5, 15), v_range=(-5, 5), n_samples=10000):
    """Extract symbolic equation from network using LASSO on polynomial features."""
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import PolynomialFeatures

    ys = np.random.uniform(y_range[0], y_range[1], n_samples)
    vs = np.random.uniform(v_range[0], v_range[1], n_samples)
    Fs = np.zeros(n_samples)  # F=0 to isolate intrinsic dynamics

    inputs = torch.tensor(np.column_stack([ys, vs, Fs]), dtype=torch.float32)
    with torch.no_grad():
        preds = model(inputs).numpy()

    dv = preds[:, 1]  # velocity residual
    dy = preds[:, 0]  # position residual

    features = np.column_stack([ys, vs])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Phi = poly.fit_transform(features)
    names = poly.get_feature_names_out(['y', 'v'])

    lasso = Lasso(alpha=1e-4, max_iter=10000, fit_intercept=False)
    lasso.fit(Phi, dv)

    print(f"\n  SINDy Results for {name} (Δv):")
    for coeff, fname in zip(lasso.coef_, names):
        if abs(coeff) > 1e-5:
            print(f"    {coeff:+.6f} * {fname}")

    # Also extract Δy
    lasso_dy = Lasso(alpha=1e-4, max_iter=10000, fit_intercept=False)
    lasso_dy.fit(Phi, dy)
    print(f"  SINDy Results for {name} (Δy):")
    for coeff, fname in zip(lasso_dy.coef_, names):
        if abs(coeff) > 1e-5:
            print(f"    {coeff:+.6f} * {fname}")

    return lasso.coef_, names

# --- Restitution coefficient extraction ---
def extract_restitution(model):
    """Probe collision network at y≈0 with known velocities to recover epsilon."""
    print("\n  Probing collision network for restitution coefficient:")
    test_vs = [-8, -5, -3, -1, 1, 3, 5]
    epsilons = []
    for v in test_vs:
        # At y just above 0 with downward velocity: should see bounce
        inp = torch.tensor([[0.05, float(v), 0.0]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(inp).numpy()[0]
        dv = pred[1]
        # No-bounce would give dv ≈ 0 (no gravity in collision env)
        # Bounce gives dv ≈ -(1+eps)*v - 0 = -(1+eps)*v
        # So eps = -dv/v - 1 (only valid for v < 0 hitting floor)
        if v < -0.5:  # only downward velocities hit the floor
            eps_recovered = -dv / v - 1
            epsilons.append(eps_recovered)
            print(f"    v={v:+.1f}: Δv={dv:+.4f}, recovered ε={eps_recovered:.4f}")
        else:
            print(f"    v={v:+.1f}: Δv={dv:+.4f} (no collision expected)")

    if epsilons:
        mean_eps = np.mean(epsilons)
        print(f"\n  Recovered ε = {mean_eps:.4f} (true: {epsilon:.4f}, accuracy: {mean_eps/epsilon*100:.1f}%)")
    return epsilons

# --- Composition evaluation ---
def evaluate_composition(brain_pure, brain_coll, n_test=5000):
    """Test composed (pure gravity + collision) on full gravity environment."""
    errors_composed = []
    errors_original_sindy = []

    for _ in range(n_test):
        y = np.random.uniform(0.1, 15.0)
        v = np.random.uniform(-8.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)

        # Ground truth
        sim = FullGravitySim(y0=y, v0=v)
        y_true, v_true = sim.step(F)
        true_dy = y_true - y
        true_dv = v_true - v

        # Composed prediction
        inp = torch.tensor([[y, v, F]], dtype=torch.float32)
        with torch.no_grad():
            res_grav = brain_pure(inp).numpy()[0]
            res_coll = brain_coll(inp).numpy()[0]

        # Compose: sum residuals, subtract shared kinematic term once
        pred_dv = res_grav[1] + res_coll[1]
        pred_dy = res_grav[0] + res_coll[0] - v * dt  # subtract double-counted v*dt

        err = (pred_dy - true_dy)**2 + (pred_dv - true_dv)**2
        errors_composed.append(err)

        # Compare with old single-constant SINDy (g=9.03)
        sindy_dv = -0.181 + F * dt  # old contaminated constant
        sindy_dy = v * dt
        err_sindy = (sindy_dy - true_dy)**2 + (sindy_dv - true_dv)**2
        errors_original_sindy.append(err_sindy)

    comp_mse = np.mean(errors_composed)
    sindy_mse = np.mean(errors_original_sindy)
    print(f"\n  Composed (pure_grav + collision) MSE: {comp_mse*1e4:.4f} x 10^-4")
    print(f"  Old SINDy (contaminated g=9.03)  MSE: {sindy_mse*1e4:.4f} x 10^-4")
    print(f"  Improvement: {sindy_mse/comp_mse:.1f}x")
    return comp_mse, sindy_mse

# --- Main ---
if __name__ == "__main__":
    print("Experiment Q: Hierarchical Gravity Decomposition")
    print("Pure Gravity (no floor) + Collision (floor, no gravity)")
    print()

    N_TRAIN = 100000
    N_SEEDS = 5

    all_g_recovered = []
    all_eps_recovered = []
    all_comp_mse = []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)

        # Train pure gravity network
        print("  Training Brain_gravity_pure (no floor)...")
        X_grav, Y_grav = collect_data(PureGravitySim, N_TRAIN,
                                       sim_kwargs={'y0': 10.0}, reset_every=50)
        brain_pure = Brain()
        train(brain_pure, X_grav, Y_grav, epochs=300)

        # Train collision network
        print("  Training Brain_collision (floor only, no gravity)...")
        X_coll, Y_coll = collect_collision_data(N_TRAIN, reset_every=30)
        brain_coll = Brain()
        train(brain_coll, X_coll, Y_coll, epochs=300)

        # SINDy on pure gravity (should recover exact g)
        coeffs, names = sindy_extract(brain_pure, "Brain_gravity_pure",
                                       y_range=(-5, 15))

        # Find the constant term (the gravity coefficient)
        const_idx = list(names).index('1')
        g_coeff = coeffs[const_idx]
        g_recovered = -g_coeff / dt
        all_g_recovered.append(g_recovered)
        print(f"\n  Recovered g = {g_recovered:.4f} (true: {g}, accuracy: {g_recovered/g*100:.1f}%)")

        # Extract restitution from collision network
        epsilons = extract_restitution(brain_coll)
        if epsilons:
            all_eps_recovered.append(np.mean(epsilons))

        # Evaluate composition
        comp_mse, _ = evaluate_composition(brain_pure, brain_coll)
        all_comp_mse.append(comp_mse)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    g_arr = np.array(all_g_recovered)
    eps_arr = np.array(all_eps_recovered)
    mse_arr = np.array(all_comp_mse)

    print(f"\nGravity constant recovery ({N_SEEDS} seeds):")
    print(f"  Old (contaminated): g = 9.03 (92.1%)")
    print(f"  New (decomposed):   g = {np.mean(g_arr):.4f} +/- {np.std(g_arr):.4f} ({np.mean(g_arr)/g*100:.1f}%)")

    if len(eps_arr) > 0:
        print(f"\nRestitution coefficient (NEW constant discovered):")
        print(f"  Recovered: eps = {np.mean(eps_arr):.4f} +/- {np.std(eps_arr):.4f} (true: {epsilon})")

    print(f"\nComposition MSE ({N_SEEDS} seeds): {np.mean(mse_arr)*1e4:.4f} +/- {np.std(mse_arr)*1e4:.4f} x 10^-4")
    print(f"\nPer-seed g values: {[f'{v:.4f}' for v in g_arr]}")
    print(f"Per-seed eps values: {[f'{v:.4f}' for v in eps_arr]}")
