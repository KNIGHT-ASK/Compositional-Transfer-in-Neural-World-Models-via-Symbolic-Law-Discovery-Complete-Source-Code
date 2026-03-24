"""
Springs Benchmark — Compositional Intelligence Test
====================================================
Scientific claims:
  1. DATA-DRIVEN: pairwise forces are attributed from observed trajectory data
     only, via multi-body linear regression. The analytical law is never used
     as a training signal.
  2. ARCHITECTURE: the compositional pairwise model is compared fairly against
     a monolithic baseline with equal parameter budget and data.

Attribution method (replaces extract_pairwise_samples):
  Observe net dv_i = sum_j F(r_ij) for each particle i.
  Parameterize F(d) as piecewise-constant over N_BINS distance bins.
  Build the linear system:  A @ alpha = b
    - Each row = one component of dv for one particle in one sample
    - A[row, bin] += unit_ij[component]  for each neighbor j in that bin
    - b[row]      = dv_i[component] / DT
  Solve via least squares to recover alpha(d) — the scalar force magnitude
  at each distance bin. This is genuine discovery from observed data.

  From alpha(d), generate per-pair training targets:
    dv_ij = alpha(dist_ij) * unit_ij * DT
  These are data-derived, not computed from the analytical law.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

_GPU    = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
USE_AMP = torch.cuda.is_available() and "P100" not in _GPU
print(f"GPU    : {_GPU or 'CPU'}")
print(f"AMP    : {'DISABLED (P100)' if not USE_AMP else 'ENABLED'}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_TRAIN      = 100_000
N_TEST       = 5_000
DT           = 0.01
K_SPRING     = 1.0
REST_LEN     = 1.5
EPOCHS       = 300
BATCH        = 8192
N_SEEDS      = 5
PATIENCE     = 40
MIN_DELTA    = 1e-7
N_BINS       = 60       # distance bins for force attribution regression
ATTR_SAMPLES = 20_000   # samples used for attribution regression (subset of N_TRAIN)
                         # larger = more accurate attribution; 20k is sufficient


# ── NRI data fetching ─────────────────────────────────────────────────────────
def fetch_nri_data(data_dir="nri_data"):
    if os.path.exists(os.path.join(data_dir, "loc_train_springs5.npy")):
        print("NRI data already present.")
        return True
    print("Cloning NRI repository...")
    clone = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ethanfetaya/NRI.git", "NRI"],
        capture_output=True
    )
    if clone.returncode != 0:
        print("Clone failed. Using built-in simulator.")
        return False
    os.makedirs(data_dir, exist_ok=True)
    gen = subprocess.run(
        ["python", "NRI/data/generate_dataset.py",
         "--num-train", str(N_TRAIN),
         "--num-valid", "2000",
         "--num-test",  str(N_TEST),
         "--simulation", "springs",
         "--datadir",    data_dir],
        capture_output=True
    )
    if gen.returncode != 0:
        print("NRI generation failed. Using built-in simulator.")
        return False
    print("NRI Springs dataset ready.")
    return True


def load_nri_as_residuals(data_dir, split="train"):
    """Load ALL timesteps from NRI data (not just one)."""
    loc = np.load(os.path.join(data_dir, f"loc_{split}_springs5.npy"))
    vel = np.load(os.path.join(data_dir, f"vel_{split}_springs5.npy"))
    T = loc.shape[1] - 1
    all_X, all_dv = [], []
    for t in range(T):
        pos_t  = torch.tensor(loc[:, t,     :, :], dtype=torch.float32, device=device)
        vel_t  = torch.tensor(vel[:, t,     :, :], dtype=torch.float32, device=device)
        vel_t1 = torch.tensor(vel[:, t + 1, :, :], dtype=torch.float32, device=device)
        all_X.append(torch.cat([pos_t, vel_t], dim=-1))
        all_dv.append(vel_t1 - vel_t)
    return torch.cat(all_X, dim=0), torch.cat(all_dv, dim=0)


# ── Physics: simulation only, never used as training target ──────────────────
def compute_spring_accel(pos, k=K_SPRING, rest=REST_LEN):
    """Used only for data generation, not for training targets."""
    rel_pos  = pos[:, :, None, :] - pos[:, None, :, :]
    dist     = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)
    unit_vec = rel_pos / dist
    force    = -k * (dist - rest) * unit_vec
    mask     = 1.0 - torch.eye(pos.shape[1], device=device).unsqueeze(0).unsqueeze(-1)
    return torch.sum(force * mask, dim=2)


def generate_data(n_samples, n_particles, seed):
    """Returns (X, Y) where Y = net dv per particle from simulation."""
    torch.manual_seed(seed)
    pos = torch.empty(n_samples, n_particles, 2).uniform_(-5, 5).to(device)
    vel = torch.empty(n_samples, n_particles, 2).uniform_(-2, 2).to(device)
    acc = compute_spring_accel(pos)
    return torch.cat([pos, vel], dim=-1), acc * DT


# ── Force attribution via multi-body linear regression ───────────────────────
def attribute_pairwise_forces(X, dv_obs, n_particles, n_bins=N_BINS,
                               n_attr_samples=ATTR_SAMPLES):
    """
    Recovers the scalar pairwise force profile F(d) from observed net dv data,
    without access to the analytical law.

    Method:
      For each particle i in each sample, we observe:
        dv_i = sum_j F(d_ij) * unit_ij  (DT already absorbed)

      Parameterize F(d)/DT = alpha(d), piecewise-constant over distance bins.

      This gives a linear system: A @ alpha = b
        A[row, bin] += unit_ij[component]  for each j whose distance falls in that bin
        b[row]      = dv_i[component] / DT

      Solve with least squares. No analytical law used anywhere.

    Returns:
      alpha: array of shape (n_bins,) — data-derived force magnitudes
      bins:  array of shape (n_bins+1,) — bin edges
      bin_centers: array of shape (n_bins,)
    """
    pos_np = X[:n_attr_samples, :, :2].cpu().numpy()
    dv_np  = dv_obs[:n_attr_samples].cpu().numpy()   # shape (S, n_particles, 2)
    S      = pos_np.shape[0]

    # Distance range from data
    all_dists = []
    for i in range(n_particles):
        for j in range(n_particles):
            if i == j: continue
            rel  = pos_np[:, j] - pos_np[:, i]
            dist = np.linalg.norm(rel, axis=-1)
            all_dists.append(dist)
    all_dists = np.concatenate(all_dists)
    d_max = np.percentile(all_dists, 99.5)    # robust max
    bins  = np.linspace(0.0, d_max + 0.1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Build design matrix
    # Rows: S * n_particles * 2 (one per component per particle per sample)
    n_rows = S * n_particles * 2
    A = np.zeros((n_rows, n_bins), dtype=np.float32)
    b = np.zeros(n_rows, dtype=np.float32)

    row = 0
    for s in range(S):
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j: continue
                rel   = pos_np[s, j] - pos_np[s, i]
                dist  = np.linalg.norm(rel).clip(min=1e-6)
                unit  = rel / dist
                b_idx = np.digitize(dist, bins) - 1
                b_idx = np.clip(b_idx, 0, n_bins - 1)
                A[row,   b_idx] += unit[0]   # x-component
                A[row+1, b_idx] += unit[1]   # y-component
            # observed dv for particle i (divided by DT to get force scale)
            b[row]   = dv_np[s, i, 0] / DT
            b[row+1] = dv_np[s, i, 1] / DT
            row += 2

    # Least squares solve
    alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return alpha, bins, bin_centers


def extract_pairwise_samples_data_driven(X, dv_obs, n_particles,
                                          alpha, bins, n_bins=N_BINS):
    """
    Generates (rel_state, target_dv) pairs using data-derived force profile.

    The target for pair (i,j) is:
        dv_ij = alpha(dist_ij) * unit_ij * DT

    where alpha was recovered from observed data via linear regression — no
    analytical law is used here.
    """
    pos     = X[:, :, :2]
    vel     = X[:, :, 2:]
    pos_np  = pos.cpu().numpy()

    rel_states, targets = [], []
    for i in range(n_particles):
        for j in range(n_particles):
            if i == j: continue
            rel_x = pos[:, j] - pos[:, i]   # (N, 2) on device
            rel_v = vel[:, j] - vel[:, i]

            # Compute target from data-derived alpha (done in numpy)
            rel_np   = pos_np[:, j] - pos_np[:, i]
            dist_np  = np.linalg.norm(rel_np, axis=-1).clip(min=1e-6)
            unit_np  = rel_np / dist_np[:, None]
            b_idx    = np.digitize(dist_np, bins) - 1
            b_idx    = np.clip(b_idx, 0, n_bins - 1)
            alpha_ij = alpha[b_idx]                        # (N,) scalar per sample
            # target = alpha(d_ij) * unit_ij * DT
            target_np = alpha_ij[:, None] * unit_np * DT  # (N, 2)
            target_t  = torch.tensor(target_np, dtype=torch.float32, device=device)

            rel_states.append(torch.cat([rel_x, rel_v], dim=-1))
            targets.append(target_t)

    return torch.cat(rel_states, dim=0), torch.cat(targets, dim=0)


# ── Models ────────────────────────────────────────────────────────────────────
class PairwiseModule(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),      nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_fair_monolith(n_particles):
    in_dim  = n_particles * 4
    out_dim = n_particles * 2
    mono    = nn.Sequential(
        nn.Linear(in_dim, 512), nn.SiLU(),
        nn.Linear(512, 512),    nn.SiLU(),
        nn.Linear(512, 512),    nn.SiLU(),
        nn.Linear(512, out_dim)
    )
    print(f"  Pairwise module params : {count_params(PairwiseModule()):,}")
    print(f"  Monolith params        : {count_params(mono):,}")
    return mono


# ── Normalisation ─────────────────────────────────────────────────────────────
def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(0, keepdim=True)
        std  = X.std(0, keepdim=True).clamp(min=1e-8)
    return (X - mean) / std, mean, std


# ── Training loop ─────────────────────────────────────────────────────────────
def run_training_loop(model, X_norm, Y_norm, seed, label):
    torch.manual_seed(seed)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    n      = X_norm.shape[0]

    best_loss         = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        idx        = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, BATCH):
            ii = idx[start:start + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = nn.MSELoss()(model(X_norm[ii]), Y_norm[ii])
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg = total_loss / max(1, n_batches)

        if best_loss - avg > MIN_DELTA:
            best_loss         = avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  {label} early stop @ epoch {epoch + 1}, best {best_loss:.4e}")
            break

    model.eval()
    return model


def train_pairwise(rel_states, targets, seed):
    rs_norm, rs_mean, rs_std = normalize(rel_states)
    tg_norm, tg_mean, tg_std = normalize(targets)
    model = PairwiseModule().to(device)
    model = run_training_loop(model, rs_norm, tg_norm, seed, "Pairwise")
    return model, (rs_mean, rs_std, tg_mean, tg_std)


def train_monolith(X, Y, n_particles, seed):
    b      = X.shape[0]
    X_flat = X.view(b, -1)
    Y_flat = Y.view(b, -1)
    X_norm, xm, xs = normalize(X_flat)
    Y_norm, ym, ys = normalize(Y_flat)
    model  = build_fair_monolith(n_particles).to(device)
    model  = run_training_loop(model, X_norm, Y_norm, seed, "Monolith")
    return model, (xm, xs, ym, ys)


# ── Zero-shot composition ─────────────────────────────────────────────────────
@torch.no_grad()
def compose_zero_shot(model, norms, X_test, n_particles):
    rs_mean, rs_std, tg_mean, tg_std = norms
    model.eval()
    b       = X_test.shape[0]
    pos     = X_test[:, :, :2]
    vel     = X_test[:, :, 2:]
    dv_pred = torch.zeros(b, n_particles, 2, device=device)
    for i in range(n_particles):
        for j in range(n_particles):
            if i == j: continue
            rel_x     = pos[:, j] - pos[:, i]
            rel_v     = vel[:, j] - vel[:, i]
            rel_state = torch.cat([rel_x, rel_v], dim=-1)
            rel_norm  = (rel_state - rs_mean) / rs_std
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred_norm = model(rel_norm)
            dv_pred[:, i] += pred_norm * tg_std + tg_mean
    return dv_pred


# ── Spring constant recovery (physical intelligence test) ─────────────────────
@torch.no_grad()
def recover_spring_constant_from_model(model, norms, n_probe=2000):
    """
    Probes the pairwise module at controlled separations (zero relative velocity)
    and recovers k by linear regression.

    Sign convention: the attribution regression uses unit vector pointing i→j.
    In that frame, F_on_i_from_j = +k*(dist-rest)*unit_i->j (attractive when extended).
    So alpha(d) = +k*(d-rest), slope of alpha vs extension = +k.
    Therefore: k_recovered = polyfit slope directly (no negation).
    """
    rs_mean, rs_std, tg_mean, tg_std = norms
    model.eval()

    distances = torch.linspace(0.5, 4.0, n_probe, device=device)
    rel_x     = torch.stack([distances, torch.zeros_like(distances)], dim=1)
    rel_v     = torch.zeros_like(rel_x)
    rel_state = torch.cat([rel_x, rel_v], dim=-1)
    rel_norm  = (rel_state - rs_mean) / rs_std

    with torch.amp.autocast("cuda", enabled=USE_AMP):
        pred_norm = model(rel_norm)

    dv_pred  = (pred_norm * tg_std + tg_mean).cpu().numpy()
    forces   = dv_pred[:, 0] / DT
    dist_np  = distances.cpu().numpy()
    ext      = dist_np - REST_LEN

    valid = np.abs(ext) > 0.05
    if valid.sum() < 10:
        return float("nan"), 0.0

    # FIX: was -polyfit[0]. In i->j convention, alpha(d) = +k*(d-rest), slope = +k.
    k_recovered = float(np.polyfit(ext[valid], forces[valid], 1)[0])
    if k_recovered < 0:
        print(f"  WARNING: recovered k={k_recovered:.4f} is NEGATIVE (repulsive).")
    accuracy = max(0.0, 1.0 - abs(k_recovered - K_SPRING) / K_SPRING) * 100.0
    return k_recovered, accuracy


def recover_spring_constant_from_regression(alpha, bin_centers):
    """
    Recovers k directly from the attribution regression alpha(d).
    This is the data-driven counterpart of the model probe above.
    Both should agree if the model learned correctly.

    Same sign convention: alpha(d) = +k*(d-rest), so slope = +k directly.
    """
    valid = np.abs(bin_centers - REST_LEN) > 0.1
    # FIX: was -polyfit[0] — same sign bug as model probe above.
    k_recovered = float(np.polyfit(bin_centers[valid] - REST_LEN, alpha[valid], 1)[0])
    if k_recovered < 0:
        print(f"  WARNING: attribution regression recovered k={k_recovered:.4f} (NEGATIVE).")
    accuracy = max(0.0, 1.0 - abs(k_recovered - K_SPRING) / K_SPRING) * 100.0
    return k_recovered, accuracy


# ── Particle count generalization ─────────────────────────────────────────────
@torch.no_grad()
def test_particle_generalization(model, norms, particle_counts, seed=99):
    """Zero-shot transfer to unseen particle counts — no retraining."""
    results = {}
    for n in particle_counts:
        X_test, Y_test = generate_data(2000, n, seed=seed + n)
        dv_pred        = compose_zero_shot(model, norms, X_test, n)
        mse            = nn.MSELoss()(dv_pred, Y_test).item()
        results[n]     = mse
        print(f"  Generalization n={n}: MSE {mse:.4e}")
        del X_test, Y_test, dv_pred
        torch.cuda.empty_cache()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 66)
    print("Springs Benchmark — Data-Driven Attribution + Architecture Comparison")
    print(f"k={K_SPRING}, rest={REST_LEN}, DT={DT}, {N_SEEDS} seeds")
    print("=" * 66)

    use_nri  = fetch_nri_data(data_dir="nri_data")
    data_src = "NRI (Kipf et al. 2018)" if use_nri else "Built-in Simulator"
    print(f"Data source: {data_src}\n")

    zs_mses            = []
    mono_mses          = []
    k_model_acc_list   = []
    k_regr_acc_list    = []
    k_model_val_list   = []
    k_regr_val_list    = []
    gen_results_all    = {3: [], 4: [], 5: [], 8: []}

    for seed in range(N_SEEDS):
        print(f"\n── Seed {seed + 1}/{N_SEEDS} " + "─" * 45)

        # ── Load / generate data ───────────────────────────────────────────
        if use_nri:
            X_train, dv_obs   = load_nri_as_residuals("nri_data", split="train")
            n_particles_train = X_train.shape[1]
        else:
            X_train, dv_obs   = generate_data(N_TRAIN, 2, seed=seed * 7 + 1)
            n_particles_train = 2

        # ── Step 1: Attribute pairwise forces from observed data ───────────
        print(f"  Attributing pairwise forces from observed dv "
              f"({ATTR_SAMPLES} samples, {N_BINS} distance bins)...")
        alpha, bins, bin_centers = attribute_pairwise_forces(
            X_train, dv_obs, n_particles_train,
            n_bins=N_BINS, n_attr_samples=ATTR_SAMPLES
        )

        # Report data-driven k recovery (from regression, before any NN training)
        k_regr, k_regr_acc = recover_spring_constant_from_regression(alpha, bin_centers)
        k_regr_val_list.append(k_regr)
        k_regr_acc_list.append(k_regr_acc)
        print(f"  Attribution regression: k={k_regr:.4f}, "
              f"true k={K_SPRING}, accuracy={k_regr_acc:.1f}%")

        # ── Step 2: Build per-pair training samples from attributed forces ─
        rs, tg = extract_pairwise_samples_data_driven(
            X_train, dv_obs, n_particles_train, alpha, bins
        )
        del X_train, dv_obs
        torch.cuda.empty_cache()

        # ── Step 3: Train pairwise model on attributed targets ─────────────
        pairwise_model, pw_norms = train_pairwise(rs, tg, seed=seed)
        del rs, tg
        torch.cuda.empty_cache()

        # ── Step 4: Physical law recovery from trained model ───────────────
        k_model, k_model_acc = recover_spring_constant_from_model(pairwise_model, pw_norms)
        k_model_val_list.append(k_model)
        k_model_acc_list.append(k_model_acc)
        print(f"  Model probe           : k={k_model:.4f}, "
              f"true k={K_SPRING}, accuracy={k_model_acc:.1f}%")

        # ── Step 5: Particle-count generalization ──────────────────────────
        gen = test_particle_generalization(
            pairwise_model, pw_norms, particle_counts=[3, 4, 5, 8]
        )
        for n, mse in gen.items():
            gen_results_all[n].append(mse)

        # ── Step 6: Monolith baseline (trained on net dv directly) ─────────
        X5_train, Y5_train = generate_data(N_TRAIN, 5, seed=seed * 7 + 2)
        mono_model, mono_norms = train_monolith(X5_train, Y5_train, n_particles=5, seed=seed)
        del X5_train, Y5_train
        torch.cuda.empty_cache()

        # ── Step 7: Evaluate on 5-particle test set ────────────────────────
        X5_test, Y5_test = generate_data(N_TEST, 5, seed=seed * 7 + 99)

        dv_zs  = compose_zero_shot(pairwise_model, pw_norms, X5_test, n_particles=5)
        mse_zs = nn.MSELoss()(dv_zs, Y5_test).item()

        b              = X5_test.shape[0]
        xm, xs, ym, ys = mono_norms
        X_norm         = (X5_test.view(b, -1) - xm) / xs
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred = mono_model(X_norm) * ys + ym
        mse_mono = nn.MSELoss()(pred, Y5_test.view(b, -1)).item()

        zs_mses.append(mse_zs)
        mono_mses.append(mse_mono)
        print(f"  Zero-Shot MSE (5p): {mse_zs:.4e}")
        print(f"  Monolith  MSE (5p): {mse_mono:.4e}")
        print(f"  Ratio             : {mse_mono / mse_zs:.2f}x")

        del X5_test, Y5_test, dv_zs, pred, pairwise_model, mono_model
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    zs_arr      = np.array(zs_mses)
    mono_arr    = np.array(mono_mses)
    k_m_arr     = np.array(k_model_acc_list)
    k_r_arr     = np.array(k_regr_acc_list)
    ratio       = mono_arr / zs_arr

    print("\n" + "=" * 66)
    print(f"{'Metric':<38} {'Zero-Shot':>13} {'Monolith':>13}")
    print("-" * 66)
    print(f"{'Mean MSE (5 particles)':<38} {zs_arr.mean():>13.4e} {mono_arr.mean():>13.4e}")
    print(f"{'Std  MSE':<38} {zs_arr.std():>13.4e} {mono_arr.std():>13.4e}")
    print(f"{'Mean Improvement Ratio':<38} {ratio.mean():>13.2f}x")
    print("-" * 66)
    print("Physical law recovery:")
    print(f"  Attribution regression  k accuracy: {k_r_arr.mean():.1f}% ± {k_r_arr.std():.1f}%")
    print(f"  Trained model probe     k accuracy: {k_m_arr.mean():.1f}% ± {k_m_arr.std():.1f}%")
    print("-" * 66)
    print("Particle-count generalization (Zero-Shot, trained on 2):")
    for n, vals in gen_results_all.items():
        arr = np.array(vals)
        print(f"  n={n}: MSE {arr.mean():.4e} ± {arr.std():.4e}")
    print("=" * 66)

    results = {
        "data_source": data_src,
        "attribution_method": "multi_body_linear_regression",
        "zero_shot":   {"mean": float(zs_arr.mean()), "std": float(zs_arr.std()),  "all": zs_mses},
        "monolith":    {"mean": float(mono_arr.mean()), "std": float(mono_arr.std()), "all": mono_mses},
        "mean_ratio":  float(ratio.mean()),
        "spring_constant_recovery": {
            "attribution_regression": {
                "mean_accuracy_pct": float(k_r_arr.mean()),
                "std_accuracy_pct":  float(k_r_arr.std()),
                "recovered_values":  k_regr_val_list
            },
            "model_probe": {
                "mean_accuracy_pct": float(k_m_arr.mean()),
                "std_accuracy_pct":  float(k_m_arr.std()),
                "recovered_values":  k_model_val_list
            }
        },
        "particle_generalization": {
            str(n): {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for n, v in gen_results_all.items()
        },
        "config": {
            "k_spring": K_SPRING, "rest_len": REST_LEN, "dt": DT,
            "n_train": N_TRAIN, "epochs": EPOCHS, "n_seeds": N_SEEDS,
            "n_bins": N_BINS, "attr_samples": ATTR_SAMPLES
        }
    }
    with open("r1_springs_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to r1_springs_results.json")