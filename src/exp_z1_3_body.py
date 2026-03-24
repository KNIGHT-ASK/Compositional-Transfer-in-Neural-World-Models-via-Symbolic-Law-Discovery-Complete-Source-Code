import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "CUDA GPU required."
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# ── P100 check: AMP (float16) is slow on P100 — disable it automatically ─────
_GPU = torch.cuda.get_device_name(0)
USE_AMP = "P100" not in _GPU  # P100 has no fast FP16 tensor cores
print(f"GPU : {_GPU}")
print(f"AMP : {'DISABLED (P100 detected — FP32 is faster here)' if not USE_AMP else 'ENABLED'}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
G        = 1.0
DT       = 0.02
EPS_SFT  = 0.001
SCALE_K  = 0.01
N_BRAINS = 8
N_TRAIN  = 500_000
EPOCHS   = 400          # per-brain epochs (unchanged)
MONO_EPOCHS = 800       # FIX: was EPOCHS * N_BRAINS = 400 * 8 = 3200 !
BATCH    = 65536
MASSES   = np.array([100.0, 1.0, 0.01])
N_STEPS  = 5000


# ── Symlog / inv-symlog ───────────────────────────────────────────────────────
def symlog(x):
    return torch.sign(x) * torch.log(1.0 + torch.abs(x) / SCALE_K)


def inv_symlog(y):
    return torch.sign(y) * SCALE_K * (torch.exp(torch.abs(y).clamp(max=20.0)) - 1.0)


def rand_uniform(n, low, high):
    return torch.empty(n, dtype=torch.float32).uniform_(low, high).to(device)


def rand_normal(n):
    return torch.empty(n, dtype=torch.float32).normal_().to(device)


# ── Dataset generation ────────────────────────────────────────────────────────
def make_pairwise_data(n, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        log_r = rand_uniform(n, np.log(0.05), np.log(50.0))
        r_mag = torch.exp(log_r)
        phi   = rand_uniform(n, 0.0, 2 * np.pi)
        cos_t = rand_uniform(n, -1.0, 1.0)
        sin_t = torch.sqrt((1 - cos_t**2).clamp(min=0))
        rx = r_mag * sin_t * torch.cos(phi)
        ry = r_mag * sin_t * torch.sin(phi)
        rz = r_mag * cos_t
        vx = rand_normal(n)
        vy = rand_normal(n)
        vz = rand_normal(n)
        rel_state = torch.stack([rx, ry, rz, vx, vy, vz], dim=1)
        r_vec   = rel_state[:, :3]
        dist_sq = torch.sum(r_vec**2, dim=1, keepdim=True) + EPS_SFT**2
        accel   = -G * r_vec / (dist_sq**1.5)
        return rel_state, symlog(accel)


def make_monolith_data(n, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        positions = []
        for _ in range(3):
            log_r = rand_uniform(n, np.log(0.1), np.log(20.0))
            r_mag = torch.exp(log_r)
            phi   = rand_uniform(n, 0.0, 2 * np.pi)
            cos_t = rand_uniform(n, -1.0, 1.0)
            sin_t = torch.sqrt((1 - cos_t**2).clamp(min=0))
            px = r_mag * sin_t * torch.cos(phi)
            py = r_mag * sin_t * torch.sin(phi)
            pz = r_mag * cos_t
            positions.append(torch.stack([px, py, pz], dim=1))

        velocities = [torch.stack([rand_normal(n), rand_normal(n), rand_normal(n)], dim=1)
                      for _ in range(3)]

        m = torch.tensor(MASSES, dtype=torch.float32, device=device)

        full_state = torch.cat([
            torch.cat([positions[i], velocities[i]], dim=1) for i in range(3)
        ], dim=1)

        accels = torch.zeros(n, 9, device=device)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r_vec   = positions[j] - positions[i]
                dist_sq = torch.sum(r_vec**2, dim=1, keepdim=True) + EPS_SFT**2
                accels[:, i*3:i*3+3] += G * m[j] * r_vec / (dist_sq**1.5)

        return full_state, symlog(accels)


# ── Model ─────────────────────────────────────────────────────────────────────
class Brain(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.Tanh(),
            nn.Linear(512, 512),    nn.Tanh(),
            nn.Linear(512, 512),    nn.Tanh(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(S, Y, seed, name, epochs=EPOCHS):
    torch.manual_seed(seed)
    in_dim, out_dim = S.shape[1], Y.shape[1]
    model = Brain(in_dim, out_dim).to(device)

    # Normalise once — tensors stay on GPU the whole time
    Sm, Ss = S.mean(0), S.std(0) + 1e-8
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-8
    Sn = (S - Sm) / Ss
    Yn = (Y - Ym) / Ys

    opt    = optim.Adam(model.parameters(), lr=1e-3)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    n      = S.shape[0]

    for epoch in range(epochs):
        idx        = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n, BATCH):
            ii      = idx[start:start + BATCH]
            bx, by  = Sn[ii], Yn[ii]
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = nn.HuberLoss()(model(bx), by)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches  += 1

        sched.step()

        if epoch % 25 == 0:
            avg = total_loss / max(1, n_batches)
            print(f"  {name} | Epoch {epoch:4d}/{epochs} | Loss {avg:.8f}")

    model.eval()
    norms = (Sm.cpu(), Ss.cpu(), Ym.cpu(), Ys.cpu())
    return model.cpu(), norms


# ── Inference agents ──────────────────────────────────────────────────────────
class PairwiseEnsemble:
    """
    Zero-shot agent. Each brain predicts unit-mass gravitational acceleration
    from a 6D relative state (Δr, Δv). Output is scaled by the known source
    mass — the only required external input. No oracle trajectory info used.
    """
    def __init__(self, trained_models):
        self.brains = []
        for model, (Sm, Ss, Ym, Ys) in trained_models:
            self.brains.append({
                "model": model.to(device).eval(),
                "Sm": Sm.to(device), "Ss": Ss.to(device),
                "Ym": Ym.to(device), "Ys": Ys.to(device),
            })

    @torch.no_grad()
    def get_pairwise_accel(self, rel_state_6d):
        preds = []
        for b in self.brains:
            sn = (rel_state_6d - b["Sm"]) / b["Ss"]
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                yn = b["model"](sn)
            preds.append(yn * b["Ys"] + b["Ym"])
        return inv_symlog(torch.stack(preds).mean(0))

    def compute_accelerations(self, s_np):
        accels = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                rel = np.concatenate([s_np[j, :3] - s_np[i, :3],
                                       s_np[j, 3:6] - s_np[i, 3:6]])
                inp  = torch.tensor(rel, dtype=torch.float32, device=device).unsqueeze(0)
                pred = self.get_pairwise_accel(inp).cpu().numpy().squeeze()
                accels[i] += pred * MASSES[j]
        return accels


class MonolithAgent:
    def __init__(self, model, norms):
        Sm, Ss, Ym, Ys = norms
        self.model = model.to(device).eval()
        self.Sm = Sm.to(device)
        self.Ss = Ss.to(device)
        self.Ym = Ym.to(device)
        self.Ys = Ys.to(device)

    @torch.no_grad()
    def compute_accelerations(self, s_np):
        inp = torch.tensor(s_np.reshape(1, 18), dtype=torch.float32, device=device)
        sn  = (inp - self.Sm) / self.Ss
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            yn = self.model(sn)
        y = yn * self.Ys + self.Ym
        return inv_symlog(y).cpu().numpy().reshape(3, 3)


# ── Physics helpers ───────────────────────────────────────────────────────────
def make_initial_conditions():
    v_planet = np.sqrt(G * MASSES[0] / 10.0)
    v_moon   = np.sqrt(G * MASSES[1] / 0.5)
    s = np.array([
        [0.0,  0.0, 0.0,  0.0,                   0.0, 0.0],
        [10.0, 0.0, 0.0,  0.0,           v_planet,    0.0],
        [10.5, 0.0, 0.0,  0.0, v_planet + v_moon,      0.0],
    ], dtype=np.float64)
    com_v = np.sum(s[:, 3:6] * MASSES[:, None], axis=0) / MASSES.sum()
    s[:, 3:6] -= com_v
    return s


def exact_accelerations(s):
    a = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            rv      = s[j, :3] - s[i, :3]
            dist_sq = np.dot(rv, rv) + EPS_SFT**2
            a[i]   += G * MASSES[j] * rv / dist_sq**1.5
    return a


def verlet_step(s, accel_fn):
    a1  = accel_fn(s)
    vh  = s[:, 3:6] + 0.5 * a1 * DT
    s2  = s.copy()
    s2[:, :3]  = s[:, :3] + vh * DT
    s2[:, 3:6] = vh
    a2  = accel_fn(s2)
    s2[:, 3:6] = vh + 0.5 * a2 * DT
    return s2


def total_energy(s):
    KE = 0.5 * np.sum(MASSES[:, None] * s[:, 3:6]**2)
    PE = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r   = np.linalg.norm(s[i, :3] - s[j, :3])
            PE -= G * MASSES[i] * MASSES[j] / (r + EPS_SFT)
    return KE + PE


def total_momentum(s):
    L = np.sum(MASSES[:, None] * np.cross(s[:, :3], s[:, 3:6]), axis=0)
    return np.linalg.norm(L)


def run_rollout(agent, n_steps=N_STEPS):
    s0   = make_initial_conditions()
    s_gt = s0.copy()
    s_ag = s0.copy()
    E0   = total_energy(s0)
    L0   = total_momentum(s0)
    energy_drifts   = np.zeros(n_steps)
    momentum_drifts = np.zeros(n_steps)
    mse_total       = 0.0
    for step in range(n_steps):
        s_gt = verlet_step(s_gt, exact_accelerations)
        s_ag = verlet_step(s_ag, agent.compute_accelerations)
        energy_drifts[step]   = abs(total_energy(s_ag) - E0) / (abs(E0) + 1e-12)
        momentum_drifts[step] = abs(total_momentum(s_ag) - L0) / (L0 + 1e-12)
        mse_total += np.mean((s_ag - s_gt)**2)
    return {
        "energy_drift":        energy_drifts,
        "momentum_drift":      momentum_drifts,
        "trajectory_mse":      (mse_total / n_steps) * 1e4,
        "mean_energy_drift":   float(np.mean(energy_drifts)),
        "mean_momentum_drift": float(np.mean(momentum_drifts)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(res_zs, res_mono):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Physical Manifold Persistence: Modular Zero-Shot vs Monolith", fontsize=13)

    axes[0].plot(res_zs["energy_drift"],   label="Modular Zero-Shot (Ours)", color="royalblue", lw=1.2)
    axes[0].plot(res_mono["energy_drift"], label="Monolithic Baseline",      color="crimson",   lw=1.0, alpha=0.75)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative Energy Drift |ΔE / E₀|")
    axes[0].set_xlabel("Integration Step")
    axes[0].set_title("Energy Conservation")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(res_zs["momentum_drift"],   label="Modular Zero-Shot (Ours)", color="royalblue", lw=1.2)
    axes[1].plot(res_mono["momentum_drift"], label="Monolithic Baseline",      color="crimson",   lw=1.0, alpha=0.75)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Relative Momentum Drift |ΔL / L₀|")
    axes[1].set_xlabel("Integration Step")
    axes[1].set_title("Angular Momentum Conservation")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("threebody_manifold_persistence.png", dpi=150)
    print("Figure saved: threebody_manifold_persistence.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"Training zero-shot pairwise ensemble")
    print(f"  {N_BRAINS} brains  x  {N_TRAIN:,} samples  x  {EPOCHS} epochs each")
    print("=" * 60)
    ensemble_data = []
    for i in range(N_BRAINS):
        S, Y = make_pairwise_data(N_TRAIN, seed=i * 17 + 3)
        result = train_model(S, Y, seed=i, name=f"Brain-{i}", epochs=EPOCHS)
        ensemble_data.append(result)
        del S, Y
        torch.cuda.empty_cache()

    ensemble_agent = PairwiseEnsemble(ensemble_data)

    print()
    print("=" * 60)
    print(f"Training monolithic baseline")
    print(f"  {N_BRAINS * N_TRAIN:,} samples  x  {MONO_EPOCHS} epochs")
    print(f"  (BUG FIX: was epochs=EPOCHS*N_BRAINS = {EPOCHS}*{N_BRAINS} = {EPOCHS*N_BRAINS})")
    print("=" * 60)
    S_m, Y_m = make_monolith_data(N_BRAINS * N_TRAIN, seed=99)
    mono_model, mono_norms = train_model(
        S_m, Y_m, seed=42, name="Monolith",
        epochs=MONO_EPOCHS   # FIX: explicit 800, not EPOCHS * N_BRAINS
    )
    del S_m, Y_m
    torch.cuda.empty_cache()

    monolith_agent = MonolithAgent(mono_model, mono_norms)

    print(f"\nRunning rollouts ({N_STEPS} steps each)...")
    res_zs   = run_rollout(ensemble_agent)
    res_mono = run_rollout(monolith_agent)

    header = f"{'Metric':<25} {'Zero-Shot':>15} {'Monolith':>15}"
    print(f"\n{header}")
    print("-" * len(header))
    print(f"{'Mean Energy Drift':<25} {res_zs['mean_energy_drift']:>15.6e} {res_mono['mean_energy_drift']:>15.6e}")
    print(f"{'Mean Momentum Drift':<25} {res_zs['mean_momentum_drift']:>15.6e} {res_mono['mean_momentum_drift']:>15.6e}")
    print(f"{'Trajectory MSE (x1e4)':<25} {res_zs['trajectory_mse']:>15.4f} {res_mono['trajectory_mse']:>15.4f}")

    plot_results(res_zs, res_mono)