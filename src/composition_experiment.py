import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. THE ARCHITECTURE
class Brain1_v2(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        super(Brain1_v2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# 2. ENVIRONMENT C: THE TESTBED
class CombinedSim:
    def __init__(self, y0=0.0, v0=0.0, g=9.8, k=2.0, m=1.0, dt=0.02):
        self.y = y0
        self.v = v0
        self.g = g
        self.k = k
        self.m = m
        self.dt = dt

    def step(self, F=0.0):
        a = -self.g - (self.k * self.y / self.m) + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        return np.array([self.y, self.v], dtype=np.float32)

# 3. BASELINE TRAINING HELPER
def train_baseline(num_steps):
    print(f"Training baseline from scratch for {num_steps} transitions...")
    # Collection phase for Environment C
    states = np.zeros((num_steps, 2), dtype=np.float32)
    actions = np.zeros((num_steps, 1), dtype=np.float32)
    next_states = np.zeros((num_steps, 2), dtype=np.float32)
    
    sim = CombinedSim(y0=np.random.uniform(-10.0, 0.0), v0=np.random.uniform(-5.0, 5.0))
    curr_state = np.array([sim.y, sim.v], dtype=np.float32)
    
    for i in range(num_steps):
        if i % 50 == 0:
            sim = CombinedSim(y0=np.random.uniform(-10.0, 0.0), v0=np.random.uniform(-5.0, 5.0))
            curr_state = np.array([sim.y, sim.v], dtype=np.float32)
            
        F = np.random.uniform(-2.0, 2.0)
        
        states[i] = curr_state
        actions[i] = [F]
        
        curr_state = sim.step(F)
        next_states[i] = curr_state
        
    inputs = np.concatenate([states, actions], axis=-1)
    targets = next_states
    residuals = targets - inputs[:, 0:2]
    
    hx = torch.tensor(inputs)
    hy = torch.tensor(residuals)
    
    baseline_model = Brain1_v2()
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train heavily on these collected transitions
    baseline_model.train()
    for _ in range(100):
        optimizer.zero_grad()
        preds = baseline_model(hx)
        loss = criterion(preds, hy)
        loss.backward()
        optimizer.step()
        
    return baseline_model

# 4. MEASURE ERROR HELPER
def measure_average_error(model_fn, num_tests=1000):
    states = np.zeros((num_tests, 2), dtype=np.float32)
    actions = np.zeros((num_tests, 1), dtype=np.float32)
    residuals_true = np.zeros((num_tests, 2), dtype=np.float32)
    
    sim = CombinedSim()
    for i in range(num_tests):
        y = np.random.uniform(-10.0, 0.0)
        v = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        
        sim.y = y
        sim.v = v
        # Step calculates the true next state
        next_state = sim.step(F)
        
        states[i] = [y, v]
        actions[i] = [F]
        residuals_true[i] = next_state - [y, v]
        
    inputs = np.concatenate([states, actions], axis=-1)
    hx = torch.tensor(inputs)
    hy = torch.tensor(residuals_true)
    
    # model_fn must return residual predictions given hx: [y, v, F]
    with torch.no_grad():
        preds = model_fn(hx)
        
    print(f"DEBUG SAMPLE - Inputs [y, v, F]: {hx[0]}")
    print(f"DEBUG SAMPLE - True Residual [dy, dv]: {hy[0]}")
    print(f"DEBUG SAMPLE - Pred Residual [dy, dv]: {preds[0]}")
    print(f"DEBUG SAMPLE - L2 Error: {nn.MSELoss()(preds[0:1], hy[0:1]).item()}")
        
    mse = nn.MSELoss()(preds, hy).item()
    return mse


def execute_the_paper_experiment():
    print("--- LOADING TRAINED INDIVIDUAL LAWS ---")
    brain_gravity = Brain1_v2()
    brain_gravity.load_state_dict(torch.load("../models/brain1_gravity_model.pth", weights_only=True))
    brain_gravity.eval()
    
    brain_spring = Brain1_v2()
    brain_spring.load_state_dict(torch.load("../models/brain1_spring_model.pth", weights_only=True))
    brain_spring.eval()

    print("Networks loaded. Constructing Compositional Network...")
    
    # 5. COMPOSITION FUNCTION
    def composed_forward(x_tensor):
        # x_tensor is (Batch, 3) where columns are [y, v, F]
        # We need to ask gravity: "y, v, F=0"
        batch_sz = x_tensor.shape[0]
        zeros = torch.zeros((batch_sz, 1), dtype=torch.float32)
        
        # We need to query Gravity with a safe height (e.g. y=10.0)
        # Why? Because Environment A (where gravity was learned) had a floor at y=0.
        # Environment C brings y down to -4.9. If we pass actual y, Gravity network
        # will incorrectly predict a surface bounce! Since gravity is height-invariant,
        # we can pass y=10.0 to safely get the fundamental gravity acceleration.
        safe_y = torch.full((batch_sz, 1), 10.0, dtype=torch.float32)
        
        # State for gravity: [10.0, v, 0]
        # State for spring: [y, v, 0] 
        gravity_inputs = torch.cat([safe_y, x_tensor[:, 1:2], zeros], dim=-1)
        spring_inputs = torch.cat([x_tensor[:, 0:2], zeros], dim=-1)
        
        with torch.no_grad():
            res_gravity = brain_gravity(gravity_inputs)
            res_spring = brain_spring(spring_inputs)
            
        # The force calculation F / m * dt (mass = 1.0, dt = 0.02)
        F_tensor = x_tensor[:, 2].unsqueeze(1)
        dv_force = F_tensor * 0.02  # Newton's Force Law linear superposition
        
        dy_gravity = res_gravity[:, 0].unsqueeze(1)
        dv_gravity = res_gravity[:, 1].unsqueeze(1)
        
        dy_spring = res_spring[:, 0].unsqueeze(1)
        dv_spring = res_spring[:, 1].unsqueeze(1)
        
        # The models both predict [delta_y, delta_v].
        # In reality, delta_y = v * dt + 0.5 * a * dt^2
        # Both models will predict the `v * dt` component because they get the same `v`.
        # To compose displacement, we add them and subtract the overlap: `v * dt`.
        # dy_composed = dy_gravity + dy_spring - (v * dt)
        
        # Actually, if we just use our known equation: dy_composed = v * dt + 0.5 * (a_g + a_s + a_f) * dt^2
        # It comes down to adding the predicted `dv`!
        # Remember, standard Euler step: delta_y = (v + dv) * dt, or v * dt.
        # But for these neural nets, let's just use EXACTLY what we did for dv:
        v_tensor = x_tensor[:, 1].unsqueeze(1)
        v_dt_baseline = v_tensor * 0.02
        
        dy_force = 0.5 * dv_force * 0.02 # optional quadratic term, but let's ignore it or use simple dt tracking
        dy_composed = dy_gravity + dy_spring - v_dt_baseline
        
        # We find the baseline velocity deviation (prediction at y=0, F=0)
        # Because both models were trained on states with `v`, their prediction of `dv`
        # may include a small term related to `v` (e.g. if the MLP didn't perfectly isolate it).
        # But more importantly, the prompt states: 
        # delta_v_composed = delta_v_gravity(F=0) + delta_v_spring(F=0) + F*dt/m
        
        # ACTUALLY, if we just do: dv_g + dv_s + dv_force, does it blow up because BOTH models 
        # are trying to enforce a velocity decay? 
        # WAIT. A bouncing ball simulation has an epsilon factor!
        # if y < 0: v = -epsilon * v.
        # But Environment B (spring) does NOT have this epsilon bounce.
        # However, they BOTH just predict `dv`. 
        
        dv_composed = dv_gravity + dv_spring + dv_force
        
        return torch.cat([dy_composed, dv_composed], dim=-1)

    print("\\n--- MEASUREMENT 1: ZERO-SHOT COMPOSITION ERROR ---")
    zero_shot_error = measure_average_error(composed_forward, num_tests=5000)
    print(f"Composed Model (0 training steps on Env C) Error: {zero_shot_error:.5f}")
    
    print("\\n--- MEASUREMENT 2: BASELINE LEARNING CURVE ---")
    training_steps_list = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    baseline_errors = []
    
    for steps in training_steps_list:
        if steps == 0:
            # Random initialized model error
            random_model = Brain1_v2()
            err = measure_average_error(lambda x: random_model(x), num_tests=1000)
        else:
            trained_baseline = train_baseline(steps)
            trained_baseline.eval()
            err = measure_average_error(lambda x: trained_baseline(x), num_tests=1000)
            
        baseline_errors.append(err)
        print(f"Baseline ({steps:5d} steps) Error: {err:.5f}")

    print("\\n--- MEASUREMENT 3: GENERATING THE MONEY PLOT ---")
    plt.figure(figsize=(10, 6))
    plt.plot(training_steps_list, baseline_errors, marker='o', linewidth=2, label="Baseline (Trained from Scratch)", color='tab:orange')
    plt.axhline(y=zero_shot_error, color='tab:blue', linestyle='--', linewidth=3, label=f"CASM Zero-Shot Composition (Error={zero_shot_error:.4f})")
    
    plt.yscale('log')
    plt.xlabel('Training Transitions Used')
    plt.ylabel('Prediction Error (MSE - Log Scale)')
    plt.title('Compositional Transfer Efficiency: CASM vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("paper_money_plot.png", dpi=300)
    print("Saved Money Plot to paper_money_plot.png")

    print("\\n--- MEASUREMENT 4: MULTI-STEP ROLLOUT ACCURACY ---")
    rollout_len = 200
    y_true, v_true = [], []
    y_comp, v_comp = [], []
    
    # Initial state
    sim_true = CombinedSim(y0=-1.0, v0=0.0)
    y_curr_comp = -1.0
    v_curr_comp = 0.0
    
    for t in range(rollout_len):
        state_true = sim_true.step(F=0.0)
        y_true.append(state_true[0])
        v_true.append(state_true[1])
        
        # Roll forward composed model
        x_in = torch.tensor([[y_curr_comp, v_curr_comp, 0.0]], dtype=torch.float32)
        residuals = composed_forward(x_in)
        
        y_curr_comp += residuals[0, 0].item()
        v_curr_comp += residuals[0, 1].item()
        
        y_comp.append(y_curr_comp)
        v_comp.append(v_curr_comp)

    time_axis = np.arange(rollout_len) * 0.02
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, y_true, 'k-', linewidth=3, label='True Physics')
    plt.plot(time_axis, y_comp, 'b--', linewidth=2, label='Composed Model (Zero-Shot)')
    plt.xlabel('Time (s)')
    plt.ylabel('Equilibrium Displacement y (m)')
    plt.title('200-Step Rollout: Zero-Shot Composition vs Ground Truth\n(Environment C - Never Trained On)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("multi_step_rollout.png", dpi=300)
    print("Saved Multistep Rollout Plot to multi_step_rollout.png")

if __name__ == "__main__":
    execute_the_paper_experiment()
