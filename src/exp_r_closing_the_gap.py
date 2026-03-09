import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- GOOGLE COLAB PYTORCH BUG FIX ---
if not hasattr(torch, '_utils'):
    class DummyUtils:
        @staticmethod
        def _get_device_index(device):
            return 0
    torch._utils = DummyUtils

# Shared Constants
dt = 0.02
mass = 1.0
g = 9.8
k_spring = 2.0

class BouncingBallSim:
    def __init__(self, y0=0.0, v0=0.0):
        self.y, self.v = y0, v0
    def step(self, F=0.0):
        a = -g + (F / mass)
        self.v += a * dt
        self.y += self.v * dt
        return self.y, self.v

class SpringSim:
    def __init__(self, x0=0.0, v0=0.0):
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

class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

class TinyCalibrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

def collect_gravity_data(n, bounds=(-15, 15)):
    X, Y = [], []
    sim = BouncingBallSim(y0=np.random.uniform(*bounds))
    for i in range(n):
        if i % 50 == 0:
            sim = BouncingBallSim(y0=np.random.uniform(*bounds), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def collect_spring_data(n, bounds=(-15, 15)):
    X, Y = [], []
    sim = SpringSim(x0=np.random.uniform(*bounds))
    for i in range(n):
        if i % 50 == 0:
            sim = SpringSim(x0=np.random.uniform(*bounds), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        x_old, v_old = sim.x, sim.v
        x_new, v_new = sim.step(F)
        X.append([x_old, v_old, F])
        Y.append([x_new - x_old, v_new - v_old])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def collect_combined_data(n, bounds=(-10, 0)):
    X, Y = [], []
    sim = CombinedSim(y0=np.random.uniform(*bounds))
    for i in range(n):
        if i % 50 == 0:
            sim = CombinedSim(y0=np.random.uniform(*bounds), v0=np.random.uniform(-5, 5))
        F = np.random.uniform(-2, 2)
        y_old, v_old = sim.y, sim.v
        y_new, v_new = sim.step(F)
        X.append([y_old, v_old, F])
        Y.append([y_new - y_old, v_new - v_old])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

def train_model(model, X, Y, epochs=200, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(512, len(X)), shuffle=True)
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def eval_composition(gravity_models, spring_models, calibration_model=None, n_test=5000):
    errors = []
    
    # Optional: if passing single models, wrap in list
    if not isinstance(gravity_models, list):
        gravity_models = [gravity_models]
    if not isinstance(spring_models, list):
        spring_models = [spring_models]
        
    for _ in range(n_test):
        y = np.random.uniform(-10, 0)
        v = np.random.uniform(-5, 5)
        F = np.random.uniform(-2, 2)
        
        sim = CombinedSim(y0=y, v0=v)
        y_new, v_new = sim.step(F)
        true_dy, true_dv = y_new - y, v_new - v

        x = torch.tensor([[y, v, 0.0]], dtype=torch.float32)
        x_target = torch.tensor([[y, v, F]], dtype=torch.float32) # For calibration
        
        with torch.no_grad():
            rg_list = [g(x).numpy()[0] for g in gravity_models]
            rs_list = [s(x).numpy()[0] for s in spring_models]
            
            rg = np.mean(rg_list, axis=0)
            rs = np.mean(rs_list, axis=0)
            
            dv_pred = rg[1] + rs[1] + F * dt
            dy_pred = rg[0] + rs[0] - v * dt
            
            if calibration_model is not None:
                calib = calibration_model(x_target).numpy()[0]
                dy_pred += calib[0]
                dv_pred += calib[1]
                
        err = (dy_pred - true_dy)**2 + (dv_pred - true_dv)**2
        errors.append(err)
        
    return np.mean(errors)

if __name__ == "__main__":
    print("=== EXPERIMENT R: CLOSING THE GAP ===")
    print("Testing Ensemble Composition (0 target data)")
    print("Testing Calibration Bridge (500 target data)")
    
    n_seeds = 3 # For rapid testing
    ensemble_size = 5
    
    comp_mses = []
    ens_mses = []
    calib_mses = []
    
    for seed in range(n_seeds):
        print(f"\n--- RUNNING METASEED {seed+1}/{n_seeds} ---")
        torch.manual_seed(seed * 42)
        np.random.seed(seed * 42)
        
        g_ensemble = []
        s_ensemble = []
        
        print(f"Training Ensemble of {ensemble_size} components...")
        for i in range(ensemble_size):
            # Reduced data to 20K per ensemble member for speed in diagnostics
            Xg, Yg = collect_gravity_data(20000, bounds=(-15, 15))
            bg = Brain()
            train_model(bg, Xg, Yg, epochs=50)
            g_ensemble.append(bg)
            
            Xs, Ys = collect_spring_data(20000, bounds=(-15, 15))
            bs = Brain()
            train_model(bs, Xs, Ys, epochs=50)
            s_ensemble.append(bs)
            
        print("Evaluating Base Composition (1 network)...")
        mse_base = eval_composition(g_ensemble[0], s_ensemble[0])
        comp_mses.append(mse_base)
        print(f"  Base MSE:     {mse_base*1e4:.4f} x 10^-4")
        
        print("Evaluating Ensemble Composition (5 networks)...")
        mse_ens = eval_composition(g_ensemble, s_ensemble)
        ens_mses.append(mse_ens)
        print(f"  Ensemble MSE: {mse_ens*1e4:.4f} x 10^-4")
        
        print("Training Calibration Bridge (500 combined points)...")
        Xc, Yc = collect_combined_data(500, bounds=(-10, 0))
        
        # Calculate raw compositional residual error to supervise tiny net
        res_Y = []
        for i in range(len(Xc)):
            with torch.no_grad():
                # Formulate input carefully
                y_val, v_val, F_val = Xc[i][0].item(), Xc[i][1].item(), Xc[i][2].item()
                x0 = torch.tensor([[y_val, v_val, 0.0]], dtype=torch.float32)
                
                rg = g_ensemble[0](x0).numpy()[0]
                rs = s_ensemble[0](x0).numpy()[0]
                
                pred_dy = rg[0] + rs[0] - v_val * dt
                pred_dv = rg[1] + rs[1] + F_val * dt
                
                true_dy = Yc[i][0].item()
                true_dv = Yc[i][1].item()
                
                res_Y.append([true_dy - pred_dy, true_dv - pred_dv])
                
        res_Y = torch.tensor(np.array(res_Y), dtype=torch.float32)
        
        calib_net = TinyCalibrationNet()
        train_model(calib_net, Xc, res_Y, epochs=200, lr=0.005)
        
        print("Evaluating Calibration Bridge Composition...")
        mse_calib = eval_composition(g_ensemble[0], s_ensemble[0], calibration_model=calib_net)
        calib_mses.append(mse_calib)
        print(f"  Calib MSE:    {mse_calib*1e4:.4f} x 10^-4")
        
    print("\n" + "="*50)
    print("FINAL RESULTS (Means over 3 Meta-Seeds)")
    print("="*50)
    print(f"Base Composition (0 target data):   {np.mean(comp_mses)*1e4:.4f} x 10^-4")
    print(f"Ensemble Comp.   (0 target data):   {np.mean(ens_mses)*1e4:.4f} x 10^-4")
    print(f"Calibration Comp.(500 target data): {np.mean(calib_mses)*1e4:.4f} x 10^-4")
