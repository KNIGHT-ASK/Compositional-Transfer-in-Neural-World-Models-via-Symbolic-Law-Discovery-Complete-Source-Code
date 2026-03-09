import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

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

def generate_pendulum_data(n_samples=50000, g=9.8, L=1.0, dt=0.02):
    np.random.seed(1)
    inputs = []
    targets = []
    for _ in range(n_samples):
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        omega = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        
        a = -(g/L) * np.sin(theta) + F
        dtheta = omega * dt
        domega = a * dt
        
        inputs.append([theta, omega, F])
        targets.append([dtheta, domega])
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

def generate_torsion_data(n_samples=50000, k=2.0, dt=0.02):
    np.random.seed(2)
    inputs = []
    targets = []
    for _ in range(n_samples):
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        omega = np.random.uniform(-5.0, 5.0)
        F = np.random.uniform(-2.0, 2.0)
        
        a = -k * theta + F
        dtheta = omega * dt
        domega = a * dt
        
        inputs.append([theta, omega, F])
        targets.append([dtheta, domega])
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)

def train_model(inputs, targets):
    inputs_t = torch.tensor(inputs)
    targets_t = torch.tensor(targets)
    model = Brain1_v2()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for _ in range(250):
        optimizer.zero_grad()
        loss = criterion(model(inputs_t), targets_t)
        loss.backward()
        optimizer.step()
    return model

if __name__ == "__main__":
    print("Training Pendulum Brain...")
    p_in, p_tgt = generate_pendulum_data()
    p_in, p_tgt = p_in[np.random.permutation(len(p_in))], p_tgt[np.random.permutation(len(p_tgt))]
    model_p = train_model(p_in, p_tgt)

    print("Training Torsion Brain...")
    t_in, t_tgt = generate_torsion_data()
    t_in, t_tgt = t_in[np.random.permutation(len(t_in))], t_tgt[np.random.permutation(len(t_tgt))]
    model_t = train_model(t_in, t_tgt)

    print("\nEvaluating Zero-Shot Composition on Non-Linear Combined Environment...")
    np.random.seed(3)
    test_inputs = []
    true_targets = []
    for _ in range(5000):
        t = np.random.uniform(-np.pi/2, np.pi/2)
        o = np.random.uniform(-2.0, 2.0)
        F = np.random.uniform(-1.0, 1.0)
        test_inputs.append([t, o, F])
        a = -9.8 * np.sin(t) - 2.0 * t + F
        true_targets.append([o * 0.02, a * 0.02])

    test_inputs_t = torch.tensor(test_inputs, dtype=torch.float32)
    true_targets_t = torch.tensor(true_targets, dtype=torch.float32)

    with torch.no_grad():
        res_p = model_p(test_inputs_t)
        t_in_iso = test_inputs_t.clone()
        t_in_iso[:, 2] = 0.0
        res_t = model_t(t_in_iso)

    val_dt = test_inputs_t[:, 1:2] * 0.02
    pred_dtheta = res_p[:, 0:1] + res_t[:, 0:1] - val_dt
    pred_domega = res_p[:, 1:2] + res_t[:, 1:2] 

    composed_preds = torch.cat([pred_dtheta, pred_domega], dim=1)
    mse = nn.MSELoss()(composed_preds, true_targets_t)
    print(f"Zero-Shot Neural Composition MSE: {mse.item():.8f}")

    def discover_equation(inputs, outputs, feature_names, alpha=0.0001, degree=3):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(inputs)
        poly_names = poly.get_feature_names_out(feature_names)
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_poly, outputs)
        coefs = model.coef_
        eq = []
        for name, coef in zip(poly_names, coefs):
            if abs(coef) > 1e-4:
                eq.append(f"{coef:+.6f} * {name}")
        return eq

    probe_in = []
    probe_out = []
    for t in np.linspace(-np.pi/2, np.pi/2, 30):
        for o in np.linspace(-5.0, 5.0, 30):
            state = torch.tensor([[t, o, 0.0]], dtype=torch.float32)
            with torch.no_grad():
                dom = model_p(state).numpy()[0, 1]
            probe_in.append([t, o])
            probe_out.append(dom)

    eq = discover_equation(np.array(probe_in), np.array(probe_out), ['th', 'om'], degree=3)
    print(f"\nSINDy on Pendulum Brain (degree 3): d_omega = {' '.join(eq)}")
    print("Note: Taylor series for sin(th) is th - th^3/6")
    print(f"Expected: a = -9.8 * sin(th)*0.02 = -0.196 * th + 0.0326 * th^3")
