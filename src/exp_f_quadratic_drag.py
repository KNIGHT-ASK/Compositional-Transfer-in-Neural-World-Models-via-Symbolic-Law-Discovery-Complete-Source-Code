import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

# Model
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

def generate_quad_drag_data(n_samples=50000, c=0.1, m=1.0, dt=0.02):
    np.random.seed(42)
    inputs = []
    targets = []
    
    for _ in range(n_samples // 50):
        y = np.random.uniform(2.0, 15.0)
        v = np.random.uniform(-5.0, 5.0)
        
        for _ in range(50):
            F = np.random.uniform(-2.0, 2.0)
            a = -c * v * abs(v) / m + (F / m)
            dv = a * dt
            dy = v * dt
            
            inputs.append([y, v, F])
            targets.append([dy, dv])
            
            v += dv
            y += dy
            if y < 0:
                y = 0
                v = -0.8 * v
                
    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    return inputs, targets

if __name__ == "__main__":
    inputs, targets = generate_quad_drag_data()
    indices = np.random.permutation(len(inputs))
    inputs, targets = inputs[indices], targets[indices]

    inputs_t = torch.tensor(inputs)
    targets_t = torch.tensor(targets)

    model = Brain1_v2()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training Quad Drag Brain...")
    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(inputs_t)
        loss = criterion(preds, targets_t)
        loss.backward()
        optimizer.step()

    print(f"Final MLP Loss: {loss.item():.6f}")

    # SINDy Extraction
    def discover_equation(inputs, outputs, feature_names, alpha=0.0001, degree=2):
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(inputs)
        poly_names = poly.get_feature_names_out(feature_names)
        
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        model.fit(X_poly, outputs)
        
        coefs = model.coef_
        equation_terms = []
        for name, coef in zip(poly_names, coefs):
            if abs(coef) > 1e-4:
                equation_terms.append(f"{coef:+.6f} * {name}")
        return equation_terms

    # Probe
    probe_inputs = []
    probe_outputs = []
    for y in np.linspace(2.0, 15.0, 30):
        for v in np.linspace(-5.0, 5.0, 30):
            F = 0.0
            state = torch.tensor([[y, v, F]], dtype=torch.float32)
            with torch.no_grad():
                delta = model(state).numpy()[0]
            probe_inputs.append([y, v])
            probe_outputs.append(delta[1]) # dv
            
    probe_inputs = np.array(probe_inputs)
    probe_outputs = np.array(probe_outputs)

    eq = discover_equation(probe_inputs, probe_outputs, ['y', 'v'])
    print("\n[Out-of-library Quad Drag SINDy Result]")
    print(f"True physics: dv = -0.1 * v * abs(v) * dt = -0.002 * v * abs(v)")
    print("SINDy recovered: dv = " + " ".join(eq))
