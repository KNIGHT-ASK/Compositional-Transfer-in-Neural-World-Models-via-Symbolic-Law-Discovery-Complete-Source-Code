import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

class Brain1_v2(nn.Module):
    def __init__(self):
        super(Brain1_v2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def run_sindy_noise_ablation():
    print("--- SINDy Noise Robustness Ablation ---")
    
    # Load the spring model for a cleaner test case (since it recovered 99.8% accurately without floor bounds)
    model = Brain1_v2()
    try:
        model.load_state_dict(torch.load("../models/brain1_spring_model.pth", map_location='cpu'))
        model.eval()
    except:
        print("Model not found. Run training first.")
        return
        
    state_grid = []
    # Use realistic state ranges
    for y in np.linspace(2, 10, 20):
        for v in np.linspace(-5, 5, 20):
            for f in np.linspace(-2, 2, 5):
                state_grid.append([y, v, f])
                
    X_test_clean = torch.tensor(state_grid, dtype=torch.float32)
    with torch.no_grad():
        Y_pred_clean = model(X_test_clean).numpy()
        
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1]
    
    for snr in noise_levels:
        # Add gaussian noise to the NEURAL NETWORK'S outputs before feeding to SINDy
        # This simulates uncertainty or measurement noise in the prediction manifold
        noise_y = np.random.normal(0, snr * np.std(Y_pred_clean[:, 0]), size=Y_pred_clean[:, 0].shape)
        noise_v = np.random.normal(0, snr * np.std(Y_pred_clean[:, 1]), size=Y_pred_clean[:, 1].shape)
        
        Y_noisy = Y_pred_clean.copy()
        Y_noisy[:, 0] += noise_y
        Y_noisy[:, 1] += noise_v
        
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X_test_clean.numpy())
        feature_names = poly.get_feature_names_out(['y', 'v', 'F'])
        
        # Test standard Lasso
        lasso = Lasso(alpha=1e-4, fit_intercept=False, max_iter=10000)
        lasso.fit(X_poly, Y_noisy[:, 1])
        
        coefs = lasso.coef_
        terms = []
        recovered_k_m = 0
        for name, c in zip(feature_names, coefs):
            if abs(c) > 1e-3:
                terms.append(f"{c:.4f}*{name}")
                if name == 'y':  # Looking for -k/m * y * dt => delta_v = -c * y -> k/m = -c/dt
                    recovered_k_m = -c / 0.02
                    
        print(f"Noise SNR {snr:.3f} | Recovered k/m: {recovered_k_m:.4f} | Terms: {' + '.join(terms)}")

if __name__ == "__main__":
    run_sindy_noise_ablation()
