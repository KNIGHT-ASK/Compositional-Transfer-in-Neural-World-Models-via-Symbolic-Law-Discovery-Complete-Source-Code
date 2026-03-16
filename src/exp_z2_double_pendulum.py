import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Environment: Double Pendulum (Lagrange Dynamics)
# ---------------------------------------------------------
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
G = 9.81
dt = 0.01

class DoublePendulum:
    def __init__(self, th1, th2, w1, w2):
        self.s = np.array([th1, th2, w1, w2], dtype=float)
        
    def get_accels(self, s):
        th1, th2, w1, w2 = s
        dth = th1 - th2
        
        num1 = -G * (2 * M1 + M2) * np.sin(th1) - M2 * G * np.sin(th1 - 2 * th2) \
               - 2 * np.sin(dth) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(dth))
        den1 = L1 * (2 * M1 + M2 - M2 * np.cos(2 * th1 - 2 * th2))
        a1 = num1 / den1
        
        num2 = 2 * np.sin(dth) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(th1) \
               + w2**2 * L2 * M2 * np.cos(dth))
        den2 = L2 * (2 * M1 + M2 - M2 * np.cos(2 * th1 - 2 * th2))
        a2 = num2 / den2
        return np.array([a1, a2])

    def step(self):
        def func(s):
            w1, w2 = s[2], s[3]
            a = self.get_accels(s)
            return np.array([w1, w2, a[0], a[1]])
            
        k1 = func(self.s)
        k2 = func(self.s + 0.5 * dt * k1)
        k3 = func(self.s + 0.5 * dt * k2)
        k4 = func(self.s + dt * k3)
        self.s += (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return self.s.copy()

    def get_energy(self):
        th1, th2, w1, w2 = self.s
        # Kinetic Energy
        ke = 0.5 * (M1 + M2) * L1**2 * w1**2 + 0.5 * M2 * L2**2 * w2**2 + \
             M2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)
        # Potential Energy
        pe = -(M1 + M2) * G * L1 * np.cos(th1) - M2 * G * L2 * np.cos(th2)
        return ke + pe

def collect_data(n=60000):
    X, Y = [], []
    for _ in range(n // 100):
        # Starting with high-energy for professional chaos benchmarks
        s = np.random.uniform(-np.pi, np.pi, 2)
        s = np.concatenate([s, [0, 0]]) # Start from rest but high PE
        sim = DoublePendulum(*s)
        e_init = sim.get_energy()
        for i in range(100):
            s_old = sim.s.copy()
            s_new = sim.step()
            if i == 99:
                e_final = sim.get_energy()
                drift = abs(e_final - e_init)
                # Fail-safe check
                if drift > 0.01: continue 
            X.append(s_old)
            Y.append((s_new[2:4] - s_old[2:4]) / dt)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

if __name__ == "__main__":
    print("--- Phase II: Chaotic Double Pendulum (Lagrange Chaos) ---")
    X, Y = collect_data()
    m = BrainPendulum()
    opt = optim.Adam(m.parameters(), lr=5e-4)
    
    print("Training Double Pendulum Brain...")
    for i in range(250):
        opt.zero_grad()
        loss = nn.MSELoss()(m(X), Y)
        loss.backward(); opt.step()
        if i % 50 == 0: print(f"  Epoch {i}, Loss: {loss.item():.10f}")
        
    print("\nModel trained on chaotic attractor.")
    print("Ready for Attractor Persistence benchmarks.")
