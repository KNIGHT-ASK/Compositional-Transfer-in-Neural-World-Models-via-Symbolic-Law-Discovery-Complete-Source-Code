import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

class BouncingBallSim:
    def __init__(self, y0=5.0, v0=0.0, g=9.8, m=1.0, dt=0.02, epsilon=0.8):
        self.y = y0
        self.v = v0
        self.g = g
        self.m = m
        self.dt = dt
        self.epsilon = epsilon
        self.time = 0.0

    def get_state(self):
        return np.array([self.y, self.v], dtype=np.float32)

    def step(self, F=0.0):
        a = -self.g + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        if self.y < 0:
            self.y = 0
            self.v = -self.epsilon * self.v
        self.time += self.dt
        return self.get_state()

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

def execute_extended_training():
    print("Collecting 1000 episodes (50,000 transitions) of experience...")
    batch_size = 1000
    seq_len = 50
    
    # [y, v, F]
    states = np.zeros((batch_size, seq_len + 1, 2), dtype=np.float32)
    actions = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
    
    for b in range(batch_size):
        sim = BouncingBallSim(y0=np.random.uniform(1.0, 10.0), v0=np.random.uniform(-5.0, 5.0))
        states[b, 0] = sim.get_state()
        for t in range(seq_len):
            F = np.random.uniform(-2.0, 2.0)
            actions[b, t, 0] = F
            states[b, t+1] = sim.step(F)
            
    inputs_seq = np.concatenate([states[:, :-1, :], actions], axis=-1)
    targets_seq = states[:, 1:, :]
    
    inputs_flat = inputs_seq.reshape(-1, 3)
    targets_flat = targets_seq.reshape(-1, 2)
    
    residuals_flat = targets_flat - inputs_flat[:, 0:2]
    
    dataset_size = inputs_flat.shape[0]
    indices = np.random.permutation(dataset_size)
    inputs_shuffled = inputs_flat[indices]
    residuals_shuffled = residuals_flat[indices]

    hx = torch.tensor(inputs_shuffled)
    hy = torch.tensor(residuals_shuffled)

    brain = Brain1_v2()
    optimizer = optim.Adam(brain.parameters(), lr=0.0005) # Lower LR for fine-tuning
    criterion = nn.MSELoss()

    print("Starting Extended Training (1000 epochs over shuffled transitions)...")
    batch_sz = 1000
    for ep in range(1000):
        # We process in mini-batches to get more updates per epoch
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_sz):
            batch_hx = hx[i:i+batch_sz]
            batch_hy = hy[i:i+batch_sz]
            
            optimizer.zero_grad()
            preds = brain(batch_hx)
            loss = criterion(preds, batch_hy)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (ep+1) % 100 == 0:
            print(f'Epoch {ep+1:04d} | Avg Loss: {epoch_loss / (dataset_size//batch_sz):.5f}')

    print('\\n--- FINAL GRAVITY EXTRACTION TEST ---')
    brain.eval()
    with torch.no_grad():
        for y_i in [2.0, 5.0, 8.0, 12.0]:
            xin = torch.tensor([[y_i, 0.0, 0.0]], dtype=torch.float32)
            pred = brain(xin)
            print(f'Init Height: {y_i} | Predicted Δv: {pred[0,1].item():.4f}')
            
    torch.save(brain.state_dict(), "../models/brain1_gravity_model.pth")
    print("Saved optimal model to brain1_gravity_model.pth")

if __name__ == "__main__":
    execute_extended_training()
