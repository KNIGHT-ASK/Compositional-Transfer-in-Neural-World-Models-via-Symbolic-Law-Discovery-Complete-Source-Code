import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SpringSim:
    def __init__(self, x0=3.0, v0=0.0, k=2.0, m=1.0, dt=0.02):
        self.x = x0
        self.v = v0
        self.k = k
        self.m = m
        self.dt = dt
        self.time = 0.0

    def get_state(self):
        return np.array([self.x, self.v], dtype=np.float32)

    def step(self, F=0.0):
        a = (-self.k * self.x + F) / self.m
        self.v += a * self.dt
        self.x += self.v * self.dt
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

def train_brain1_spring():
    print("Collecting 50,000 transitions from Environment B (Spring)...")
    batch_size = 1000
    seq_len = 50
    
    # State: [x, v, F]
    states = np.zeros((batch_size, seq_len + 1, 2), dtype=np.float32)
    actions = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
    
    for b in range(batch_size):
        sim = SpringSim(x0=np.random.uniform(-5.0, 5.0), v0=np.random.uniform(-5.0, 5.0))
        states[b, 0] = sim.get_state()
        for t in range(seq_len):
            F = np.random.uniform(-2.0, 2.0)
            actions[b, t, 0] = F
            states[b, t+1] = sim.step(F)
            
    # Format inputs and target residuals
    inputs_seq = np.concatenate([states[:, :-1, :], actions], axis=-1)
    targets_seq = states[:, 1:, :]
    
    inputs_flat = inputs_seq.reshape(-1, 3)
    targets_flat = targets_seq.reshape(-1, 2)
    residuals_flat = targets_flat - inputs_flat[:, 0:2]
    
    # Shuffle completely to break temporal chains
    dataset_size = inputs_flat.shape[0]
    indices = np.random.permutation(dataset_size)
    inputs_shuffled = inputs_flat[indices]
    residuals_shuffled = residuals_flat[indices]

    hx = torch.tensor(inputs_shuffled)
    hy = torch.tensor(residuals_shuffled)

    brain_spring = Brain1_v2()
    optimizer = optim.Adam(brain_spring.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training Brain 1 on Spring Dynamics (MLP + Residuals)...")
    batch_sz = 1000
    for ep in range(500):
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_sz):
            batch_hx = hx[i:i+batch_sz]
            batch_hy = hy[i:i+batch_sz]
            
            optimizer.zero_grad()
            preds = brain_spring(batch_hx)
            loss = criterion(preds, batch_hy)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (ep+1) % 50 == 0:
            print(f'Epoch {ep+1:04d} | Avg Loss: {epoch_loss / (dataset_size//batch_sz):.5f}')

    print('\\n--- SPRING EXTRACTION TEST ---')
    print("Testing implicit extraction of Hooke's Law: Δv = -k * x * dt")
    print('Expected Δv formula: -2.0 * x * 0.02 = -0.04 * x')
    brain_spring.eval()
    with torch.no_grad():
        test_x_values = [0.0, 1.0, 2.0, 5.0, -2.0]
        for x_i in test_x_values:
            expected_dv = -2.0 * x_i * 0.02
            xin = torch.tensor([[x_i, 0.0, 0.0]], dtype=torch.float32)
            pred = brain_spring(xin)
            print(f'Displacement x: {x_i:4.1f} | Expected Δv: {expected_dv:6.3f} | Predicted Δv: {pred[0,1].item():6.4f}')
            
    torch.save(brain_spring.state_dict(), "../models/brain1_spring_model.pth")
    print('\\nSaved optimal spring model to brain1_spring_model.pth')

if __name__ == "__main__":
    train_brain1_spring()
