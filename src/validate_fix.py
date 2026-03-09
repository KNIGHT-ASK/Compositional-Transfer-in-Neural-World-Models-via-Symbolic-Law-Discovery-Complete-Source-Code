import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class BouncingBallSim:
    def __init__(self, y0=10.0, v0=0.0):
        self.state = [y0, v0] # [y, v]
        self.g = -9.8
        self.m = 1.0
        self.dt = 0.02
        self.e = 0.8 # coefficient of restitution
        
    def step(self, F_t=0.0):
        y, v = self.state
        a = self.g + F_t/self.m
        v_next = v + a * self.dt
        y_next = y + v_next * self.dt
        
        if y_next < 0:
            y_next = 0.0
            v_next = -self.e * v_next
            
        self.state = [y_next, v_next]
        return self.state

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

brain1_v2 = Brain1_v2()

data_pool = []
for _ in range(1000):
    sim = BouncingBallSim()
    sim.state[0] = random.uniform(1.0, 10.0)
    sim.state[1] = random.uniform(-5.0, 5.0)
    
    for _ in range(50):
        y_t, v_t = sim.state.copy()
        F_t = random.uniform(-2.0, 2.0)
        
        sim.step(F_t)
        
        y_t1, v_t1 = sim.state.copy()
        
        dy = y_t1 - y_t
        dv = v_t1 - v_t
        
        data_pool.append([y_t, v_t, F_t, dy, dv])

data_tensor = torch.FloatTensor(data_pool)
X = data_tensor[:, :3]
Y = data_tensor[:, 3:]
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

optimizer_v2 = optim.Adam(brain1_v2.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training Brain 1 v1.1...")
for epoch in range(100):
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        predictions = brain1_v2(batch_X)
        loss = criterion(predictions, batch_Y)
        optimizer_v2.zero_grad()
        loss.backward()
        optimizer_v2.step()
        epoch_loss += loss.item()

print(f"Final Loss: {epoch_loss / len(dataloader):.6f}")

brain1_v2.eval()
print("\\n==== GRAVITY EXTRACTION TEST (v1.1) ====")
test_heights = [2.0, 5.0, 8.0, 12.0, 20.0, 50.0]
with torch.no_grad():
    for h in test_heights:
        test_input = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
        prediction = brain1_v2(test_input)
        pred_dv = prediction[0, 1].item()
        print(f"Height {h:4.1f} -> Predicted \\Delta v = {pred_dv:7.4f}")

print("\\n==== FORCE RESPONSE TEST (v1.1) ====")
test_forces = [-5.0, -2.0, 0.0, 2.0, 5.0, 9.8]
with torch.no_grad():
    for f in test_forces:
        test_input = torch.tensor([[5.0, 0.0, f]], dtype=torch.float32)
        prediction = brain1_v2(test_input)
        pred_dv = prediction[0, 1].item()
        th_dv = (-9.8 + f) * 0.02
        print(f"Force {f:4.1f} -> Pred \\Delta v = {pred_dv:7.4f} (True: {th_dv:7.4f})")
