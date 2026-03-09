import nbformat
import os

def append_to_notebook():
    notebook_path = os.path.join(os.getcwd(), "CASM_Brain1_Full.ipynb")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    markdown_cell = nbformat.v4.new_markdown_cell("""---
# PART D: Brain 1 v1.1 — The MLP Residual Fix

Based on the adversarial diagnosis, the GRU was pattern-matching sequence histories. We now implement the theoretically correct architecture for learning local Markovian dynamics:
1. **Architecture:** MLP (No memory, No hidden state)
2. **Target:** Predict Change ($\Delta y, \Delta v$), not absolute state.
3. **Data:** Train on shuffled, individual transitions, not ordered sequences.
""")

    code_cell_1 = nbformat.v4.new_code_cell("""import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------
# 1. The Fixed Architecture: MLP for Residuals
# ---------------------------------------------------------
class Brain1_v2(nn.Module):
    def __init__(self):
        super(Brain1_v2, self).__init__()
        # Input: [y_t, v_t, F_t] -> Output: [\Delta y, \Delta v]
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
print(f"Brain 1 v1.1 Parameter Count: {sum(p.numel() for p in brain1_v2.parameters())}")

# ---------------------------------------------------------
# 2. Collect Shuffled Training Data
# ---------------------------------------------------------
print("Collecting 50,000 random transitions...")
data_pool = []

for _ in range(1000):
    sim = BouncingBallSim()
    # Randomize initial state 
    sim.state[0] = random.uniform(1.0, 10.0) # y
    sim.state[1] = random.uniform(-5.0, 5.0) # v
    
    for _ in range(50):
        y_t, v_t = sim.state.copy()
        F_t = random.uniform(-2.0, 2.0)
        
        sim.step(F_t)
        
        y_t1, v_t1 = sim.state.copy()
        
        # Calculate true residuals (change)
        dy = y_t1 - y_t
        dv = v_t1 - v_t
        
        data_pool.append([y_t, v_t, F_t, dy, dv])

# Convert to tensors
import numpy as np
data_tensor = torch.FloatTensor(data_pool)

# Inputs: [y_t, v_t, F_t]
X = data_tensor[:, :3]
# Targets: [\Delta y, \Delta v]
Y = data_tensor[:, 3:]

# Create DataLoader for random batching
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

print(f"Dataset created with {len(dataset)} transitions. Ready for training.")
""")

    code_cell_2 = nbformat.v4.new_code_cell("""# ---------------------------------------------------------
# 3. Train on Shuffled Transitions
# ---------------------------------------------------------
optimizer_v2 = optim.Adam(brain1_v2.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 100
loss_history_v2 = []

print("Training Brain 1 v1.1 (MLP Residuals)...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        
        # Forward pass
        predictions = brain1_v2(batch_X)
        
        # Loss
        loss = criterion(predictions, batch_Y)
        
        # Backward and optimize
        optimizer_v2.zero_grad()
        loss.backward()
        optimizer_v2.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    loss_history_v2.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(loss_history_v2, color='green')
plt.title('Brain 1 v1.1 Training Loss (Predicting Residuals)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True)
plt.show()    
""")

    code_cell_3 = nbformat.v4.new_code_cell("""# ---------------------------------------------------------
# 4. The Critical Tests
# ---------------------------------------------------------
brain1_v2.eval()
print("==== GRAVITY EXTRACTION TEST (v1.1) ====")
print("Hypothesis: \\Delta v should be approx -0.196 regardless of height.")

test_heights = [2.0, 5.0, 8.0, 12.0, 20.0, 50.0]
with torch.no_grad():
    for h in test_heights:
        # [y=h, v=0, F=0]
        test_input = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)
        prediction = brain1_v2(test_input)
        pred_dv = prediction[0, 1].item()
        print(f"Height {h:4.1f} -> Predicted \\Delta v = {pred_dv:7.4f}")

print("\\n==== FORCE RESPONSE TEST (v1.1) ====")
print("Hypothesis: \\Delta v = (-g + F/m) * dt")
print("Expected: a = -9.8 + F -> \\Delta v = a * 0.02")

test_forces = [-5.0, -2.0, 0.0, 2.0, 5.0, 9.8]
with torch.no_grad():
    for f in test_forces:
        # [y=5.0, v=0, F=f]
        test_input = torch.tensor([[5.0, 0.0, f]], dtype=torch.float32)
        prediction = brain1_v2(test_input)
        pred_dv = prediction[0, 1].item()
        
        # Theoretical calculation
        theoretical_a = -9.8 + f
        theoretical_dv = theoretical_a * 0.02
        
        print(f"Force {f:4.1f} -> Predicted \\Delta v = {pred_dv:7.4f} (Theoretical: {theoretical_dv:7.4f})")
""")

    nb.cells.extend([markdown_cell, code_cell_1, code_cell_2, code_cell_3])
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print("Successfully appended Part D (Brain 1 v1.1) to CASM_Brain1_Full.ipynb")

if __name__ == "__main__":
    append_to_notebook()
