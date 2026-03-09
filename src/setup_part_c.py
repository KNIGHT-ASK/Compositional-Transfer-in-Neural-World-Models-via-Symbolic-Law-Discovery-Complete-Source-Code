import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

# Ensure reproducibility parameters
torch.manual_seed(42)
np.random.seed(42)

class BouncingBallSim:
    def __init__(self, y0=5.0, v0=0.0, g=9.8, m=1.0, dt=0.02, epsilon=0.8):
        self.y = y0; self.v = v0; self.g = g; self.m = m
        self.dt = dt; self.epsilon = epsilon
    def get_state(self):
        return np.array([self.y, self.v], dtype=np.float32)
    def step(self, F=0.0):
        a = -self.g + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        if self.y < 0:
            self.y = 0; self.v = -self.epsilon * self.v
        return self.get_state()

class Brain1WorldModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.pred = nn.Linear(hidden_size, output_size)
    def forward(self, x, h_0=None):
        out, h_n = self.gru(x, h_0)
        return self.pred(out), h_n

def generate_batch(batch_size=32, seq_len=50):
    states = np.zeros((batch_size, seq_len + 1, 2), dtype=np.float32)
    actions = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
    
    for b in range(batch_size):
        # Random initialization
        y0 = np.random.uniform(1.0, 10.0)
        v0 = np.random.uniform(-5.0, 5.0)
        sim = BouncingBallSim(y0=y0, v0=v0)
        states[b, 0] = sim.get_state()
        
        for t in range(seq_len):
            F = np.random.uniform(-2.0, 2.0)
            actions[b, t, 0] = F
            states[b, t+1] = sim.step(F)
            
    # Input sequence: [S_t, A_t] -> size (batch, seq, 3)
    inputs = np.concatenate([states[:, :-1, :], actions], axis=-1)
    # Target sequence: [S_{t+1}] -> size (batch, seq, 2)
    targets = states[:, 1:, :]
    return torch.tensor(inputs), torch.tensor(targets)

brain = Brain1WorldModel()
optimizer = optim.Adam(brain.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\n=== CASM Brain 1: Part C Training Verification ===")

epochs = 1000
for ep in range(epochs):
    x_batch, y_batch = generate_batch()
    optimizer.zero_grad()
    predictions, _ = brain(x_batch)
    loss = criterion(predictions, y_batch)
    loss.backward()
    optimizer.step()
    
    if (ep + 1) % 100 == 0 or ep == 0:
        print(f"Episode {ep+1:04d} | Loss: {loss.item():.4f}")

# Extraction Test: g
print("\n[Gravity Extraction Test]")
brain.eval()
with torch.no_grad():
    g_preds = []
    print(f"| Input_y | Input_v |   F   |   pred_v_next  |     Δv     |")
    print(f"|---------|---------|-------|----------------|------------|")
    for y_init in [2.0, 5.0, 8.0, 12.0, 20.0]:
        x_test = torch.tensor([[[y_init, 0.0, 0.0]]]) # batch 1, seq 1, [y,v,F]
        pred_state, _ = brain(x_test)
        pred_v_next = pred_state[0, 0, 1].item()
        delta_v = pred_v_next - 0.0 # because v_0 was 0
        g_preds.append(delta_v)
        print(f"| {y_init:7.2f} | {0.0:7.2f} | {0.0:5.2f} | {pred_v_next:14.4f} | {delta_v:10.4f} |")

print(f"\nAverage Δv per 0.02s step: {np.mean(g_preds):.4f} (Theoretical True: -0.1960)")
print("========================================================\n")

# Injecting Part C into Jupyter Notebook
workspace = r"d:\ALL MY ML AND AI"
nb_path = os.path.join(workspace, "CASM_Brain1_Full.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb_dict = json.load(f)

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# PART C: The Training Loop\n", "The Brain 1 optimization mapping over $\\mathcal{L}_t = \\|\\hat{S}_{t+1} - S_{t+1}\\|^2$"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import torch.optim as optim\n\n",
        "def generate_batch(batch_size=32, seq_len=50):\n",
        "    states = np.zeros((batch_size, seq_len + 1, 2), dtype=np.float32)\n",
        "    actions = np.zeros((batch_size, seq_len, 1), dtype=np.float32)\n",
        "    for b in range(batch_size):\n",
        "        sim = BouncingBallSim(y0=np.random.uniform(1.0, 10.0), v0=np.random.uniform(-5.0, 5.0))\n",
        "        states[b, 0] = sim.get_state()\n",
        "        for t in range(seq_len):\n",
        "            F = np.random.uniform(-2.0, 2.0)\n",
        "            actions[b, t, 0] = F\n",
        "            states[b, t+1] = sim.step(F)\n",
        "    inputs = np.concatenate([states[:, :-1, :], actions], axis=-1)\n",
        "    targets = states[:, 1:, :]\n",
        "    return torch.tensor(inputs), torch.tensor(targets)\n\n",
        "optimizer = optim.Adam(brain.parameters(), lr=0.001)\n",
        "criterion = nn.MSELoss()\n\n",
        "train_loss = []\n",
        "for ep in range(1000):\n",
        "    hx, hy = generate_batch()\n",
        "    optimizer.zero_grad()\n",
        "    preds, _ = brain(hx)\n",
        "    loss = criterion(preds, hy)\n",
        "    loss.backward()\n",
        "    optimizer.step()\n",
        "    train_loss.append(loss.item())\n",
        "    if (ep+1) % 100 == 0: print(f'Episode {ep+1:04d} | Loss: {loss.item():.4f}')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Gravity Extraction Demonstration ($g \\approx 9.8$)"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "brain.eval()\n",
        "with torch.no_grad():\n",
        "    print('Testing implicit physics discovery (Expected Δv = -0.196)')\n",
        "    for y_i in [2.0, 5.0, 8.0, 12.0]:\n",
        "        xin = torch.tensor([[[y_i, 0.0, 0.0]]], dtype=torch.float32)\n",
        "        pred, _ = brain(xin)\n",
        "        print(f'Init Height: {y_i} | Predicted Δv: {pred[0,0,1].item():.4f}')\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "plt.figure(figsize=(8,3))\n",
        "plt.plot(train_loss)\n",
        "plt.title('Brain 1 Training Loss')\n",
        "plt.xlabel('Episodes'); plt.ylabel('MSE')\n",
        "plt.yscale('log')\n",
        "plt.grid(True)\n",
        "plt.show()\n"
    ]}
]

nb_dict["cells"].extend(new_cells)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2)

print(f"✅ Appended Part C (Training and Plots) to notebook: {nb_path}")
