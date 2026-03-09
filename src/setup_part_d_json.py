import json
import os

def append_to_notebook():
    notebook_path = os.path.join(os.getcwd(), "CASM_Brain1_Full.ipynb")
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# PART D: Brain 1 v1.1 \u2014 The MLP Residual Fix\n",
            "\n",
            "Based on the adversarial diagnosis, the GRU was pattern-matching sequence histories. We now implement the theoretically correct architecture for learning local Markovian dynamics:\n",
            "1. **Architecture:** MLP (No memory, No hidden state)\n",
            "2. **Target:** Predict Change ($\\Delta y, \\Delta v$), not absolute state.\n",
            "3. **Data:** Train on shuffled, individual transitions, not ordered sequences.\n"
        ]
    }

    code_cell_1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import random\n",
            "from torch.utils.data import TensorDataset, DataLoader\n",
            "\n",
            "# ---------------------------------------------------------\n",
            "# 1. The Fixed Architecture: MLP for Residuals\n",
            "# ---------------------------------------------------------\n",
            "class Brain1_v2(nn.Module):\n",
            "    def __init__(self):\n",
            "        super(Brain1_v2, self).__init__()\n",
            "        # Input: [y_t, v_t, F_t] -> Output: [\\Delta y, \\Delta v]\n",
            "        self.net = nn.Sequential(\n",
            "            nn.Linear(3, 64),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(64, 64),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(64, 2)\n",
            "        )\n",
            "        \n",
            "    def forward(self, x):\n",
            "        return self.net(x)\n",
            "\n",
            "../models/brain1_v2 = Brain1_v2()\n",
            "print(f\"Brain 1 v1.1 Parameter Count: {sum(p.numel() for p in brain1_v2.parameters())}\")\n",
            "\n",
            "# ---------------------------------------------------------\n",
            "# 2. Collect Shuffled Training Data\n",
            "# ---------------------------------------------------------\n",
            "print(\"Collecting 50,000 random transitions...\")\n",
            "data_pool = []\n",
            "\n",
            "for _ in range(1000):\n",
            "    sim = BouncingBallSim()\n",
            "    # Randomize initial state \n",
            "    sim.state[0] = random.uniform(1.0, 10.0) # y\n",
            "    sim.state[1] = random.uniform(-5.0, 5.0) # v\n",
            "    \n",
            "    for _ in range(50):\n",
            "        y_t, v_t = sim.state.copy()\n",
            "        F_t = random.uniform(-2.0, 2.0)\n",
            "        \n",
            "        sim.step(F_t)\n",
            "        \n",
            "        y_t1, v_t1 = sim.state.copy()\n",
            "        \n",
            "        # Calculate true residuals (change)\n",
            "        dy = y_t1 - y_t\n",
            "        dv = v_t1 - v_t\n",
            "        \n",
            "        data_pool.append([y_t, v_t, F_t, dy, dv])\n",
            "\n",
            "# Convert to tensors\n",
            "import numpy as np\n",
            "data_tensor = torch.FloatTensor(data_pool)\n",
            "\n",
            "# Inputs: [y_t, v_t, F_t]\n",
            "X = data_tensor[:, :3]\n",
            "# Targets: [\\Delta y, \\Delta v]\n",
            "Y = data_tensor[:, 3:]\n",
            "\n",
            "# Create DataLoader for random batching\n",
            "dataset = TensorDataset(X, Y)\n",
            "dataloader = DataLoader(dataset, batch_size=256, shuffle=True)\n",
            "\n",
            "print(f\"Dataset created with {len(dataset)} transitions. Ready for training.\")\n"
        ]
    }

    code_cell_2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------\n",
            "# 3. Train on Shuffled Transitions\n",
            "# ---------------------------------------------------------\n",
            "optimizer_v2 = optim.Adam(brain1_v2.parameters(), lr=0.001)\n",
            "criterion = nn.MSELoss()\n",
            "\n",
            "epochs = 100\n",
            "loss_history_v2 = []\n",
            "\n",
            "print(\"Training Brain 1 v1.1 (MLP Residuals)...\")\n",
            "for epoch in range(epochs):\n",
            "    epoch_loss = 0.0\n",
            "    for batch_X, batch_Y in dataloader:\n",
            "        \n",
            "        # Forward pass\n",
            "        predictions = brain1_v2(batch_X)\n",
            "        \n",
            "        # Loss\n",
            "        loss = criterion(predictions, batch_Y)\n",
            "        \n",
            "        # Backward and optimize\n",
            "        optimizer_v2.zero_grad()\n",
            "        loss.backward()\n",
            "        optimizer_v2.step()\n",
            "        \n",
            "        epoch_loss += loss.item()\n",
            "        \n",
            "    avg_loss = epoch_loss / len(dataloader)\n",
            "    loss_history_v2.append(avg_loss)\n",
            "    \n",
            "    if (epoch + 1) % 10 == 0:\n",
            "        print(f\"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}\")\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "plt.plot(loss_history_v2, color='green')\n",
            "plt.title('Brain 1 v1.1 Training Loss (Predicting Residuals)')\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Mean Squared Error')\n",
            "plt.yscale('log')\n",
            "plt.grid(True)\n",
            "plt.show()\n"
        ]
    }

    code_cell_3 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ---------------------------------------------------------\n",
            "# 4. The Critical Tests\n",
            "# ---------------------------------------------------------\n",
            "../models/brain1_v2.eval()\n",
            "print(\"==== GRAVITY EXTRACTION TEST (v1.1) ====\")\n",
            "print(\"Hypothesis: \\\\Delta v should be approx -0.196 regardless of height.\")\n",
            "\n",
            "test_heights = [2.0, 5.0, 8.0, 12.0, 20.0, 50.0]\n",
            "with torch.no_grad():\n",
            "    for h in test_heights:\n",
            "        # [y=h, v=0, F=0]\n",
            "        test_input = torch.tensor([[h, 0.0, 0.0]], dtype=torch.float32)\n",
            "        prediction = brain1_v2(test_input)\n",
            "        pred_dv = prediction[0, 1].item()\n",
            "        print(f\"Height {h:4.1f} -> Predicted \\\\Delta v = {pred_dv:7.4f}\")\n",
            "\n",
            "print(\"\\n==== FORCE RESPONSE TEST (v1.1) ====\")\n",
            "print(\"Hypothesis: \\\\Delta v = (-g + F/m) * dt\")\n",
            "print(\"Expected: a = -9.8 + F -> \\\\Delta v = a * 0.02\")\n",
            "\n",
            "test_forces = [-5.0, -2.0, 0.0, 2.0, 5.0, 9.8]\n",
            "with torch.no_grad():\n",
            "    for f in test_forces:\n",
            "        # [y=5.0, v=0, F=f]\n",
            "        test_input = torch.tensor([[5.0, 0.0, f]], dtype=torch.float32)\n",
            "        prediction = brain1_v2(test_input)\n",
            "        pred_dv = prediction[0, 1].item()\n",
            "        \n",
            "        # Theoretical calculation\n",
            "        theoretical_a = -9.8 + f\n",
            "        theoretical_dv = theoretical_a * 0.02\n",
            "        \n",
            "        print(f\"Force {f:4.1f} -> Predicted \\\\Delta v = {pred_dv:7.4f} (Theoretical: {theoretical_dv:7.4f})\")\n"
        ]
    }

    nb['cells'].extend([markdown_cell, code_cell_1, code_cell_2, code_cell_3])
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print("Successfully appended Part D (Brain 1 v1.1) to CASM_Brain1_Full.ipynb")

if __name__ == "__main__":
    append_to_notebook()
