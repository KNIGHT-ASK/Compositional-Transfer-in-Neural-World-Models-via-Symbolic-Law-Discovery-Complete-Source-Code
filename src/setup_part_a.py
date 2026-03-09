import os
import json

workspace = r"d:\ALL MY ML AND AI"

class BouncingBallSim:
    def __init__(self, y0=5.0, v0=0.0, g=9.8, m=1.0, dt=0.02, epsilon=0.8):
        self.y = y0; self.v = v0; self.g = g; self.m = m
        self.dt = dt; self.epsilon = epsilon; self.time = 0.0
        self.history_e = [self.get_energy()]
        
    def get_energy(self):
        return 0.5 * self.m * (self.v ** 2) + self.m * self.g * self.y
        
    def step(self, F=0.0):
        a = -self.g + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        
        # Inelastic boundary condition
        if self.y < 0:
            self.y = 0
            self.v = -self.epsilon * self.v
            
        self.time += self.dt
        self.history_e.append(self.get_energy())

print("\n=== CASM Brain 1: Part A Theoretical Verifications ===")
sim = BouncingBallSim(y0=5.0, v0=0.0)
print(f"[Initial] Phase Space: y={sim.y:.2f}m, v={sim.v:.2f}m/s")
print(f"[Initial] Energy Hamiltonian: {sim.get_energy():.2f} Joules")

# Run free-fall until first contact with floor manifold
while sim.y > 0 and sim.v <= 0:
    sim.step()
    if sim.y == 0: break

print(f"\n[Test 1] Fall Time | True: ~1.01s | Actual: {sim.time:.3f}s")
print(f"[Test 2] Energy    | Expected: 49.00J | Simulated pre-bounce: {sim.history_e[-2]:.2f}J (Euler drift: {(sim.history_e[-2]-49.0):.3f}J)")

# Run until the velocity reaches zero (apex of first bounce)
while sim.v > 0: 
    sim.step()

theo_apex = 5.0 * (0.8**2)
print(f"[Test 3] Apex 1    | True: ~{theo_apex:.2f}m | Actual: {sim.y:.2f}m")
print("======================================================")

nb_dict = {
  "cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# PART A: The Simulation (The World)\n", "State space $\\mathbf{S}_t = [y_t, v_t]^T$"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import math\n\n",
        "class BouncingBallSim:\n",
        "    def __init__(self, y0=5.0, v0=0.0, g=9.8, m=1.0, dt=0.02, epsilon=0.8):\n",
        "        self.y = y0; self.v = v0; self.g = g; self.m = m\n",
        "        self.dt = dt; self.epsilon = epsilon; self.time = 0.0\n",
        "        self.history_t = [self.time]\n",
        "        self.history_y = [self.y]\n",
        "        self.history_v = [self.v]\n",
        "        self.history_e = [self.get_energy()]\n\n",
        "    def get_state(self):\n",
        "        return np.array([self.y, self.v], dtype=np.float32)\n\n",
        "    def get_energy(self):\n",
        "        return 0.5 * self.m * (self.v ** 2) + self.m * self.g * self.y\n\n",
        "    def step(self, F=0.0):\n",
        "        a = -self.g + (F / self.m)\n",
        "        self.v += a * self.dt\n",
        "        self.y += self.v * self.dt\n",
        "        if self.y < 0:\n",
        "            self.y = 0\n",
        "            self.v = -self.epsilon * self.v\n",
        "        self.time += self.dt\n",
        "        self.history_t.append(self.time)\n",
        "        self.history_y.append(self.y)\n",
        "        self.history_v.append(self.v)\n",
        "        self.history_e.append(self.get_energy())\n",
        "        return self.get_state()\n",
        "\n# Ready for instantiation."
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Physical Verification & Visualization"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "sim = BouncingBallSim(y0=5.0, v0=0.0)\n",
        "for _ in range(250):\n",
        "    sim.step()\n",
        "fig, ax1 = plt.subplots(figsize=(10, 5))\n",
        "color = 'tab:blue'\n",
        "ax1.set_xlabel('Time (s)')\n",
        "ax1.set_ylabel('Height (m)', color=color)\n",
        "ax1.plot(sim.history_t, sim.history_y, color=color, linewidth=2, label='Position (y)')\n",
        "ax1.tick_params(axis='y', labelcolor=color)\n",
        "ax1.grid(True, alpha=0.3)\n",
        "ax2 = ax1.twinx()\n",
        "color = 'tab:red'\n",
        "ax2.set_ylabel('Velocity (m/s)', color=color)\n",
        "ax2.plot(sim.history_t, sim.history_v, color=color, linestyle='--', label='Velocity (v)')\n",
        "ax2.tick_params(axis='y', labelcolor=color)\n",
        "fig.tight_layout()\n",
        "plt.title('CASM Environment: The World Simulation')\n",
        "plt.show()\n"
    ]}
  ],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
  },
  "nbformat": 4, "nbformat_minor": 4
}

nb_path = os.path.join(workspace, "Brain1_PartA_Simulation.ipynb")
os.makedirs(workspace, exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2)

print(f"\n✅ Jupyter Notebook safely generated at: {nb_path}")
