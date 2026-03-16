import numpy as np
import matplotlib.pyplot as plt

class SymplecticOrbital3D:
    def __init__(self, x, y, z, vx, vy, vz, dt=0.001):
        self.s = np.array([x, y, z, vx, vy, vz], dtype=float)
        self.GM = 1.0
        self.dt = dt
        self.epsilon = 0.001 # Minimal softening for numerical safety

    def get_accel(self, pos):
        r = np.linalg.norm(pos)
        r_soft = np.sqrt(r**2 + self.epsilon**2)
        return -self.GM * pos / (r_soft**3)

    def step(self):
        pos = self.s[0:3]
        vel = self.s[3:6]
        
        # 1. Half-step velocity
        a = self.get_accel(pos)
        vel_half = vel + 0.5 * a * self.dt
        
        # 2. Full-step position
        pos_new = pos + vel_half * self.dt
        
        # 3. Second half-step velocity
        a_new = self.get_accel(pos_new)
        vel_new = vel_half + 0.5 * a_new * self.dt
        
        self.s[0:3] = pos_new
        self.s[3:6] = vel_new
        return self.s.copy()

    def get_metrics(self):
        pos = self.s[0:3]
        vel = self.s[3:6]
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        energy = 0.5 * v**2 - self.GM / r
        angular_momentum = np.cross(pos, vel)
        L_mag = np.linalg.norm(angular_momentum)
        
        return r, energy, L_mag

if __name__ == "__main__":
    print("--- Ground Truth Symplectic Verification ---")
    # Initial condition: Circular orbit at r=1.0
    sim = SymplecticOrbital3D(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, dt=0.001)
    
    radii, energies, momenta = [], [], []
    steps = 5000 # 5 seconds of sim time
    
    for _ in range(steps):
        r, e, l = sim.get_metrics()
        radii.append(r)
        energies.append(e)
        momenta.append(l)
        sim.step()
        
    drift_e = abs(energies[-1] - energies[0])
    drift_l = abs(momenta[-1] - momenta[0])
    
    print(f"Final Energy Drift: {drift_e:.10f}")
    print(f"Final L-Momentum Drift: {drift_l:.10f}")
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(radii); axs[0].set_title("Radius over Time"); axs[0].set_ylabel("r")
    axs[1].plot(energies); axs[1].set_title("Energy Conservation"); axs[1].set_ylabel("Total Energy")
    axs[1].axhline(energies[0], color='r', linestyle='--')
    axs[2].plot(momenta); axs[2].set_title("Angular Momentum Conservation"); axs[2].set_ylabel("|L|")
    axs[2].axhline(momenta[0], color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("ground_truth_symplectic.png")
    print("Result: Verification plot saved to ground_truth_symplectic.png")
