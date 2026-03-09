import numpy as np
import matplotlib.pyplot as plt

class CombinedSim:
    def __init__(self, y0=0.0, v0=0.0, g=9.8, k=2.0, m=1.0, dt=0.02):
        self.y = y0
        self.v = v0
        self.g = g
        self.k = k
        self.m = m
        self.dt = dt
        self.time = 0.0
        
        # Trajectory tracking
        self.history_t = [self.time]
        self.history_y = [self.y]
        self.history_v = [self.v]
        
    def get_state(self):
        return np.array([self.y, self.v], dtype=np.float32)
        
    def step(self, F=0.0):
        # Both forces combined: a = -g - (k*y/m) + F/m
        # But wait! If gravity pulls DOWN (negative), and spring pulls TOWARD y=0...
        # Wait, the prompt says: a_t = -g + (-k*y_t/m) + (F_t/m) 
        a = -self.g - (self.k * self.y / self.m) + (F / self.m)
        self.v += a * self.dt
        self.y += self.v * self.dt
        self.time += self.dt
        
        self.history_t.append(self.time)
        self.history_y.append(self.y)
        self.history_v.append(self.v)
        
        return self.get_state()

def run_tests():
    print("--- ENVIRONMENT C VERIFICATION TESTS ---")
    
    # Testing equilibrium point
    # Release at natural spring rest without velocity
    sim = CombinedSim(y0=0.0, v0=0.0)
    
    equilibrium_y = -4.9
    for _ in range(500):
        sim.step()
        
    center_measured = (max(sim.history_y) + min(sim.history_y)) / 2.0
    print(f"Test 1 (Equilibrium Point): Expected = {equilibrium_y:.2f} m, Measured Center = {center_measured:.2f} m")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Vertical Position y (m)', color=color)
    ax1.plot(sim.history_t, sim.history_y, color=color, linewidth=2, label='Position (y)')
    ax1.axhline(y=equilibrium_y, color='k', linestyle=':', alpha=0.5, label='Equilibrium y=-4.9')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Velocity (m/s)', color=color)
    ax2.plot(sim.history_t, sim.history_v, color=color, linestyle='--', label='Velocity (v)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.title('Environment C: Gravity + Spring Combined')
    plt.savefig("combined_trajectory.png")
    print("Saved plot to combined_trajectory.png")

if __name__ == "__main__":
    run_tests()
