import numpy as np
import matplotlib.pyplot as plt
import math

class SpringSim:
    def __init__(self, x0=3.0, v0=0.0, k=2.0, m=1.0, dt=0.02):
        self.x = x0
        self.v = v0
        self.k = k
        self.m = m
        self.dt = dt
        self.time = 0.0
        
        # Trajectory tracking
        self.history_t = [self.time]
        self.history_x = [self.x]
        self.history_v = [self.v]
        self.history_e = [self.get_energy()]
        
    def get_state(self):
        return np.array([self.x, self.v], dtype=np.float32)
        
    def get_energy(self):
        # E = 1/2 m v^2 + 1/2 k x^2
        return 0.5 * self.m * (self.v ** 2) + 0.5 * self.k * (self.x ** 2)
        
    def step(self, F=0.0):
        # a = (-k*x + F)/m
        a = (-self.k * self.x + F) / self.m
        self.v += a * self.dt
        self.x += self.v * self.dt
        self.time += self.dt
        
        self.history_t.append(self.time)
        self.history_x.append(self.x)
        self.history_v.append(self.v)
        self.history_e.append(self.get_energy())
        
        return self.get_state()

def run_tests():
    print("--- ENVIRONMENT B VERIFICATION TESTS ---")
    sim = SpringSim(x0=3.0, v0=0.0)
    
    max_right = 3.0
    max_left = 3.0
    
    for _ in range(250):
        sim.step()
        if sim.x > max_right:
            max_right = sim.x
        if sim.x < max_left:
            max_left = sim.x
            
    print(f"Test 1 (Period Check): Theoretical period is ~4.44s. At t={sim.history_t[222]:.3f}s, x={sim.history_x[222]:.3f}m, expected x=~3.0")
    print(f"Test 2 (Energy): Initial E = {sim.history_e[0]:.4f} J, Final E = {sim.history_e[-1]:.4f} J")
    print(f"Test 3 (Symmetry): Max Right = {max_right:.3f}, Max Left = {max_left:.3f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)', color=color)
    ax1.plot(sim.history_t, sim.history_x, color=color, linewidth=2, label='Position (x)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Velocity (m/s)', color=color)
    ax2.plot(sim.history_t, sim.history_v, color=color, linestyle='--', label='Velocity (v)')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('Environment B: Spring Oscillation')
    plt.savefig("spring_trajectory.png")
    print("Saved plot to spring_trajectory.png")

if __name__ == "__main__":
    run_tests()
