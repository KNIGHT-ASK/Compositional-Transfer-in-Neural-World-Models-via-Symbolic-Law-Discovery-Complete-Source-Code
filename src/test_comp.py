import torch

DT = 0.01

def test_composition():
    # Simulate a single particle in 1D
    v = torch.tensor([[2.0]])
    a1 = torch.tensor([[1.0]])  # gravity
    a2 = torch.tensor([[3.0]])  # spring
    
    # Ground truth combined position change
    # y = v*dt + 0.5*(a1+a2)*dt^2
    gt_pos = v*DT + 0.5*(a1+a2)*(DT**2)
    
    # Individual model predictions
    p1_pos = v*DT + 0.5*a1*(DT**2)
    p2_pos = v*DT + 0.5*a2*(DT**2)
    
    # Bugged composition (used in scalling_experiment.py)
    bugged_pos = p1_pos
    
    # Corrected composition
    corrected_pos = p1_pos + p2_pos - v*DT
    
    print(f"Ground Truth : {gt_pos.item()}")
    print(f"Bugged Comp  : {bugged_pos.item()} (Error: {abs(gt_pos.item() - bugged_pos.item())})")
    print(f"Correct Comp : {corrected_pos.item()} (Error: {abs(gt_pos.item() - corrected_pos.item())})")

test_composition()
