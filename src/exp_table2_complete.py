"""
Experiment: Complete Table 2 Metrics — All Rows, All Metrics, n=5 Seeds
=======================================================================
W3 + W5 fix for JMLR review.

PHYSICS & FAIRNESS:
  ✓ Modular components: 267k params, 400 epochs per module.
  ✓ Monolith (Elite/Rich): 1M params, 510-600 epochs. (4x capacity match).
  ✓ Approach B: dx = v·dt (once), dv = Σ modules.
  ✓ SymLog: SYMLOG_K = 0.01 for inverse-square stability.
  ✓ Metrics: 1-step MSE sliced [:, 3:] (velocity-only) for fair comparison.

P100 OPTIMIZATIONS:
  ✓ Vectorized make_dataset: GPU-parallel simulation removes Python bottleneck.
  ✓ No torch.compile.
  ✓ Batch size 65386.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json, time
from itertools import combinations_with_replacement

# ══════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════
assert torch.cuda.is_available(), "Requires CUDA GPU!"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("=" * 65)
print("  TABLE 2 COMPLETE — Precision Optimized")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
DT             = 0.02
GM             = 10.0
OMEGA_Z        = 0.5
C_W            = 0.1
SYMLOG_K       = 0.1

N_TRAIN_MOD    = 150_000
N_TRAIN_RICH   = 100_000
N_TRAIN_ELITE  = 500_000
N_TEST         = 8_000
N_ROLLOUT      = 500
N_SEEDS        = 5
BATCH          = 65_536
EPOCHS_MOD     = 400
EPOCHS_MONO    = 400 # Fix Bug 3: Budget parity (400 vs 500)
EPOCHS_ELITE   = 600
LR             = 1e-3

# ══════════════════════════════════════════════════════════════
#  MATH
# ══════════════════════════════════════════════════════════════
def symlog(x): return torch.sign(x) * torch.log(1.0 + torch.abs(x) / SYMLOG_K)
def symexp(y): return torch.sign(y) * SYMLOG_K * (torch.exp(torch.abs(y)) - 1.0)

def gravity_acc(S):
    pos = S[:,:3]; r = torch.norm(pos,dim=1,keepdim=True).clamp(min=1e-6)
    return (-GM*pos/r**3)*DT

def coriolis_acc(S):
    ax, ay = 2*OMEGA_Z*S[:,4:5], -2*OMEGA_Z*S[:,3:4]
    return torch.cat([ax, ay, torch.zeros_like(ax)], dim=1)*DT

def wind_acc(S): return (-C_W*S[:,3:])*DT

def combined_acc(S): return gravity_acc(S) + coriolis_acc(S) + wind_acc(S)

def triple_full_delta(S):
    dv = combined_acc(S); dx = S[:,3:]*DT  # Fix Bug 1: Use current velocity for dx
    return torch.cat([dx, dv], dim=1)

# ══════════════════════════════════════════════════════════════
#  DATA GENERATION (Vectorized)
# ══════════════════════════════════════════════════════════════
def sample_orbit_states(n, seed=None):
    if seed is not None: torch.manual_seed(seed)
    with torch.no_grad():
        lr = torch.empty(n,device=device).uniform_(np.log(2.0),np.log(8.0))
        r = torch.exp(lr); theta = torch.empty(n,device=device).uniform_(0,2*np.pi); phi=torch.empty(n,device=device).uniform_(-np.pi/4,np.pi/4)
        x, y, z = r*torch.cos(theta)*torch.cos(phi), r*torch.sin(theta)*torch.cos(phi), r*torch.sin(phi)
        v = torch.sqrt(GM/r)*(0.7+0.4*torch.rand(n,device=device))
        vx = -torch.sin(theta)*v + torch.empty(n,device=device).uniform_(-0.5,0.5)
        vy =  torch.cos(theta)*v + torch.empty(n,device=device).uniform_(-0.5,0.5)
        vz = torch.empty(n,device=device).uniform_(-0.5,0.5)
        return torch.stack([x,y,z,vx,vy,vz],dim=1)

def make_module_dataset(acc_fn, n, seed=None):
    step=50; traj=n//step; S=sample_orbit_states(traj,seed=seed)
    XL, YL = [], []
    for _ in range(step):
        XL.append(S.clone()); dv=acc_fn(S); YL.append(symlog(dv))
        S = (S + torch.cat([S[:,3:]*DT, dv], dim=1)).detach() # Fix Bug 2: .detach()
    X, Y = torch.stack(XL,1).view(n,6), torch.stack(YL,1).view(n,3)
    idx = torch.randperm(n); return X[idx], Y[idx] # Fix Bug 2: Shuffle

def make_monolith_dataset(n, seed=None):
    step=50; traj=n//step; S=sample_orbit_states(traj,seed=seed)
    XL, YL = [], []
    for _ in range(step):
        XL.append(S.clone()); ds=triple_full_delta(S); YL.append(ds)
        S = (S + ds).detach() # Fix Bug 2: .detach()
    X, Y = torch.stack(XL,1).view(n,6), torch.stack(YL,1).view(n,6)
    idx = torch.randperm(n); return X[idx], Y[idx] # Fix Bug 2: Shuffle

def make_eval_set(n=N_TEST, seed=None):
    S = sample_orbit_states(n, seed=seed); dv=combined_acc(S); dx=S[:,3:]*DT
    return S, torch.cat([dx, dv], dim=1)

# ══════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════
class ModularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6,512), nn.Tanh(), nn.Linear(512,512), nn.Tanh(), nn.Linear(512,3))
    def forward(self,x): return self.net(x)

class MonolithicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6,1024), nn.Tanh(), nn.Linear(1024,1024), nn.Tanh(), nn.Linear(1024,6))
    def forward(self,x): return self.net(x)

def train_model(model, X, Y, epochs, lr=LR, tag=""):
    model = model.to(device); n=X.shape[0]
    Xm, Xs = X.mean(0), X.std(0)+1e-8; Ym, Ys = Y.mean(0), Y.std(0)+1e-8
    Xn, Yn = (X-Xm)/Xs, (Y-Ym)/Ys
    opt = optim.Adam(model.parameters(), lr=lr); sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs); scaler = torch.amp.GradScaler('cuda')
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0,n,BATCH):
            ii = idx[s:s+BATCH]; opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'): loss=F.mse_loss(model(Xn[ii]), Yn[ii])
            scaler.scale(loss).backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        sched.step()
    model.eval()
    return model, (Xm,Xs,Ym,Ys)

@torch.no_grad()
def compose_modular(mods, S):
    dx = S[:,3:]*DT; dv = torch.zeros_like(dx)
    for m,(Xm,Xs,Ym,Ys) in mods:
        Xn = (S-Xm)/Xs
        with torch.amp.autocast('cuda'): p = m(Xn)
        dv += symexp(p*Ys+Ym)
    return torch.cat([dx, dv], 1)

@torch.no_grad()
def predict_monolith(mn, S):
    m,(Xm,Xs,Ym,Ys) = mn; Xn=(S-Xm)/Xs
    with torch.amp.autocast('cuda'): p=m(Xn)
    return p*Ys+Ym

@torch.no_grad()
def one_step_mse(pred_fn, Xs, Yd): return F.mse_loss(pred_fn(Xs)[:,3:], Yd[:,3:]).item()*1e4

@torch.no_grad()
def rollout_and_drift(pred_fn, seed, n_traj=20, n_steps=N_ROLLOUT):
    S0 = sample_orbit_states(n_traj, seed=seed)
    E0 = 0.5*torch.sum(S0[:,3:]**2, 1) - GM/torch.linalg.norm(S0[:,:3], dim=1)
    sp, st = S0.clone(), S0.clone()
    ma = torch.zeros(n_traj, device=device)
    for _ in range(n_steps):
        dv, dx = combined_acc(st), st[:,3:]*DT; st = (st+torch.cat([dx,dv],1)).detach()
        dsp = pred_fn(sp); sp = (sp+dsp).detach()
        ma += torch.mean((sp-st)**2, 1)
    E1 = 0.5*torch.sum(sp[:,3:]**2, 1) - GM/torch.linalg.norm(sp[:,:3], dim=1).clamp(min=1e-9)
    return torch.mean(ma/n_steps).item()*1e4, torch.mean(torch.abs(E1-E0)).item()

def count_params(model): return sum(p.numel() for p in model.parameters())

def sindy_compose(mods, seed, n_p=5000):
    from sklearn.linear_model import Lasso
    S = sample_orbit_states(n_p, seed=seed)
    with torch.no_grad(): D = compose_modular(mods, S)
    S_np, D_np = S.cpu().numpy(), D.cpu().numpy()
    pairs = list(combinations_with_replacement(range(6),2))
    Phi = np.column_stack([np.ones((n_p,1)), S_np, np.column_stack([S_np[:,i]*S_np[:,j] for i,j in pairs])]).astype(np.float32)
    # Fix Bug 6: Only fit velocity dimensions (indices 3,4,5)
    coefs_v = np.zeros((Phi.shape[1], 3), dtype=np.float32)
    for d in range(3):
        reg = Lasso(alpha=1e-4, max_iter=20000, fit_intercept=False).fit(Phi, D_np[:,3+d])
        coefs_v[:,d] = reg.coef_
    ct = torch.from_numpy(coefs_v).to(device)
    def pred(Sg):
        n=Sg.shape[0]; Ph = torch.cat([torch.ones(n,1,device=device), Sg, torch.stack([Sg[:,i]*Sg[:,j] for i,j in pairs],1)], 1).float()
        dv = Ph @ ct; dx = Sg[:,3:] * DT # Fix Bug 6: Exact kinematics
        return torch.cat([dx, dv], 1)
    return pred

if __name__ == "__main__":
    t0 = time.time(); rows = ['modular_ensemble','monolith_fair','monolith_elite','monolith_rich','sindy_composed']
    accum = {r: {m:[] for m in ['m1','me','mt']} for r in rows}
    for seed in range(N_SEEDS):
        s = seed*113+7; torch.manual_seed(s); np.random.seed(s)
        Xg,Yg = make_module_dataset(gravity_acc, N_TRAIN_MOD,s); Xc,Yc = make_module_dataset(coriolis_acc, N_TRAIN_MOD,s+1); Xw,Yw = make_module_dataset(wind_acc, N_TRAIN_MOD,s+2)
        Xri,Yri = make_monolith_dataset(N_TRAIN_RICH, s+3); Xel,Yel = make_monolith_dataset(N_TRAIN_ELITE, s+4)
        Xev, Yev = make_eval_set(seed=s+99)
        # Modular
        mg,ng = train_model(ModularMLP(), Xg, Yg, EPOCHS_MOD, tag="ModG"); mc,nc = train_model(ModularMLP(), Xc, Yc, EPOCHS_MOD, tag="ModC"); mw,nw = train_model(ModularMLP(), Xw, Yw, EPOCHS_MOD, tag="ModW")
        en = [(mg,ng),(mc,nc),(mw,nw)]; ep = lambda x: compose_modular(en, x)
        m1 = one_step_mse(ep,Xev,Yev); mt,me = rollout_and_drift(ep,s+300) # Fix Bug 4: mt (traj), me (energy)
        accum['modular_ensemble']['m1'].append(m1); accum['modular_ensemble']['mt'].append(mt); accum['modular_ensemble']['me'].append(me)
        # SINDy
        sd = sindy_compose(en, s+400); m1 = one_step_mse(sd,Xev,Yev); mt,me = rollout_and_drift(sd,s+304) # Fix Bug 4: mt, me
        accum['sindy_composed']['m1'].append(m1); accum['sindy_composed']['mt'].append(mt); accum['sindy_composed']['me'].append(me)
        # Monoliths
        def run_m(tag, X, Y, ep, row, seed, Xev, Yev): # Fix Bug 5: Safe arguments
            m, n = train_model(MonolithicMLP(), X, Y, ep, tag=tag); fn = lambda x: predict_monolith((m,n),x)
            m1 = one_step_mse(fn,Xev,Yev); mt,me = rollout_and_drift(fn,seed+300) # Fix Bug 4 & 5
            accum[row]['m1'].append(m1); accum[row]['mt'].append(mt); accum[row]['me'].append(me)
        
        # Note: Elite monolith gets ~1.8x more grad steps than ensemble (Bug 9 documentation)
        run_m("Rich", Xri, Yri, EPOCHS_MONO, 'monolith_rich', s, Xev, Yev); run_m("Elite", Xel, Yel, EPOCHS_ELITE, 'monolith_elite', s, Xev, Yev)
        # Fix Bug 3: Fair data parity (N_TRAIN_MOD*3 vs n3*3)
        Xf, Yf = make_monolith_dataset(N_TRAIN_MOD*3, s+10); run_m("Fair", Xf, Yf, EPOCHS_MONO, 'monolith_fair', s, Xev, Yev)
        
        # Bug 8: Memory cleanup between seeds
        del Xg, Yg, Xc, Yc, Xw, Yw, Xri, Yri, Xel, Yel, Xev, Yev, Xf, Yf
        torch.cuda.empty_cache()
    final = {r: {m: {'mean': np.mean(v), 'std': np.std(v)} for m,v in d.items()} for r,d in accum.items()}
    with open('table2_complete_results.json','w') as f: json.dump(final, f, indent=2)
    print("\n" + "="*70 + "\n  FINAL AGGREGATE (n=5)\n" + "="*70)
    for r in rows:
        d = final[r]; print(f"  {r.upper():<20} | 1s: {d['m1']['mean']:.4f} | E: {d['me']['mean']:.6f} | Traj: {d['mt']['mean']:.2f}")
    print(f"\n[DONE] Total: {(time.time()-t0)/60:.1f} min")
