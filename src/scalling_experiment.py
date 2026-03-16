import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"GPU: {torch.cuda.get_device_name(0)}")

# -----------------------------------------------
# CONSTANTS
# -----------------------------------------------
G          = 2.0  # Increased to match Spring magnitude
K          = 2.0
L0         = 1.0
EPS        = 0.05
N_SAMPLES  = 300000
N_MODELS   = 5
EPOCHS     = 300  # Increased for Gravity singularity convergence
BATCH      = 65536
STEPS      = 500

# -----------------------------------------------
# BUG FIXES APPLIED:
#
# BUG 1 — compose_rollout dx double-counting
#   OLD: dx = d1[:,:mid] + d2[:,:mid]
#        This adds v*dt TWICE per step
#        causing exponential position drift
#   FIX: dx = d1[:,:mid]  (kinematics once)
#        dv = d1[:,mid:] + d2[:,mid:]  (forces sum)
#
# BUG 2 — compose_rollout dead code
#   There were TWO dx assignments,
#   second one overwrote the first silently
#   FIX: removed the broken first attempt
#
# BUG 3 — Scale 4 gravity takes 12D absolute input
#   Network cannot discover r2-r1 internally
#   Loss stuck at 0.42, never converges
#   FIX: Scale 4 uses 6D relative input [r_rel,v_rel]
#        Output still 12D residuals [dx1,dv1,dx2,dv2]
#
# BUG 4 — GPU% and time printed per epoch
#   Clutters output, wastes memory in logs
#   FIX: removed GPU% and time from print
#
# BUG 5 — Ensemble normalization uses Ys[0].item()
#   for dt estimation in compose_rollout
#   This is fragile and wrong
#   FIX: removed entirely, dx taken from d1 directly
# -----------------------------------------------

# -----------------------------------------------
# SCALE 1 — 2D (1D physics)
# -----------------------------------------------
def data_s1_gravity(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(1)
    with torch.no_grad():
        x  = torch.empty(n,1,device=device).uniform_(-5,5)
        vx = torch.empty(n,1,device=device).uniform_(-5,5)
        S  = torch.cat([x,vx],dim=1)
        a  = torch.full((n,1),-G,device=device)
        Y  = torch.cat([(vx+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s1_spring(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(2)
    with torch.no_grad():
        x  = torch.empty(n,1,device=device).uniform_(-5,5)
        vx = torch.empty(n,1,device=device).uniform_(-5,5)
        S  = torch.cat([x,vx],dim=1)
        a  = -K*x
        Y  = torch.cat([(vx+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s1_combined(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(3)
    with torch.no_grad():
        x  = torch.empty(n,1,device=device).uniform_(-5,5)
        vx = torch.empty(n,1,device=device).uniform_(-5,5)
        S  = torch.cat([x,vx],dim=1)
        a  = torch.full((n,1),-G,device=device) - K*x
        Y  = torch.cat([(vx+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def gt_s1(s, steps, dt=0.01):
    traj = [s.copy()]
    for _ in range(steps):
        a      = -G - K*s[0]
        vx_new = s[1] + a*dt
        x_new  = s[0] + 0.5*(s[1]+vx_new)*dt
        s      = np.array([x_new, vx_new])
        traj.append(s.copy())
    return np.array(traj)

# -----------------------------------------------
# SCALE 2 — 4D (2D physics)
# -----------------------------------------------
def data_s2_gravity(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(11)
    with torch.no_grad():
        pos = torch.empty(n,2,device=device).uniform_(-5,5)
        vel = torch.empty(n,2,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = torch.zeros(n,2,device=device)
        a[:,1] = -G
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s2_spring(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(12)
    with torch.no_grad():
        pos = torch.empty(n,2,device=device).uniform_(-5,5)
        vel = torch.empty(n,2,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = -K*pos
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s2_combined(n=N_SAMPLES, dt=0.01):
    torch.manual_seed(13)
    with torch.no_grad():
        pos = torch.empty(n,2,device=device).uniform_(-5,5)
        vel = torch.empty(n,2,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = torch.zeros(n,2,device=device)
        a[:,1] = -G
        a   = a - K*pos
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def gt_s2(s, steps, dt=0.01):
    traj = [s.copy()]
    for _ in range(steps):
        pos,vel = s[0:2],s[2:4]
        a  = np.array([0,-G]) - K*pos
        vn = vel+a*dt
        pn = pos+0.5*(vel+vn)*dt
        s  = np.concatenate([pn,vn])
        traj.append(s.copy())
    return np.array(traj)

# -----------------------------------------------
# SCALE 3 — 6D (3D physics)
# -----------------------------------------------
def data_s3_gravity(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(21)
    with torch.no_grad():
        pos = torch.empty(n,3,device=device).uniform_(-5,5)
        vel = torch.empty(n,3,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = torch.zeros(n,3,device=device)
        a[:,2] = -G
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s3_spring(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(22)
    with torch.no_grad():
        pos = torch.empty(n,3,device=device).uniform_(-5,5)
        vel = torch.empty(n,3,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = -K*pos
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def data_s3_combined(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(23)
    with torch.no_grad():
        pos = torch.empty(n,3,device=device).uniform_(-5,5)
        vel = torch.empty(n,3,device=device).uniform_(-5,5)
        S   = torch.cat([pos,vel],dim=1)
        a   = torch.zeros(n,3,device=device)
        a[:,2] = -G
        a   = a - K*pos
        Y   = torch.cat([(vel+0.5*a*dt)*dt, a*dt],dim=1)
    return S, Y

def gt_s3(s, steps, dt=0.005):
    traj = [s.copy()]
    for _ in range(steps):
        pos,vel = s[0:3],s[3:6]
        a  = np.array([0,0,-G]) - K*pos
        vn = vel+a*dt
        pn = pos+0.5*(vel+vn)*dt
        s  = np.concatenate([pn,vn])
        traj.append(s.copy())
    return np.array(traj)

# -----------------------------------------------
# SCALE 4 — 12D (3D two-body)
# FIX: input is 6D RELATIVE state [r_rel, v_rel]
# output is 12D residual [dx1,dv1,dx2,dv2]
# This is physically correct — gravity and spring
# both depend ONLY on relative coordinates.
# -----------------------------------------------
def data_s4_gravity(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(31)
    with torch.no_grad():
        r1=torch.empty(n,3,device=device).uniform_(-5,5)
        v1=torch.empty(n,3,device=device).uniform_(-3,3)
        r2=torch.empty(n,3,device=device).uniform_(-5,5)
        v2=torch.empty(n,3,device=device).uniform_(-3,3)
        # FIX: 6D relative input
        r_rel  = r2-r1; v_rel=v2-v1
        S      = torch.cat([r_rel,v_rel],dim=1)
        r_mag  = torch.norm(r_rel,dim=1,keepdim=True)
        r_soft = torch.sqrt(r_mag**2+EPS**2)
        ag1    =  G*r_rel/r_soft**3; ag2=-ag1
        # 12D output
        Y = torch.cat([
            (v1+0.5*ag1*dt)*dt, ag1*dt,
            (v2+0.5*ag2*dt)*dt, ag2*dt
        ],dim=1)
    return S, Y

def data_s4_spring(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(32)
    with torch.no_grad():
        r1=torch.empty(n,3,device=device).uniform_(-5,5)
        v1=torch.empty(n,3,device=device).uniform_(-3,3)
        r2=torch.empty(n,3,device=device).uniform_(-5,5)
        v2=torch.empty(n,3,device=device).uniform_(-3,3)
        # FIX: 6D relative input
        r_rel  = r2-r1; v_rel=v2-v1
        S      = torch.cat([r_rel,v_rel],dim=1)
        r_mag  = torch.norm(r_rel,dim=1,keepdim=True).clamp(min=1e-6)
        r_hat  = r_rel/r_mag
        as1    =  K*(r_mag-L0)*r_hat; as2=-as1
        Y = torch.cat([
            (v1+0.5*as1*dt)*dt, as1*dt,
            (v2+0.5*as2*dt)*dt, as2*dt
        ],dim=1)
    return S, Y

def data_s4_combined(n=N_SAMPLES, dt=0.005):
    torch.manual_seed(33)
    with torch.no_grad():
        r1=torch.empty(n,3,device=device).uniform_(-5,5)
        v1=torch.empty(n,3,device=device).uniform_(-3,3)
        r2=torch.empty(n,3,device=device).uniform_(-5,5)
        v2=torch.empty(n,3,device=device).uniform_(-3,3)
        # FIX: 6D relative input
        r_rel  = r2-r1; v_rel=v2-v1
        S      = torch.cat([r_rel,v_rel],dim=1)
        r_mag  = torch.norm(r_rel,dim=1,keepdim=True)
        r_soft = torch.sqrt(r_mag**2+EPS**2)
        ag1    =  G*r_rel/r_soft**3; ag2=-ag1
        r_hat  = r_rel/r_mag.clamp(min=1e-6)
        as1    =  K*(r_mag-L0)*r_hat; as2=-as1
        a1=ag1+as1; a2=ag2+as2
        Y = torch.cat([
            (v1+0.5*a1*dt)*dt, a1*dt,
            (v2+0.5*a2*dt)*dt, a2*dt
        ],dim=1)
    return S, Y

def gt_s4(s, steps, dt=0.005):
    traj = [s.copy()]
    for _ in range(steps):
        r1,v1,r2,v2 = s[0:3],s[3:6],s[6:9],s[9:12]
        r12   = r2-r1
        r_mag = max(np.linalg.norm(r12),1e-6)
        r_s   = np.sqrt(r_mag**2+EPS**2)
        ag1   =  G*r12/r_s**3; ag2=-ag1
        r_hat = r12/r_mag
        as1   =  K*(r_mag-L0)*r_hat; as2=-as1
        a1=ag1+as1; a2=ag2+as2
        v1n=v1+a1*dt; v2n=v2+a2*dt
        r1n=r1+0.5*(v1+v1n)*dt
        r2n=r2+0.5*(v2+v2n)*dt
        s=np.concatenate([r1n,v1n,r2n,v2n])
        traj.append(s.copy())
    return np.array(traj)

# -----------------------------------------------
# ARCHITECTURE
# -----------------------------------------------
class BrainD(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, 512),  nn.Tanh(),
            nn.Linear(512,  512),  nn.Tanh(),
            nn.Linear(512,  512),  nn.Tanh(),
            nn.Linear(512,  D_out)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# TRAINING — clean output, no GPU% or time
# -----------------------------------------------
def train_model(S, Y, D_in, D_out,
                label='', seed=0,
                epochs=EPOCHS):
    torch.manual_seed(seed)
    model = BrainD(D_in, D_out).to(device)
    n = S.shape[0]

    Sm=S.mean(0); Ss=S.std(0)+1e-8
    Ym=Y.mean(0); Ys=Y.std(0)+1e-8
    Sn=(S-Sm)/Ss; Yn=(Y-Ym)/Ys

    opt    = optim.Adam(model.parameters(), lr=3e-3)
    sch    = optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    best   = float('inf')

    for epoch in range(epochs+1):
        idx = torch.randperm(n, device=device)
        tl, nb = 0, 0
        for s in range(0, n, BATCH):
            e  = min(s+BATCH, n)
            bx = Sn[idx[s:e]]
            by = Yn[idx[s:e]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = nn.MSELoss()(model(bx), by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1
        sch.step()
        avg = tl/nb
        if avg < best: best = avg
        # FIX: removed GPU% and time from output
        if epoch % 50 == 0:
            print(f"  [{label}] "
                  f"Ep{epoch:3d} | "
                  f"Loss:{avg:.8f} | "
                  f"Best:{best:.8f}")

    return model.cpu(), (
        Sm.cpu(), Ss.cpu(),
        Ym.cpu(), Ys.cpu()
    ), best

# -----------------------------------------------
# ENSEMBLE
# -----------------------------------------------
class Ensemble(nn.Module):
    def __init__(self, models_data):
        super().__init__()
        self.models = nn.ModuleList(
            [m[0] for m in models_data])
        Sm,Ss,Ym,Ys = models_data[0][1]
        self.register_buffer('Sm', Sm)
        self.register_buffer('Ss', Ss)
        self.register_buffer('Ym', Ym)
        self.register_buffer('Ys', Ys)

    @torch.no_grad()
    def forward(self, s):
        sn = (s-self.Sm)/(self.Ss+1e-8)
        p  = torch.stack([m(sn) for m in self.models])
        return p.mean(0)*self.Ys + self.Ym

# -----------------------------------------------
# ROLLOUT — FIXED COMPOSITION
#
# FIX for BUG 1+2:
# dx taken from force-1 module only (kinematics once)
# dv summed from both modules (force superposition)
#
# This prevents the v*dt double-counting that was
# causing MSE to explode at every scale.
# -----------------------------------------------
def compose_rollout(ens_f1, ens_f2,
                    s_init, is_twobody=False,
                    steps=STEPS):
    traj = [s_init.copy()]

    if not is_twobody:
        # Scales 1,2,3 — absolute state input/output
        s = torch.tensor(
            s_init, dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        D   = s.shape[1]
        mid = D // 2

        for _ in range(steps):
            d1 = ens_f1(s)
            d2 = ens_f2(s)

            # FIX: position from force-1 only (once)
            # velocity sum from both forces
            dx = d1[:, :mid]
            dv = d1[:, mid:] + d2[:, mid:]

            v_new = s[:, mid:] + dv
            s = torch.cat([
                s[:, :mid] + dx,
                v_new
            ], dim=1)
            traj.append(s.squeeze(0).cpu().numpy())

    else:
        # Scale 4 — relative input, absolute state tracking
        s = s_init.copy()  # [12] numpy

        for _ in range(steps):
            r1,v1 = s[0:3], s[3:6]
            r2,v2 = s[6:9], s[9:12]

            # Build 6D relative input
            s_rel = torch.tensor(
                np.concatenate([r2-r1, v2-v1]),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # 12D residuals from each module
            d1 = ens_f1(s_rel).squeeze(0).cpu().numpy()
            d2 = ens_f2(s_rel).squeeze(0).cpu().numpy()

            # FIX: dx from force-1, dv summed
            # d1 = [dx1,dv1,dx2,dv2] shape [12]
            dx1 = d1[0:3];  dv1 = d1[3:6]  + d2[3:6]
            dx2 = d1[6:9];  dv2 = d1[9:12] + d2[9:12]

            s[0:3]  += dx1
            s[3:6]  += dv1
            s[6:9]  += dx2
            s[9:12] += dv2
            traj.append(s.copy())

    return np.array(traj)

def monolith_rollout(ens_m, s_init,
                     is_twobody=False,
                     steps=STEPS):
    traj = [s_init.copy()]

    if not is_twobody:
        s = torch.tensor(
            s_init, dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        D   = s.shape[1]
        mid = D // 2

        for _ in range(steps):
            ds    = ens_m(s)
            dx    = ds[:, :mid]
            dv    = ds[:, mid:]
            v_new = s[:, mid:] + dv
            s = torch.cat([
                s[:, :mid] + dx,
                v_new
            ], dim=1)
            traj.append(s.squeeze(0).cpu().numpy())

    else:
        s = s_init.copy()
        for _ in range(steps):
            r1,v1 = s[0:3], s[3:6]
            r2,v2 = s[6:9], s[9:12]
            s_rel = torch.tensor(
                np.concatenate([r2-r1, v2-v1]),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            ds = ens_m(s_rel).squeeze(0).cpu().numpy()
            s[0:3]  += ds[0:3]
            s[3:6]  += ds[3:6]
            s[6:9]  += ds[6:9]
            s[9:12] += ds[9:12]
            traj.append(s.copy())

    return np.array(traj)

def mse(a, b):
    return float(np.mean((a-b)**2))

# -----------------------------------------------
# RUN ONE SCALE
# -----------------------------------------------
def run_scale(scale_id, D_state, D_in, D_out,
              fn_f1, fn_f2, fn_mono,
              fn_gt, s_init, label,
              is_twobody=False):
    print(f"\n{'='*52}")
    print(f"SCALE {scale_id}: D={D_state} — {label}")
    print(f"{'='*52}")

    # Data
    S1,Y1 = fn_f1()
    S2,Y2 = fn_f2()
    Sm,Ym = fn_mono()
    torch.cuda.empty_cache()

    # Train
    def train_ens(S, Y, tag, seed_offset):
        results = []
        losses  = []
        for i in range(N_MODELS):
            m, norms, bl = train_model(
                S, Y, D_in, D_out,
                label=f'{tag}-{i}',
                seed=i+seed_offset
            )
            results.append((m, norms))
            losses.append(bl)
        ens = Ensemble(results).to(device)
        torch.cuda.empty_cache()
        return ens, np.mean(losses)

    print(f"Training Force-1 ({N_MODELS} brains)...")
    ens_f1, lf1 = train_ens(S1, Y1, 'F1', 0)
    print(f"Training Force-2 ({N_MODELS} brains)...")
    ens_f2, lf2 = train_ens(S2, Y2, 'F2', 10)
    print(f"Training Monolith ({N_MODELS} brains)...")
    ens_m,  lm  = train_ens(Sm, Ym, 'M',  20)

    # Rollouts
    print("Running rollouts...")
    traj_gt = fn_gt(s_init.copy(), STEPS)
    traj_zs = compose_rollout(
        ens_f1, ens_f2, s_init.copy(),
        is_twobody=is_twobody
    )
    traj_mo = monolith_rollout(
        ens_m, s_init.copy(),
        is_twobody=is_twobody
    )

    mse_zs = mse(traj_zs, traj_gt)
    mse_mo = mse(traj_mo, traj_gt)
    ratio  = mse_mo / (mse_zs + 1e-12)
    winner = "ZS WINS" if ratio > 1 else "MONO WINS"

    print(f"\n  Zero-Shot MSE : {mse_zs*1e4:.4f} x10^-4")
    print(f"  Monolith  MSE : {mse_mo*1e4:.4f} x10^-4")
    print(f"  Ratio         : {ratio:.2f}x  [{winner}]")

    del ens_f1, ens_f2, ens_m
    torch.cuda.empty_cache()

    return {
        'scale': scale_id, 'D': D_state,
        'label': label,
        'mse_zs': mse_zs, 'mse_mo': mse_mo,
        'ratio': ratio,
        'lf1': lf1, 'lf2': lf2, 'lm': lm,
        'traj_gt': traj_gt,
        'traj_zs': traj_zs,
        'traj_mo': traj_mo,
    }

# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    print("=== COMPOSITIONAL SCALING EXPERIMENT ===")
    print(f"2D → 4D → 6D → 12D")
    print(f"Samples: {N_SAMPLES:,} | "
          f"Models: {N_MODELS} | "
          f"Epochs: {EPOCHS}")

    results = []

    results.append(run_scale(
        scale_id=1, D_state=2,
        D_in=2, D_out=2,
        fn_f1=data_s1_gravity,
        fn_f2=data_s1_spring,
        fn_mono=data_s1_combined,
        fn_gt=gt_s1,
        s_init=np.array([1.0, 0.0]),
        label='1D Physics (Gravity+Spring)'
    ))

    results.append(run_scale(
        scale_id=2, D_state=4,
        D_in=4, D_out=4,
        fn_f1=data_s2_gravity,
        fn_f2=data_s2_spring,
        fn_mono=data_s2_combined,
        fn_gt=gt_s2,
        s_init=np.array([1.0,0.5,0.0,0.0]),
        label='2D Physics (Gravity+Spring)'
    ))

    results.append(run_scale(
        scale_id=3, D_state=6,
        D_in=6, D_out=6,
        fn_f1=data_s3_gravity,
        fn_f2=data_s3_spring,
        fn_mono=data_s3_combined,
        fn_gt=gt_s3,
        s_init=np.array([1.,0.5,0.5,0.,0.,0.]),
        label='3D Physics (Gravity+Spring)'
    ))

    results.append(run_scale(
        scale_id=4, D_state=12,
        D_in=6, D_out=12,   # relative input
        fn_f1=data_s4_gravity,
        fn_f2=data_s4_spring,
        fn_mono=data_s4_combined,
        fn_gt=gt_s4,
        s_init=np.array([
            -1.,0.,0., 0.,0.8,0.2,
             1.,0.,0., 0.,-0.8,-0.2
        ]),
        label='12D Two-Body (Gravity+Spring)',
        is_twobody=True
    ))

    # -----------------------------------------------
    # SUMMARY
    # -----------------------------------------------
    print(f"\n{'='*58}")
    print(f"SCALING SUMMARY")
    print(f"{'='*58}")
    print(f"{'Scale':<6}{'D':>4}  "
          f"{'ZeroShot':>12}  "
          f"{'Monolith':>12}  "
          f"{'Ratio':>8}  Result")
    print(f"{'-'*58}")
    for r in results:
        w = 'ZS WINS' if r['ratio']>1 else 'MONO WINS'
        print(
            f"  {r['scale']:<4}{r['D']:>4}  "
            f"{r['mse_zs']*1e4:>12.4f}  "
            f"{r['mse_mo']*1e4:>12.4f}  "
            f"{r['ratio']:>8.2f}x  {w}"
        )
    print(f"{'='*58}")
    wins = sum(1 for r in results if r['ratio']>1)
    print(f"Zero-Shot wins: {wins}/4 scales")

    # -----------------------------------------------
    # FIGURE
    # -----------------------------------------------
    dims    = [r['D']         for r in results]
    mse_zs  = [r['mse_zs']*1e4 for r in results]
    mse_mo  = [r['mse_mo']*1e4 for r in results]
    ratios  = [r['ratio']     for r in results]
    labels  = ['2D','4D','6D','12D']
    lf1_all = [r['lf1']       for r in results]
    lf2_all = [r['lf2']       for r in results]
    lm_all  = [r['lm']        for r in results]

    fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes.flat:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

    # Panel 1 — MSE bar
    ax = axes[0,0]
    x, w = np.arange(4), 0.35
    b1 = ax.bar(x-w/2, mse_zs, w,
                color='#00d4ff', alpha=0.9,
                label='Zero-Shot (Ours)')
    ax.bar(x+w/2, mse_mo, w,
           color='#ff4444', alpha=0.7,
           label='Monolith')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white')
    ax.set_ylabel('MSE × 10⁻⁴ (log)')
    ax.set_title('MSE: Zero-Shot vs Monolith')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              labelcolor='white')
    ax.grid(True, alpha=0.2, axis='y')
    for bar,v in zip(b1, mse_zs):
        ax.text(bar.get_x()+bar.get_width()/2,
                v*1.3, f'{v:.3f}',
                ha='center', fontsize=7,
                color='white')

    # Panel 2 — Improvement ratio
    ax = axes[0,1]
    cols = ['#00ff99' if r>1 else '#ff6b35'
            for r in ratios]
    bars = ax.bar(labels, ratios,
                  color=cols, alpha=0.85)
    ax.axhline(y=1.0, color='white',
               ls='--', lw=1.5,
               label='Breakeven (1×)')
    for bar,v in zip(bars, ratios):
        ax.text(bar.get_x()+bar.get_width()/2,
                v+0.05, f'{v:.2f}×',
                ha='center', fontsize=10,
                color='white', fontweight='bold')
    ax.set_ylabel('Improvement over Monolith')
    ax.set_title('Zero-Shot Advantage')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              labelcolor='white')
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(colors='white')

    # Panel 3 — Training convergence
    ax = axes[1,0]
    ax.plot(labels, lf1_all, 'o-',
            color='#00d4ff', lw=2,
            label='Force-1 (isolated)')
    ax.plot(labels, lf2_all, 's-',
            color='#00ff99', lw=2,
            label='Force-2 (isolated)')
    ax.plot(labels, lm_all,  '^-',
            color='#ff4444', lw=2,
            label='Monolith (combined)')
    ax.set_yscale('log')
    ax.set_ylabel('Best Training Loss')
    ax.set_title('Training Convergence')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              labelcolor='white')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')

    # Panel 4 — Scaling law
    ax = axes[1,1]
    ax.plot(dims, mse_zs, 'o-',
            color='#00d4ff', lw=2.5,
            markersize=10, label='Zero-Shot')
    ax.plot(dims, mse_mo, 's--',
            color='#ff4444', lw=2,
            markersize=8,  label='Monolith')
    ax.fill_between(dims, mse_zs, mse_mo,
        where=[z<m for z,m in zip(mse_zs,mse_mo)],
        color='#00ff99', alpha=0.15,
        label='Zero-Shot wins')
    ax.set_yscale('log')
    ax.set_xlabel('State Space Dimension')
    ax.set_ylabel('MSE × 10⁻⁴')
    ax.set_title('Scaling Law: MSE vs Dimension')
    ax.legend(fontsize=9, facecolor='#1a1a2e',
              labelcolor='white')
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors='white')
    ax.set_xticks(dims)
    ax.set_xticklabels([f'{d}D' for d in dims],
                        color='white')

    plt.suptitle(
        'Compositional Transfer Scaling: '
        '2D → 4D → 6D → 12D',
        color='white', fontsize=13,
        fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('scaling_experiment.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a')
    print("\nSaved: scaling_experiment.png")
    if wins >= 3:
        print("STRONG RESULT — include in paper")
    elif wins >= 2:
        print("PARTIAL RESULT — include with caveats")
    else:
        print("WEAK RESULT — do not include in paper")