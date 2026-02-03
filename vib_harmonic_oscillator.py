"""
Variational Information Bottleneck – 2D Harmonic Oscillator
Educational demo: compression vs. conservation trade-off in Hamiltonian systems

Features:
- Leapfrog (symplectic) integration → near-perfect original L & H conservation
- VIB objective: MSE + β × KL(q(z|x) || N(0,1))
- Deterministic AE baseline
- Phase-space portraits + Poincaré sections (y=0 crossings)
- Crude Monte-Carlo I(z;x) estimate
- Clean, minimal, reproducible

MIT License – educational purpose – February 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# ─── 1. 2D isotropic harmonic oscillator ───────────────────────────────────
class HarmonicOscillator2D:
    """H = ½(x² + y² + px² + py²) = 1,  L = x·py − y·px = 1 (conserved)."""
    def __init__(self, n_steps=4000, dt=0.02):
        self.n = n_steps
        self.dt = dt
        self.states = np.zeros((n_steps, 4))
        self.states[0] = [1.0, 0.0, 0.0, 1.0]  # initial condition: L=1, H=1

    def simulate(self):
        x, y, px, py = self.states[:,0], self.states[:,1], self.states[:,2], self.states[:,3]
        for i in range(1, self.n):
            px_half = px[i-1] - 0.5 * self.dt * x[i-1]
            py_half = py[i-1] - 0.5 * self.dt * y[i-1]
            x[i]  = x[i-1]  + self.dt * px_half
            y[i]  = y[i-1]  + self.dt * py_half
            px[i] = px_half - 0.5 * self.dt * x[i]
            py[i] = py_half - 0.5 * self.dt * y[i]
        return self.states


# ─── 2. VIB / Deterministic Autoencoder ─────────────────────────────────────
class VIB(nn.Module):
    def __init__(self, beta=0.01, deterministic=False):
        super().__init__()
        self.beta = 0.0 if deterministic else beta
        self.deterministic = deterministic

        self.encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU())
        self.mu     = nn.Linear(32, 2)
        self.logvar = nn.Linear(32, 2) if not deterministic else None
        self.decoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 4))

    def reparameterize(self, mu, logvar):
        if self.deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h   = self.encoder(x)
        mu  = self.mu(h)
        if self.deterministic:
            z, kl = mu, torch.tensor(0.0, device=x.device)
        else:
            logvar = self.logvar(h)
            z = self.reparameterize(mu, logvar)
            kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
        xhat = self.decoder(z)
        recon = torch.mean((x - xhat)**2)
        loss  = recon + self.beta * kl
        return xhat, loss, recon, kl, z


# ─── 3. Metrics ─────────────────────────────────────────────────────────────
def angular_momentum(s): return s[:,0]*s[:,3] - s[:,1]*s[:,2]
def energy(s):           return 0.5 * (s**2).sum(axis=1)

def estimate_Izx(z, n_bins=40):
    """Crude binning-based I(z;x) ≈ H(z) - H(z|x) lower bound proxy."""
    z = z.detach().cpu().numpy()
    if z.shape[1] != 2: return np.nan
    hist, *_ = np.histogram2d(z[:,0], z[:,1], bins=n_bins, density=True)
    p = hist.ravel()[hist.ravel() > 0]
    h_z = entropy(p) + np.log(n_bins**2 / len(z))
    h_cond = 0.5 * np.sum(np.log(2 * np.pi * np.e * np.var(z, axis=0)))
    return max(0.0, h_z - h_cond)


# ─── 4. Training & evaluation ──────────────────────────────────────────────
def train(beta=0.01, deterministic=False, label=None,
          epochs=1500, lr=3e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    states_np = HarmonicOscillator2D().simulate()
    x = torch.tensor(states_np, dtype=torch.float32)

    model = VIB(beta=beta, deterministic=deterministic)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    recons, kls = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        xhat, loss, recon, kl, z = model(x)
        loss.backward()
        optimizer.step()
        scheduler.step()
        recons.append(recon.item())
        if not deterministic:
            kls.append(kl.item())

    with torch.no_grad():
        xhat, _, _, _, z = model(x)
        xhat_np = xhat.numpy()
        z_np = z.numpy()

    L_orig = angular_momentum(states_np)
    L_recon = angular_momentum(xhat_np)
    H_orig = energy(states_np)
    H_recon = energy(xhat_np)

    metrics = {
        'L_mean':  L_recon.mean(),
        'L_std':   L_recon.std(),
        'L_range': L_recon.ptp(),
        'H_std':   H_recon.std(),
        'I_zx':    estimate_Izx(z_np),
        'final_recon': recons[-1]
    }

    return (states_np, xhat_np, L_orig, L_recon, H_orig, H_recon,
            recons, kls if not deterministic else [], metrics,
            label or f"β = {beta:.4g}")


# ─── 5. Visualization ───────────────────────────────────────────────────────
def plot_results(results):
    fig = plt.figure(figsize=(15, 13.5))
    gs = fig.add_gridspec(5, len(results), height_ratios=[2.6, 1.0, 1.0, 1.3, 1.0])

    for i, (key, (s, xh, Lo, Lr, Ho, Hr, rec, kl, m, lab)) in enumerate(results.items()):
        # Trajectory
        ax = fig.add_subplot(gs[0,i])
        ax.plot(s[:,0], s[:,1], 'k--', lw=0.9, label='original')
        ax.plot(xh[:,0], xh[:,1], 'C0', lw=1.5)
        ax.set_title(lab, fontsize=11)
        ax.axis('equal')
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9)
            ax.set_ylabel('y')

        # Phase space x–px
        ax = fig.add_subplot(gs[1,i])
        ax.plot(s[:,0], s[:,2], 'k--', lw=0.8)
        ax.plot(xh[:,0], xh[:,2], 'C0', lw=1.3)
        ax.set_title('phase space  x – px')
        ax.grid(alpha=0.25)

        # Poincaré section (y=0, py>0)
        ax = fig.add_subplot(gs[2,i])
        mo = (s[:-1,1] * s[1:,1] <= 0) & (s[1:,3] > 0)
        mr = (xh[:-1,1] * xh[1:,1] <= 0) & (xh[1:,3] > 0)
        ax.scatter(s[1:,0][mo], s[1:,2][mo], s=5, c='k', alpha=0.65)
        ax.scatter(xh[1:,0][mr], xh[1:,2][mr], s=9, c='C0')
        ax.set_title('Poincaré  y=0')
        ax.grid(alpha=0.25)

        # L & H
        ax = fig.add_subplot(gs[3,i])
        ax.plot(Lo, 'k--', lw=1.0)
        ax.plot(Lr, 'C0', lw=1.5)
        ax.set_title(f"std(L) = {m['L_std']:.4f}    I(z;x) ≈ {m['I_zx']:.2f}")
        ax.grid(alpha=0.25)
        axH = ax.twinx()
        axH.plot(Ho, 'k--', lw=1.0, alpha=0.5)
        axH.plot(Hr, 'C3', lw=1.4, alpha=0.85)
        if i == 0:
            ax.set_ylabel('L & H')

        # Losses
        ax = fig.add_subplot(gs[4,i])
        ax.plot(rec, 'C0', label='recon')
        if kl:
            ax.plot(kl, 'C1', alpha=0.75, label='KL')
        ax.set_yscale('log')
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9)
            ax.set_ylabel('loss')

    fig.suptitle("VIB Compression vs. Conservation Laws\n2D Harmonic Oscillator  •  leapfrog integration", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("vib_harmonic_results.png", dpi=180, bbox_inches='tight')
    plt.show()


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("VIB + baseline – 2D harmonic oscillator\n")

    configs = [
        ("det-AE",  0.0,   True),
        ("β=0.001", 0.001, False),
        ("β=0.01",  0.01,  False),
        ("β=0.1",   0.1,   False),
    ]

    results = {}
    for name, beta, det in configs:
        print(f"  Training {name:8} ... ", end="", flush=True)
        results[name] = train(beta=beta, deterministic=det, label=name)
        print("done")

    print("\nSummary:")
    print("config      L_mean   L_std    L_range   H_std    I(z;x)≈")
    print("──────────  ───────  ───────  ────────  ───────  ────────")
    for name, (_, _, _, Lr, _, Hr, _, _, m, lab) in results.items():
        print(f"{lab:10}  {Lr.mean():7.4f}  {m['L_std']:7.4f}  {m['L_range']:8.4f}  "
              f"{m['H_std']:7.4f}  {m['I_zx']:7.2f}")

    plot_results(results)
    print("\nResults saved →  vib_harmonic_results.png")
