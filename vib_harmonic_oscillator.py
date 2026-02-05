import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ─── Physics Engine (Energy-Preserving Oscillator) ───
class Oscillator:
    def __init__(self, steps=5000):
        self.dt = 0.05
        self.steps = steps
        self.s = np.zeros((steps, 4))
        self.s[0] = [1.2, 0.0, 0.0, 0.8]

    def run(self):
        for i in range(1, self.steps):
            self.s[i, 2] = self.s[i-1, 2] - self.dt * self.s[i-1, 0]
            self.s[i, 3] = self.s[i-1, 3] - self.dt * self.s[i-1, 1]
            self.s[i, 0] = self.s[i-1, 0] + self.dt * self.s[i, 2]
            self.s[i, 1] = self.s[i-1, 1] + self.dt * self.s[i, 3]
        return self.s

# ─── Variational Information Bottleneck Model ───
class PhysicsVIB(nn.Module):
    def __init__(self, beta=1e-3):
        super().__init__()
        self.beta = beta
        self.enc = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.mu = nn.Linear(64, 2)
        self.logvar = nn.Linear(64, 2)
        self.dec = nn.Sequential(nn.Linear(2, 64), nn.GELU(),
                                 nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        kl = -0.5 * torch.sum(1 + lv - mu**2 - lv.exp(), dim=1).mean()
        rec = torch.mean((x - self.dec(z))**2)
        return self.dec(z), rec + self.beta * kl, mu, rec, kl

# ─── Unified Real-Time Visual Tracker ───
def unified_live_tracker(beta=1e-3, steps=1000, update_interval=10):
    data = torch.tensor(Oscillator().run(), dtype=torch.float32)
    L_true = data[:,0]*data[:,3] - data[:,1]*data[:,2]  # Angular momentum
    ang_momentum_error = lambda x: torch.mean((x[:,0]*x[:,3]-x[:,1]*x[:,2]-L_true)**2)

    model = PhysicsVIB(beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Interactive plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8), facecolor='#121212')
    fig.canvas.manager.set_window_title("Unified Physics-VIB Tracker")

    # Trajectory
    traj_actual, = ax.plot([], [], color='cyan', alpha=0.3, label='True Trajectory')
    traj_pred, = ax.plot([], [], color='magenta', lw=1.5, label='Predicted Trajectory')

    # Latent scatter
    latent_scatter = ax.scatter([], [], c=[], cmap='plasma', s=5)

    # Summary metrics overlay
    summary_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                           color='white', fontsize=10, verticalalignment='top',
                           bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))

    ax.set_facecolor('#1e1e1e')
    ax.set_title("Trajectory + Latent Space + Metrics", color='white')
    ax.legend()

    # Axis limits
    ax.set_xlim(data[:,0].min()*1.1, data[:,0].max()*1.1)
    ax.set_ylim(data[:,1].min()*1.1, data[:,1].max()*1.1)

    for step in range(steps):
        # Training step
        _, loss, latent, rec_val, kl_val = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update visuals periodically
        if step % update_interval == 0 or step == steps - 1:
            with torch.no_grad():
                pred, _, latent, rec_val, kl_val = model(data)

            # Update trajectory
            traj_actual.set_data(data[:,0], data[:,1])
            traj_pred.set_data(pred[:,0], pred[:,1])

            # Update latent scatter (project onto trajectory plane)
            latent_scatter.set_offsets(latent.numpy())
            latent_scatter.set_array(L_true)

            # Update metrics summary
            am_error = ang_momentum_error(pred).item()
            summary = f"Step: {step}\nMSE: {rec_val.item():.5f}\nKL: {kl_val.item():.5f}\nConservation Error: {am_error:.5f}"
            summary_text.set_text(summary)

            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.ioff()
    plt.show()

# ─── Run Tracker ───
unified_live_tracker(beta=1e-3, steps=1000, update_interval=5)
