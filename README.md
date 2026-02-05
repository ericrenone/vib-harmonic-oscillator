# Physics-Informed Variational Information Bottleneck (Physics-VIB)

**Fframework for real-time physics simulation, latent space compression, and trajectory tracking.**

## Core

- A **symplectic energy-preserving oscillator** (physics engine)
- A **variational information bottleneck (VIB)** for latent space compression
- **Real-time visualization** of trajectory, latent space, and conservation metrics

The model learns to compress high-dimensional physics trajectories while preserving key invariants (e.g., angular momentum), with live plotting for interactive analysis.

---

## Features

| Feature                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **Physics Engine**          | Symplectic integrator for energy-preserving harmonic oscillator dynamics.   |
| **VIB Model**               | Neural network with GELU activations, KL divergence, and reconstruction.  |
| **Real-Time Tracking**      | Live matplotlib visualization of trajectory, latent space, and metrics. |
| **Conservation Monitoring** | Tracks angular momentum error during training.                             |


The circle is a direct consequence of conserving angular momentum in an isotropic harmonic potential.
