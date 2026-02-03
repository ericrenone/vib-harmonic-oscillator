# Variational Information Bottleneck on the 2D Harmonic Oscillator

Educational demonstration showing how the variational information bottleneck (VIB) trades reconstruction quality against preservation of conserved quantities (angular momentum *L*, energy *H*) in a simple integrable Hamiltonian system.

![Results overview](vib_harmonic_results.png)

### Core observation

Stronger compression (higher β) systematically violates classical conservation laws — even though the underlying dynamics are perfectly integrable.

### Features

- Leapfrog / velocity Verlet integrator → near-perfect original conservation (drift ≪ 10⁻⁶)
- VIB objective: MSE + β × KL(q(z|x) ‖ 𝒩(0,1))
- Deterministic autoencoder baseline (β = 0, no KL)
- Phase-space portraits + approximate Poincaré sections (y=0 crossings)
- Crude Monte-Carlo estimate of mutual information I(z;x)
- Quantitative metrics: mean / std / range of *L* and *H*

### Related literature

Connects to recent work on learning / enforcing symmetries and conserved quantities in neural models of physics:

- van der Ouderaa et al. (2024). "Noether's razor: Learning Conserved Quantities" — NeurIPS 2024
- Inoue et al. (2021). "Interpretable conservation law estimation…" — Phys. Rev. E
- Symplectic & Hamiltonian neural network literature (various 2022–2025)

### Requirements

```bash
pip install torch numpy matplotlib scipy
