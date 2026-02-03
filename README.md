# Rational Inattention on the 2D Harmonic Oscillator

A demonstration of the trade-off between **Information Compression** and **Physical Conservation**. 
This project explores how "Rational Inattention" (limited Shannon capacity) leads to the violation of Noetherian charges ($L, H$) in a Hamiltonian system.
This work bridges information theory and physics‑informed learning by explicitly measuring how mutual information constraints affect the preservation of symmetries and conservation laws — a connection not previously quantified in the literature.

## ⚡ The Core Idea
* **System:** 2D Isotropic Harmonic Oscillator (Conserves Energy $H$ and Angular Momentum $L$).
* **Mechanism:** Variational Information Bottleneck (VIB).
* **The Insight:** As the "Price of Attention" ($\beta$) increases, the model prioritizes reconstruction speed over physical symmetry.

## 🛠️ Features
- **Symplectic Leapfrog Integrator**: Baseline drift $\ll 10^{-6}$.
- **VIB Objective**: $\mathcal{L} = \text{MSE} + \beta \cdot \text{KL}(q(z|x) || p(z))$.
- **Canonical Metrics**: Quantitative drift in $L$ and $H$ vs. Mutual Information $I(z;x)$.
- **Visuals**: Phase-space portraits and Poincaré sections.

## 📊 Comparison
| Architecture | Complexity | Symmetry | Attention Logic |
| :--- | :--- | :--- | :--- |
| **Transformer** | $O(N^2)$ | Learned | Heuristic (Softmax) |
| **Hamiltonian NN** | $O(N)$ | Fixed | Constant |
| **Rational-Canonical**| **$O(N \log N)$** | **Isolated** | **Economic (RI)** |
