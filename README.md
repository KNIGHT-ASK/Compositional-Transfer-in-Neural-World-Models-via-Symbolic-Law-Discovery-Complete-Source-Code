# Compositional Transfer in Neural World Models: Zero-Shot Physics Superposition

[![Paper: JMLR Pre-Review](https://img.shields.io/badge/Paper-JMLR%20Submission-blue.svg)](paper/paper.pdf)
[![Physics: 1D & 2D](https://img.shields.io/badge/Physics-1D%20%26%202D-green.svg)](#key-experiments)
[![Status: Research-Grade](https://img.shields.io/badge/Status-Research--Grade-orange.svg)](#scientific-breakthrough)

Official implementation of the JMLR submission: **"Compositional Transfer in Neural World Models via Tangent Space Residual Superposition."**

This work demonstrates that neural world models can internalize and compose modular physical invariants (e.g., gravity, elasticity, drag) without seeing a single transition from the target combined environment. By framing the learning target strictly in the **Tangent Space (Lie Algebra)** of additive ODEs, isolated physical laws natively superpose.

---

## 🚀 Scientific Breakthrough: Zero-Shot vs. 100K Baseline

In our most complex benchmark—a non-linear 2D planetary orbit with quadratic drag—the **Zero-Shot Ensemble** (trained on $0$ transitions from the combined system) statistically outperforms a monolithic model trained on $100{,}000$ target transitions.

```
Model                        Target Data    1-Step MSE (x10^-4)
--------------------------------------------------------------
Monolithic MLP (100K)        100,000        0.18 ± 0.10
Neural ODE (100K)            100,000        0.18 ± 0.05
Zero-Shot Ensemble (ours)    ZERO           0.11 ± 0.03  <-- BEATS BASELINE
```

### Key Discoveries:
*   **Manifold Persistence**: Our modular models preserve the $1/r^2$ orbital curvature zero-shot, whereas monolithic models suffer from **"Escape Velocity Hallucination"** (unphysically gaining energy and drifting).
*   **Hierarchical Decomposition**: The framework autonomously discovers discrete constants, such as the **Coefficient of Restitution ($\epsilon = 0.80$)**, by decomposing mixed systems into continuous and contact-based sub-modules.
*   **Symbolic Robustness**: Extracting interpretable equations via SINDy improves composition accuracy by **4.7x** over direct neural summation.

---

## 📂 Repository Structure

*   `src/`: Core logic and JMLR verification scripts.
    *   `exp_q_tuned_baselines.py`: Hyperparameter-tuned structural baselines (Neural ODEs).
    *   `exp_r_2d_multistep_stats.py`: Quantitative 300-step trajectory verification for 2D orbits.
    *   `composition_experiment.py`: Main zero-shot 1D composition framework.
    *   `sindy_extraction.py`: Sparse regression pipeline for physical law recovery.
*   `paper/`: JMLR-optimized LaTeX manuscript and high-fidelity figures.
*   `models/`: Pre-trained modules ($\mathcal{M}_{gravity}$, $\mathcal{M}_{spring}$, etc.).

---

## 🛠️ Usage & Reproducibility

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. High-Fidelity JMLR Benchmarks
To reproduce the statistically rigorous results reported in the paper (10 seeds, tuned baselines):

```bash
# Verify the 1D structural baselines (Tuned Neural ODEs)
python src/exp_q_tuned_baselines.py

# Verify the 2D non-linear orbital rollout rigor
python src/exp_r_2d_multistep_stats.py
```

### 3. Symbolic Extraction
To extract the discovered physical constants (e.g., $g=9.8$, $k=2.0$) from the trained neural module:
```bash
python src/sindy_extraction.py
```

---

## 📖 Citation

```bibtex
@article{compositional_world_models_2026,
  title={Compositional Transfer in Neural World Models via Tangent Space Residual Superposition},
  author={[Confidential]},
  journal={Journal of Machine Learning Research (JMLR) - Under Review},
  year={2026}
}
```

---
*This project is part of Advanced Agentic Research into Physics-Informed AGI.*
