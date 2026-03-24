# Large deviations and conditioned monitored quantum systems: a tensor network approach

This repository contains the code used to produce the results and figures of the paper:

**"Large deviations and conditioned monitored quantum systems: a tensor network approach"**

---

## 📦 Overview

This repository provides a tensor network framework to study large deviation properties in monitored quantum many-body systems.

It implements the numerical methods introduced in the paper, enabling the study of dynamical phase transitions at the level of quantum trajectories. In particular, the code allows:

* Simulation of monitored quantum dynamics using tensor networks
* Computation of large deviation quantities (e.g. SCGF, activity)
* Sampling of quantum trajectories
* Access to conditioned quantum many-body states

The repository includes both the core tensor network infrastructure (MPS, MPO, MPDO) and the scripts required to reproduce all results presented in the paper.

---

## 📁 Structure

* `src/` – Core tensor network code and simulation routines
* `scripts/` – Scripts to generate figures
* `figures/` – Output figures (PDF)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mariacea/monitored-quantum-systems
cd monitored-quantum-systems
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running simulations

The main datasets used in the paper are generated using:

```bash
python scripts/main.py
python scripts/compute_activity.py
python scripts/2d_sampling.py
```

---

## 📊 Reproducing figures

Figures from the paper can be reproduced using the plotting scripts in `scripts/`:

* Fig. 1 → `plot_phase_diagram.py`
* Fig. 3 → `plot_SCGF.py` `plot_activity.py` `plot_rate_function.py` `plot_critical_s.py`
* Fig. 4 → `plot_ensembles.py`
* Fig. S2 → `plot_convergence.py`
* Fig. S3 → `plot_critical_s.py`

Example:

```bash
python scripts/plot_SCGF.py
```

---

## 📌 Notes

* Simulations can be computationally demanding depending on system size and bond dimension
* Some scripts rely on stochastic sampling; results may vary slightly between runs
* Precomputed data used in the figures of the paper are provided in `figures_data/` for convenience and reproducibility
* All datasets can be regenerated from scratch using the scripts in `scr/`, ensuring full reproducibility of the results

---

## 📄 License

MIT License 

---

## 📚 Citation

If you use this code, please cite:

```
@article{paper,
  title={Large deviations and conditioned monitored quantum systems: a tensor network approach},
  author={Your Name},
  year={2026}
}
```

---

## 🔗 Zenodo archive

A frozen version of this repository is available on Zenodo:
(ADD LINK AFTER UPLOAD)

---

🔗 arXiv

Preprint available at:
(ADD LINK AFTER SUBMISSION)
