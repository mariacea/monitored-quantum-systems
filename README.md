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
* `figures/` – Output figures (PDF) + figures data

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
python -m src.main
python -m src.compute_activity
python -m src.2d_sampling
```
For advanced usage, the core simulation module can be run directly with custom parameters.

To see all available options:

```bash
python -m src.main --help
python -m src.compute_activity --help
python -m src.2d_sampling --help
```

Example minimal run:

```bash
python -m src.main --num_sites 4 --V 5.6 --s 0.0 --max_bond_dim 5 --max_iterations 10 --base_dir test

---

## 📊 Reproducing figures

Figures from the paper can be reproduced using the plotting scripts in `plots/`:

* Fig. 1 → `plot_phase_diagram.py`
* Fig. 3 → `plot_SCGF.py`, `plot_activity.py`, `plot_rate_function.py`, `plot_critical_s.py`
* Fig. 4 → `plot_ensembles.py`
* Fig. S2 → `plot_convergence.py`
* Fig. S3 → `plot_critical_s.py`

Example:

```bash
python -m plots.run_all
```

---

## 📌 Notes

* Simulations can be computationally demanding depending on system size and bond dimension
* Some scripts rely on stochastic sampling; results may vary slightly between runs
* Precomputed data used in the figures of the paper are provided in `figures/data/` for convenience and reproducibility
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
  author={María Cea, Marcel Cech, Federico Carollo, Igor Lesanovsky, and Mari Carmen Bañuls},
  year={2026}
}
```

---

## 🔗 Zenodo archive

A frozen version of this repository is available on Zenodo:
[![DOI](https://zenodo.org/badge/1187282172.svg)](https://doi.org/10.5281/zenodo.19207104)

---

## 🔗 arXiv

Preprint available at:
(ADD LINK AFTER SUBMISSION)
