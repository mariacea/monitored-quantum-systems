# Large deviations and conditioned monitored quantum systems: a tensor network approach

This repository contains the code used to produce the results and figures of the paper:

**"Large deviations and conditioned monitored quantum systems: a tensor network approach"**

---

## 📦 Overview

We study monitored quantum systems using tensor network techniques, focusing on large deviation properties and conditioned dynamics.

The repository includes:

* Tensor network simulation code (MPS/MPO/MPDO)
* Sampling methods
* Scripts to reproduce all figures in the paper

---

## 📁 Structure

* `src/` – Core simulation code (tensor network methods)
* `scripts/` – Scripts to generate figures
* `figures/` – Output figures (PDF)
* `notebooks/` – Optional interactive exploration

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

To run all simulations and reproduce the main results:

```bash
python scripts/run_all.py
```

---

## 📊 Reproducing figures

Each figure in the paper can be generated using scripts in `scripts/`:

* Fig. 1 → `plot_phase_diagram.py`
* Fig. 3 → `plot_SCGF.py`
* Fig. 4 → `plot_ensembles.py`

Example:

```bash
python -m scripts.plot_SCGF
```

---

## 🧠 Core components

* `MPSClass.py` – Matrix Product States
* `MPOClass.py` – Matrix Product Operators
* `TensorNetworkSolver.py` – Main solver
* `simulation_runner.py` – Simulation pipeline

---

## 📌 Notes

* Simulations may be computationally intensive
* Some results require long runtimes depending on system size

---

## 📄 License

MIT License (or specify your license)

---

## 📚 Citation

If you use this code, please cite:

```
@article{yourpaper2026,
  title={Large deviations and conditioned monitored quantum systems: a tensor network approach},
  author={Your Name},
  year={2026}
}
```

---

## 🔗 Zenodo archive

A frozen version of this repository is available on Zenodo:
(ADD LINK AFTER UPLOAD)
