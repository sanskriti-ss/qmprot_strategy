# Copilot Instructions for QMProt Strategy Repository

## Project Overview
- This repository supports quantum simulation of proteins using the QMProt dataset and VQE (Variational Quantum Eigensolver) algorithms.
- Two main research tracks:
  - **QMProt**: Dataset and code for generating, analyzing, and plotting quantum properties of biomolecules.
  - **Quantum Strategy for the Simulation of Large Proteins**: Resource estimation and modeling for scalable quantum simulation of large biomolecules.

## Key Components & Data Flow
- `framework/` is the main codebase for VQE experiments:
  - `main.py`: Entry point for running VQE jobs.
  - `core/`: Hamiltonian loading (`hamiltonian_loader.py`), VQE base class (`base_vqe.py`), and results management (`results_manager.py`).
  - `algorithms/`: Multiple VQE algorithm variants (vanilla, ADAPT, hardware-efficient, QAOA-inspired).
  - `data/hamiltonians/`: Hamiltonian files (`hamiltonian_<molecule>.txt`)—tab-separated, see README for format.
  - `data/qmprot.json`: Metadata for molecules.
  - `results/`, `plots/`: Output directories for experiment results and visualizations.
- `QMProt/` and `Quantum Strategy for the Simulation of Large Proteins/`: Notebooks and data for dataset analysis and resource modeling.

## Developer Workflows
- **Install dependencies:**
  ```bash
  pip install -r framework/requirements.txt
  ```
- **Run VQE experiments:**
  ```bash
  python framework/main.py --help
  # See CLI options for molecule selection, algorithm, etc.
  ```
- **Add new Hamiltonians:**
  - Place in `framework/data/hamiltonians/` as `hamiltonian_<abbreviation>.txt` (see format in `README.md`).
- **Results:**
  - Results are saved in `framework/results/` (CSV, JSON) and plots in `framework/plots/`.

## Project Conventions
- Hamiltonian files: Tab-separated, columns = [coefficient, Pauli string].
- Molecule metadata: `qmprot.json` (abbreviation, name, n_qubits, n_coefficients, energy).
- Algorithms are modular—add new VQE variants in `framework/algorithms/`.
- Use the provided plotting utilities in `framework/plotting/visualizer.py` for result visualization.

## Integration & Extensibility
- To add new molecules, update both `data/hamiltonians/` and `data/qmprot.json`.
- For new algorithms, subclass `core/base_vqe.py` and register in `main.py`.
- Notebooks in `QMProt/` and `Quantum Strategy for the Simulation of Large Proteins/` are for analysis/modeling, not direct execution from the main framework.

## References
- See `framework/README.md` and `framework/data/hamiltonians/README.md` for detailed file formats and workflow examples.
- For dataset/modeling details, see `QMProt/` and `Quantum Strategy for the Simulation of Large Proteins/README.md`.
