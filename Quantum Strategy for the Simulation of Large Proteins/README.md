
# QMProt: A Dataset and Modeling Toolkit for Quantum Protein Simulations

## Overview

**QMProt** is a curated dataset and modeling suite designed to support research in quantum computing applications for protein simulation. It includes quantum properties of biologically relevant molecules and provides predictive models to estimate quantum resource requirements.

This repository includes:

- `QMProt.json`: High-fidelity quantum data for 45 molecules.
- `model_QMProt.ipynb`: Regression models to estimate qubit and gate requirements.
- `final_metrics_models_comparison_OLS_RANSAC_EN_FULL.ipynb`: Benchmarking of OLS vs. RANSAC models.

---

## Dataset: `QMProt.json`

This file contains molecular and quantum data for:
- 20 essential amino acids
- Terminal groups (e.g., amino, carboxyl)
- Small peptides (e.g., Glucagon)

Each entry contains:
- Molecular name, abbreviation, formula
- Number of qubits required (`n_qubits`)
- Hamiltonian coefficients (`n_coefficients`)
- Ground state energy (in Hartree)

**Example:**
```json
{
  "abbreviation": "his",
  "name": "histidine",
  "n_qubits": 128,
  "n_coefficients": 23831261,
  "energy": -538.52442
}
```

---

## Modeling: `model_QMProt.ipynb`

This notebook:

- Loads and filters the dataset
- Uses features like number of electrons to predict:
  - Number of qubits
  - Hamiltonian size
  - Toffoli gate estimates
- Implements:
  - Linear regression
  - Logarithmic transformation
  - Curve fitting with `curve_fit`

---

## Benchmarking: `final_metrics_models_comparison_OLS_RANSAC_EN_FULL.ipynb`

This notebook compares:

- **Ordinary Least Squares (OLS)** vs. **RANSAC** models
- Model performance for extrapolation (e.g., large proteins like Glucagon)
- Key metrics:
  - R² score
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)

---

## Applications

- Estimating quantum resources for biological molecules
- Designing fragmentation strategies for large proteins
- Preparing hybrid VQE/QPE pipelines
- Benchmarking fault-tolerant scalability

---

## Citation

If you use this dataset or modeling workflow, please cite:

> Sala, Laia Coronas and Atchade-Adelomou, Parfait. *QMProt: A Comprehensive Dataset of Quantum Properties for Proteins.* arXiv:2505.08956, 2025.  
> https://arxiv.org/abs/2505.08956

```bibtex
@article{sala2025comprenhensive,
  title={A Comprehensive Dataset of Quantum Properties for Proteins},
  author={Sala, Laia Coronas and Atchade-Adelomou, Parfait},
  journal={arXiv preprint arXiv:2505.08956},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

- **Parfait Atchade-Adelomou**  
  `parfait@lighthouse-dig.com` 
  Lighthouse Disruptive Innovation Group (LDIG-US) 
  MIT Media Lab – City Science Group  
  

- **Laia Coronas Sala**  
  `laia.coronas@lighthouse-dig.com`  
  LDIG Europe – Barcelona
