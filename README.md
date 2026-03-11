<div align="center">

  <img src="LOGO.png" width="100%">

</div>

[![GitHub Stars](https://img.shields.io/github/stars/Psygoal/CTClustering?style=social)](https://github.com/Psygoal/CTClustering/stargazers)
[![MIT License](https://img.shields.io/github/license/Psygoal/CTClustering?color=blue)](https://github.com/Psygoal/CTClustering/blob/main/LICENSE)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/Psygoal/CTClustering?color=green)](https://github.com/Psygoal/CTClustering)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/Psygoal/CTClustering?color=orange)](https://github.com/Psygoal/CTClustering/commits/main)
# Unveiling Hidden Intermediate States in Protein Folding with AI-Based Conditional Transition Clustering              
The work has been published on [PNAS](https://www.pnas.org/doi/10.1073/pnas.2531221123). Please cite our 
paper:  
```
X. Liu, W. Cai, H. Fu, & X. Shao, Unveiling hidden intermediate states in protein folding with AI-based conditional transition clustering, Proc. Natl. Acad. Sci. U.S.A. 123 (10) e2531221123, https://doi.org/10.1073/pnas.2531221123 (2026).              
```
## Conditional Transition Clustering (CTC)

Conditional Transition Clustering (CTC) is a machine learning framework for clustering time-series data points based on their transition dynamics, utilizing the [Real NVP (Real-valued Non-Volume Preserving)](https://arxiv.org/abs/1605.08803) neural network model implemented in TensorFlow. This repository provides an implementation of CTC, with a focus on systems exhibiting dynamic transitions, such as the double-well potential system.
CTC is used for time point clustering based on the  neural network model. 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example: Double-Well Potential System](#example-double-well-potential-system)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
CTC uses the Real NVP model to learn conditional transition probabilities in time-series data, enabling clustering of time points based on their dynamic behavior. This is particularly useful for analyzing systems with non-linear dynamics, such as those in physical, chemical, or biological contexts. The repository includes a test example for the double-well potential system.

## Features
- **Flow model-based Clustering**: Utilizes TensorFlow-based Real NVP for modeling transition dynamics.
- **Time-Series Clustering**: Clusters time points based on conditional transition probabilities.
- **Double-Well Potential Example**: Includes a Jupyter notebook ([`ctc_example.ipynb`](/codes/ctc_example.ipynb)) to demonstrate CTC on a double-well potential system.
- **Config-Driven Workflow**: Supports configuration via [`config.json`](/codes/config.json) for flexible model training and inference.

## Installation
To set up the CTC project locally, follow these steps:

1. **Create a Virtual Environment** (recommended):
   ```bash
   conda create -n ctc python=3.8.11
   conda activate ctc
   ```

2. **PyPI Install**:
   ```bash
   pip install ctclustering
   ```

## Usage
CTC is designed to work with time-series data and is configured via a JSON file ([`config.json`](/codes/config.json)). The primary entry point for running CTC is the `run_ctc_from_config` function, which handles model initialization, training, and clustering.

### Basic Workflow
1. **Prepare Data**: Format your time-series data as NumPy arrays.
2. **Configure Settings**: Update [`config.json`](/codes/config.json) with your dataset paths and model parameters.
3. **Run CTC**: Use the `run_ctc_from_config` function to train the model and cluster data.
4. **Visualize Results**: Generate plots to analyze clustering outcomes.

Use the provided Jupyter notebook ([`ctc_example.ipynb`](/codes/ctc_example.ipynb)) for an interactive workflow.

## Example: Double-Well Potential System
The repository includes a Jupyter notebook ([`ctc_example.ipynb`](/codes/ctc_example.ipynb)) demonstrating CTC on a double-well potential system, a classic model with two stable states separated by an energy barrier.

### Running the Example
1. Ensure dependencies are installed.
2. Open [`ctc_example.ipynb`](/codes/ctc_example.ipynb) in Jupyter Notebook or JupyterLab.
3. Run the notebook cells to:
   - Train a CTC model from scratch using `run_ctc_from_config`.
   - Visualize clustering results with Matplotlib.
   - Load a trained model for prediction on new data.

Key steps in the notebook:
- **Training**: Uses `run_ctc_from_config` with [`config.json`](/codes/config.json) to train Real NVP models for marginal (`px`) and joint (`pxy`) probability estimation.
- **Visualization**: Plots clustered time points with color-coded labels.
- **Prediction**: Loads a trained model to predict cluster labels for unseen data.

Example output:
- Training logs show loss and validation loss for `px_flow.h5` and `pxy_flow.h5` models.
- A scatter plot visualizes clustered data points.
- Prediction on new data returns cluster labels (e.g., `[1 0 0 0 1 0 0 1 1 1]`).

## Project Structure
```
CTClustering/
├── codes/                  # Main source code and examples directory
│   ├── CTClustering/       # Python package for CTC implementation
│   │   ├── __init__.py     # Package initializer
│   │   ├── calcp.py        # Script for calculating transition probabilities
│   │   ├── clustering.py   # Clustering algorithms (e.g., CTC class)
│   │   ├── ctc.py          # Main CTC execution script (run_ctc_from_config)
│   │   ├── estimating.py   # Estimation-related functions
│   │   ├── flow.py         # Script for building the flow model
│   │   ├── jointdata.py    # Script for generating joint data
│   │   └── merging.py      # Segment merging logic
│   ├── data.npy            # Trajectory of the double-well potential system
│   ├── ctc_example.ipynb   # Jupyter notebook example for double-well potential
│   └── config.json         # Configuration file for CTC
├── LICENSE                 # MIT License
├── LOGO.png                # Project logo
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Dependencies
The project requires the following Python packages (Note: If installed via PyPI, these dependencies are handled automatically without the need for manual installation):
- Python 3.8+
- tensorflow>=2.11.0, <2.16.0
- tensorflow-probability>=0.17.0, <0.24.0
- numpy>=1.22.4, <2.0.0
- scipy>=1.10.0
- tqdm>=4.60.0
- igraph>=0.11.8, <0.12.0

A full list of dependencies is provided in `requirements.txt`.

## Configuration
CTC uses a [`config.json`](/codes/config.json) file to specify model and clustering parameters. Key sections include:
- **init_params**: Defines data dimensions (`dim`), lag time, and file paths for marginal (`Px_path`), models (`Px_model_path`, `Pxy_model_path`), and transition matrix (`transition_mat_path`).
- **px_estimator_params**: Configures the marginal probability estimator (e.g., `output_dim=128`, `num_coupling_layers=12`, `epochs=200`).
- **pxy_estimator_params**: Configures the joint probability estimator (e.g., `output_dim=256`, `num_coupling_layers=12`, `epochs=200`).
- **valley_finding_params**: Sets parameters for peak detection of marginal probability profile (e.g., `window_length=50`, `peak_distance=50`).
- **merging_params**: Controls segment merging (e.g., `min_cluster_size=100`, `tolerance=10`).

Example [`config.json`](/codes/config.json) snippet:
```json
{
  "init_params": {
    "dim": null,
    "lag_time": 1,
    "Px_path": "px.npy",
    "Px_model_path": "px_flow.h5",
    "Pxy_model_path": "pxy_flow.h5",
    "transition_mat_path": "transmat.npy"
  },
  "fit_predict_main_params": {
    "save": true,
    "ctc_model_path": "ctc_model.pkl"
  },
  "px_estimator_params": {
    "output_dim": 128,
    "reg": 1e-4,
    "num_coupling_layers": 12,
    "learning_rate": 3e-4,
    "epochs": 200,
    "batch_size": 2048,
    "validation_split": 0.1
  },
  "pxy_estimator_params": {
    "output_dim": 256,
    "reg": 1e-4,
    "num_coupling_layers": 12,
    "learning_rate": 3e-4,
    "epochs": 200,
    "batch_size": 2048,
    "validation_split": 0.1
  },
  "px_estimate_params": {
    "inference_batch_size": 200000
  },
  "valley_finding_params": {
    "window_length": 50,
    "polyorder": 7,
    "peak_distance": 50
  },
  "cal_transition_mat_params": {
    "inference_batch_size": 1000000
  },
  "merging_params": {
    "ds_scale": 50,
    "bins": 400,
    "sigma": 2,
    "tolerance": 10,
    "min_cluster_size": 100,
    "patience": 15,
    "change_patient_tolerence": 10
  }
}
```

Update `dim` in [`config.json`](/codes/config.json) to match your data’s dimensionality before running.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows the project’s coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or suggestions, please open an issue on GitHub or contact the repository owner at liuxuyang@mail.nankai.edu.cn.
