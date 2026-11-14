# Identification of Tau Particles Using a DeepTau-Inspired Algorithm

## Overview

This project explores the identification of hadronic tau leptons against quark–gluon (QCD) jets using a multiclass deep neural network inspired by the DeepTau algorithm used in the CMS experiment.

The implementation combines high-level reconstructed tau and event variables with particle-level information represented through inner and outer grids. The network is designed to distinguish four classes:

- **Tau:** hadronic tau candidates
- **QCD jet:** quark–gluon jet background
- **Muon:** muon candidates
- **Electron:** electron candidates

The project is based on CMS-style event data processed using a modified `WZAnalyzer`, with the resulting ROOT files used to construct the neural-network inputs.

## Project Pipeline

The implementation follows four main stages:

1. **High-level feature extraction:** Export reconstructed tau and event-level variables from ROOT files into NumPy arrays.
2. **Particle-grid construction:** Build sparse inner and outer grids using particle-level candidate information.
3. **Multiclass model training:** Train a DeepTau-inspired neural network using high-level variables and, optionally, particle grids.
4. **Evaluation:** Evaluate classification performance using accuracy, confusion matrices, ROC curves, AUC, tau efficiency, and background misidentification rates.

## Repository Structure

```text
Identification-of-Tau-Particles-using-DeepTau-Algorithm/
│
├── README.md
├── build_grids.py
├── export_hl_41_features.py
├── export_hl_47_features.py
├── smoke_test_mc.py
├── train_deeptau_mc.py
│
└── notebooks/
    ├── deeptau_41_features.ipynb
    └── deeptau_47_features.ipynb
```

The notebooks contain the corresponding Kaggle workflows. The Python scripts implement the reusable data-processing and model-training pipeline.

## Implementation Details

### 1. High-Level Feature Extraction

The two feature exporters extract high-level reconstructed variables from the ROOT output produced by the modified `WZAnalyzer`.

- `export_hl_41_features.py`: 41 high-level features.
- `export_hl_47_features.py`: 47 high-level features, including six additional jet-related variables.

The exported data are stored in compressed NumPy `.npz` files, along with feature names, event identifiers, and selection information.

### 2. Particle-Grid Construction

`build_grids.py` constructs sparse particle grids from candidate-level information.

Two grids are implemented:

| Grid | Dimensions | Cell size |
|---|---:|---:|
| Inner grid | 11 × 11 | 0.02 × 0.02 |
| Outer grid | 21 × 21 | 0.05 × 0.05 |

The grids are centred on the tau axis when a tau candidate is found, or on the leading-jet axis otherwise.

Particles are grouped into seven categories:

- Charged hadrons
- Neutral hadrons
- Photons
- Particle-flow electrons
- Particle-flow muons
- Electrons
- Muons

For each grid cell and particle type, the highest-transverse-momentum particle is retained. The sparse representation reduces memory usage during data preparation.

### 3. Multiclass DeepTau-Inspired Neural Network

`train_deeptau_mc.py` implements a four-class neural-network classifier.

Two training configurations are supported:

- **High-level-only mode (`hl`):** Uses the exported high-level variables as input. This provides a baseline configuration.
- **Full mode (`full`):** Combines high-level variables with the inner and outer particle-grid representations.

The full architecture uses separate processing branches for the high-level variables and the particle-grid inputs. The resulting representations are combined and passed through a shared classification head to produce four class probabilities.

The model uses batch normalization, PReLU activations, and dropout.

### 4. Evaluation

The training implementation includes multiclass and tau-versus-background evaluation.

The evaluation metrics include:

- Classification accuracy
- Confusion matrix
- One-vs-rest ROC curves and AUC
- Tau-versus-QCD, tau-versus-muon, and tau-versus-electron ROC curves
- Tau identification efficiency at specified background misidentification rates

## Requirements

The Python implementation uses:

- Python 3
- NumPy
- PyTorch
- Uproot (for ROOT-file reading)
- Additional standard Python libraries used by the scripts

The ROOT input files must be produced with the corresponding modified `WZAnalyzer` branches required by the exporters and grid builder.

## Data and Reproducibility

The ROOT datasets and generated NumPy arrays are not included in this repository.

The data-processing scripts expect ROOT files containing the required high-level and candidate-level branches. The training scripts consume the `.npz` files generated during preprocessing.

The `smoke_test_mc.py` script provides a synthetic-data test of the training pipeline. Its generated data are intended to test code execution and data flow, not to represent physical CMS results.

## Acknowledgements

This work is inspired by the CMS DeepTau algorithm and its approach to tau-lepton identification using high-level reconstructed information and particle-level inputs.

**Reference:** [DeepTau: A Deep Neural Network for the Identification of Tau Leptons](https://arxiv.org/abs/2201.08458)
