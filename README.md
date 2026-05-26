# Potentially Hazardous Asteroid Detection using Deep Learning

## Overview

This project uses Deep Learning to classify whether an asteroid is **Potentially Hazardous (PHA)** based on orbital and physical characteristics obtained from NASA/JPL asteroid datasets.

The project focuses heavily on solving:
- Severe class imbalance
- Data quality issues
- Minority class detection

using:
- PyTorch
- Focal Loss
- Weighted Random Sampling
- Neural Networks

---

# Features

- Complete Deep Learning pipeline
- Extensive preprocessing
- Custom PyTorch Dataset
- Weighted Sampling
- Focal Loss implementation
- BatchNorm + Dropout Regularization
- ROC-AUC evaluation

---

# Dataset Features

The dataset contains:
- Orbital parameters
- Physical asteroid properties
- Measurement uncertainties

Important features include:
- `moid`
- `H`
- `eccentricity`
- `orbital_period`
- `sigma_*` uncertainty columns

Target Variable:
```python
pha
