# Potentially Hazardous Asteroid Detection using Deep Learning

### Deep Learning based classification of Potentially Hazardous Asteroids (PHA) using orbital characteristics and NASA asteroid datasets.

</div>

---

# Project Overview

Near-Earth asteroids constantly travel across space, and some of them may come dangerously close to Earth. NASA classifies such objects as:

## Potentially Hazardous Asteroids (PHA)

These are asteroids whose:
- Orbit passes close to Earth
- Estimated size is large enough to cause potential damage

This project builds a **Deep Learning classification system** capable of predicting whether an asteroid is hazardous or non-hazardous using:

 Orbital parameters  
 Physical properties  
 Measurement uncertainties  

The complete pipeline includes:
- Data cleaning
- Handling incorrect datatypes
- Missing value treatment
- Feature scaling
- Class imbalance handling
- Deep Neural Network training
- Performance evaluation

---

# Objective

The primary objective is:

> Predict whether an asteroid is **Potentially Hazardous (PHA)**.

### Target Variable

| Label | Meaning |
|---|---|
| 0 | Non-Hazardous |
| 1 | Hazardous |

---

# Dataset Information

The dataset contains asteroid records from NASA/JPL databases including:

- Orbital dynamics
- Distance measurements
- Orbital uncertainty metrics
- Physical asteroid properties

---

# Important Features

## Orbital Parameters

| Feature | Description |
|---|---|
| `a` | Semi-major axis |
| `e` | Orbital eccentricity |
| `i` | Orbital inclination |
| `q` | Perihelion distance |
| `ad` | Aphelion distance |
| `moid` | Minimum Orbit Intersection Distance |
| `per` | Orbital period |
| `n` | Mean motion |

---

## Hazard-Critical Features

### `moid`
Minimum distance between asteroid orbit and Earth orbit.

Smaller MOID → Higher collision risk.

### `H`
Absolute magnitude (brightness).

Used as a proxy for asteroid size.

Larger asteroids are potentially more dangerous.

---

## Measurement Uncertainty Features

Columns beginning with:

```python
sigma_
```

represent uncertainty in orbital measurements.

Examples:
- `sigma_e`
- `sigma_a`
- `sigma_per`

These help capture orbital reliability and prediction confidence.

---

# Data Preprocessing Pipeline

The dataset required extensive preprocessing before training.

---

## Step 1 — Removing Non-Predictive Columns

```python
df.drop(columns=['name'])
```

Asteroid names do not provide meaningful numerical information for prediction.

---

## Step 2 — Fixing Incorrect Data Types

Many important numerical columns were incorrectly stored as:

```python
dtype = object
```

instead of:

```python
float64
```

This usually happens because of:
- Hidden spaces
- Invalid symbols
- Missing value markers

### Solution

```python
pd.to_numeric(errors='coerce')
```

This:
- Converts valid values into numbers
- Converts invalid values into `NaN`

which can later be safely imputed.

---

## Step 3 — Handling Missing Values

Median imputation was used:

```python
x_train.fillna(x_train.median())
```

### Why Median?

Because astronomical datasets often contain extreme outliers.

Median is:
- More stable
- More robust
- Less affected by extreme values

---

## Step 4 — Feature Scaling

Standardization was applied using:

```python
StandardScaler()
```

This transforms features to:
- Mean = 0
- Standard deviation = 1

Benefits:
- Faster convergence
- Stable gradients
- Better neural network optimization

---

# Handling Extreme Class Imbalance

The dataset is **heavily imbalanced**.

| Class | Distribution |
|---|---|
| Non-Hazardous | Extremely High |
| Hazardous | Very Rare |

Without correction, the model would simply predict:
> “Everything is safe”

and still achieve high accuracy.

---

## Solution 1 — Weighted Random Sampling

Used:

```python
WeightedRandomSampler
```

This increases the probability of minority class samples during training.

---

## Solution 2 — Focal Loss

Traditional Binary Cross Entropy struggles on imbalanced datasets.

Therefore, this project uses:

# Focal Loss

Formula:

```text
FL(pt) = α(1 − pt)^γ * BCE
```

### Benefits
 Focuses on difficult samples  
 Reduces majority-class dominance  
 Improves hazardous asteroid detection  

Parameters used:

```python
alpha = 3
gamma = 2
```

---

# 🧠 Deep Learning Model Architecture

The model is a fully connected feedforward neural network built using PyTorch.

```text
Input Features
       ↓
Linear(256)
       ↓
ReLU
       ↓
BatchNorm
       ↓
Dropout(0.3)

       ↓

Linear(128)
       ↓
ReLU
       ↓
BatchNorm
       ↓
Dropout(0.3)

       ↓

Linear(64)
       ↓
ReLU

       ↓

Linear(32)
       ↓
ReLU

       ↓

Output Layer(1)
```

---

# Why These Components?

## ReLU Activation
Introduces non-linearity allowing the model to learn complex orbital relationships.

---

## Batch Normalization
Helps:
- Stabilize learning
- Accelerate convergence
- Reduce internal covariate shift

---

## Dropout
Prevents overfitting by randomly disabling neurons during training.

---

# Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Batch Size | 512 |
| Loss Function | Focal Loss |

---

# Model Performance

## Classification Report

```text
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00    280971
         1.0       0.77      0.98      0.86       610
```

---

# Confusion Matrix

```text
[[280792    179]
 [    10    600]]
```

---

# ROC-AUC Score

```text
0.99991
```

---

# Performance Analysis

The model achieved:

Extremely high ROC-AUC  
Excellent hazardous asteroid recall  
Strong minority class learning  
Very low false negatives  

This is particularly important because:
> Missing a hazardous asteroid is far more dangerous than generating false alarms.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Scikit-learn | Preprocessing & Evaluation |
| PyTorch | Deep Learning |
| Matplotlib | Visualization |
| Seaborn | Visualization |

---

# Future Improvements

- SHAP / Explainable AI
- Hyperparameter tuning
- Transformer-based tabular models
- Real-time prediction API
- Flask/FastAPI deployment
- Model interpretability dashboard

---

# Acknowledgements

- NASA JPL Small Body Database
- PyTorch Documentation
- Scikit-learn Documentation

---

<div align="center">

# 🚀 Exploring Space with Deep Learning

</div>
