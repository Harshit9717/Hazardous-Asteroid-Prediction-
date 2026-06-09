# Potentially Hazardous Asteroid Classification

A supervised ML classifier that predicts whether a Near-Earth Object (NEO) poses a hazard risk to Earth, using orbital and physical data from NASA/JPL. Built with Scikit-learn and Optuna Bayesian hyperparameter optimization — with a focus on minimizing false negatives on rare high-risk cases.

---

## Problem

NASA tracks hundreds of thousands of asteroids. Manually assessing each for hazard potential is infeasible. An asteroid is classified as Potentially Hazardous when it exceeds 140 meters in diameter and passes within 7.5 million km of Earth's orbit. Missing a true hazardous asteroid (false negative) is far costlier than a false alarm — so this project optimizes for recall on the positive (hazardous) class.

---

## Approach

```
NASA/JPL Orbital + Physical Data
           │
    Exploratory Data Analysis
    (feature distributions, class imbalance check)
           │
    Feature Engineering + Selection
    (drop non-predictive identifiers, date columns)
           │
    Stratified Train/Val/Test Split
           │
    Optuna Bayesian Hyperparameter Search
    (TPE sampler + MedianPruner)
           │
    Final Model Training
           │
    Evaluation: Precision, Recall, F1, ROC-AUC
    (optimized to minimize false negatives)
```

---

## Hyperparameter Optimization

Used **Optuna** with the TPE (Tree-structured Parzen Estimator) sampler for automated Bayesian search over model hyperparameters — a principled alternative to grid search that models the parameter space and focuses trials on promising regions.

Key choices:
- **MedianPruner**: Terminates underperforming trials early based on intermediate results, reducing total compute
- **Stratified cross-validation inside the Optuna objective**: Ensures hyperparameter selection generalizes across folds, not just a single split
- Objective metric: **Recall on hazardous class** — to minimize the cost of missing a true threat

---

## Results

| Metric | Score |
|---|---|
| Precision | [your value] |
| Recall (hazardous class) | [your value] |
| F1-Score | [your value] |
| ROC-AUC | [your value] |

> Recall on the hazardous (positive) class is the primary metric. A missed hazardous asteroid is a worse outcome than a false alarm.

---

## Dataset

**NASA/JPL Near-Earth Object Dataset**  
Source: [Kaggle — NASA Asteroids Classification](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification)

- ~4,687 asteroid records, 40 features
- Key features: absolute magnitude, estimated diameter (km/miles), relative velocity, miss distance, orbital period, eccentricity, inclination
- Target: `Hazardous` (Boolean) — severe class imbalance

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

- **Optimization:** Optuna (TPE sampler, MedianPruner)
- **Validation:** Stratified K-Fold cross-validation
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## Project Structure

```
asteroid-classification/
│
├── asteroid_classification.ipynb   # Full pipeline notebook
├── README.md
└── requirements.txt
```

---

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Install dependencies:
   ```bash
   pip install scikit-learn optuna pandas numpy matplotlib seaborn
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification) and place `nasa.csv` in the working directory
4. Run all cells sequentially

---

## What I Learned

- Bayesian optimization with Optuna converges to better hyperparameters in fewer trials than grid or random search, especially with MedianPruner cutting weak trials early
- Running cross-validation inside the Optuna objective is critical — optimizing on a single split leads to hyperparameters that don't generalize
- For safety-critical classification (hazard detection, fraud, medical), optimizing ROC-AUC alone is insufficient — Recall and PR-AUC on the minority class need to be the primary objectives
