# Surrogate Modeling of a Binary Distillation Column
## Benzene / Toluene System - Peng-Robinson EOS

**Author:** Joshua Jacob Thomas
**Task:** Screening Task 3 - Surrogate Modeling using DWSIM and Machine Learning  
**Date:** April 2026

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [My Thought Process](#2-my-thought-process)
3. [System & Thermodynamic Model](#3-system--thermodynamic-model)
4. [Input & Output Variables](#4-input--output-variables)
5. [Why I Did NOT Include Extra Features](#5-why-i-did-not-include-extra-features)
6. [Dataset Generation - The Honest Story](#6-dataset-generation--the-honest-story)
7. [Machine Learning Models - Why These Four](#7-machine-learning-models--why-these-four)
8. [Physical Consistency Validation](#8-physical-consistency-validation)
9. [Final Results](#9-final-results)
10. [Assumptions Made](#10-assumptions-made)
11. [File Structure](#11-file-structure)
12. [How to Run](#12-how-to-run)
13. [How to Open the DWSIM File](#13-how-to-open-the-dwsim-file)
14. [How to Reproduce Results](#14-how-to-reproduce-results)

---

## 1. Project Overview

This project builds a **surrogate machine learning model** for a Benzene/Toluene binary distillation column. Instead of running a full DWSIM simulation every time we want to evaluate a new set of operating conditions, the surrogate model learns the input-output mapping and can predict column performance **near-instantly**.

The surrogate predicts four outputs:
- **xD** - Distillate purity (benzene mole fraction overhead)
- **xB** - Bottoms composition (benzene mole fraction at bottom)
- **QC** - Condenser duty (kW)
- **QR** - Reboiler duty (kW)

from seven operating condition inputs: feed temperature, pressure, composition, number of stages, feed stage, reflux ratio, and bottoms withdrawal rate.

---

## 2. My Thought Process

I approached this problem in the following order:

1. **Understood the distillation system** - spent significant time in DWSIM learning how a binary distillation column works, what the key operating variables are, and how they physically affect the outputs. I built and solved the DWSIM flowsheet manually, which you can find in `DWSIM_Flowsheet_File.dwxmz`.

2. **Tried to automate DWSIM via Python API** - this is where I hit a major wall (detailed in Section 6 below).

3. **Fell back to a physics-based synthetic data generator** - using standard shortcut distillation methods (Underwood-Gilliland-Fenske) calibrated against my real DWSIM anchor cases.

4. **Designed the ML pipeline** - chose four models that represent a spectrum from simple to complex, trained them, and compared using MAE, RMSE, and R².

5. **Validated physical consistency** - checked that the model's predictions respect known distillation behavior (higher reflux -> higher purity, more stages -> better separation, and so on)

---

## 3. System & Thermodynamic Model

| Property | Value |
|---|---|
| **Binary Mixture** | Benzene - Toluene |
| **Thermodynamic Model** | Peng-Robinson (PR) EOS |
| **Feed Flow Rate** | 100 mol/s (fixed) |
| **Condenser Type** | Total condenser |
| **Feed Phase Assumption** | Saturated liquid (q = 1) |
| **Tray Efficiency** | 100% theoretical stages |
| **Column Type** | Enriching - benzene recovered overhead |

Benzene and Toluene form a **near-ideal mixture**, making Peng-Robinson an appropriate and standard choice. The relative volatility of benzene over toluene (~2.3 at 350K, 1 atm) provides good separability across the operating range.

---

## 4. Input & Output Variables

### Input Features (7 variables)

| Symbol | Description | Range Used | Units |
|---|---|---|---|
| T | Feed temperature | 330 - 380 | K |
| P | Feed pressure | 1.0 - 3.0 | atm |
| Z | Feed benzene mole fraction | 0.2 - 0.8 | - |
| N | Number of theoretical stages | 8 - 20 | - |
| F_Stage | Feed stage location | 2 - N-1 | - |
| R | Reflux ratio | 1.0 - 5.0 | - |
| B | Bottoms molar flow rate | 30 - 70 | mol/s |

All ranges are physically meaningful for an industrial benzene/toluene column operating near atmospheric pressure. The feed flow is fixed at 100 mol/s, so B directly determines D = 100 - B (distillate flow).

### Output Targets (4 variables)

| Symbol | Description | Units |
|---|---|---|
| xD | Distillate benzene purity | mole fraction |
| xB | Bottoms benzene composition | mole fraction |
| QC_kW | Condenser duty | kW |
| QR_kW | Reboiler duty | kW |

---

## 5. Why I Did NOT Include Extra Features

The task suggested optionally including: feed vapor fraction, column pressure, and other derived features.

I chose **not to** include additional derived features for the following reasons:

- **Feed vapor fraction (q):** I fixed q = 1 (saturated liquid feed) throughout the dataset. Since it does not vary, it carries zero information for the model - adding a constant feature would be noise.

- **Column pressure** is already included as input variable P. Adding a derived quantity like "pressure drop per stage" would be directly computable from P and N, making it a redundant linear combination of existing features. This inflates dimensionality without adding new information and risks introducing multicollinearity, which particularly hurts Linear Regression.

- **General philosophy:** With 1408 data points and 7 features, the feature-to-sample ratio is already reasonable. Adding more features without a proportional increase in data risks **overfitting** - the model learns spurious correlations rather than genuine physics. I preferred a clean, interpretable 7-feature space that maps directly to controllable operating conditions.

---

## 6. Dataset Generation - The Honest Story

### What I tried first (DWSIM Python API)

I spent **over 8 hours** attempting to automate DWSIM simulations via its Python/COM API. The problems I encountered:

- The API repeatedly threw `AttributeError` exceptions when trying to access flowsheet objects
- When I did get some output, it returned **identical values regardless of input** - a known issue with how DWSIM's automation server caches state between calls
- The DWSIM `.NET` interop layer behaved inconsistently across different approaches (COM automation, direct DLL loading, PYTHONNET)

Despite trying multiple approaches and referring to DWSIM forums and documentation, I could not get reliable parametric output from the API.

### What I did instead

Rather than generating 1000+ rows by hand (which is obviously not feasible), I (and to be honest, with the help of AI tools, and I had not the sligtest clue about these methods or equations) built `dwsim_automation.py` - a **physics-based synthetic data generator** using standard shortcut distillation methods:

- **Underwood equation** - computes minimum reflux ratio
- **Gilliland correlation (Molokanov form)** - relates actual stages to minimum stages
- **Fenske equation** - computes minimum stages for a given separation
- **Antoine equation + Raoult's law** - computes relative volatility α(T) for benzene/toluene
- **Overall energy balance** - computes QC and QR from vapor flow and latent heats

The script was **calibrated against my real DWSIM anchor case** (T=350K, P=1atm, Z=0.5, N=10, FS=5, R=2, B=50), which gave xD≈0.923, xB≈0.077, QC≈4619 kW, QR≈5342 kW in the actual DWSIM run. Small Gaussian noise (σ=0.0015 on compositions, σ_rel=0.2% on duties) was added to simulate realistic DWSIM numerical precision.

**Sampling strategy:** 1400 samples were generated using **Latin Hypercube Sampling (LHS)** to ensure uniform, space-filling coverage of the 6-dimensional input space (T, P, Z, N, R, B), plus 8 manually chosen anchor cases totaling 1408 rows.

### Important caveat

The synthetic data generator approximates DWSIM output well near the calibration point but **may deviate at extremes** of the operating range, particularly for very high reflux ratios (R > 4.5) or very low feed compositions (Z < 0.25). If the DWSIM API issue is resolved in the future, the entire ML pipeline (`surrogate_model.ipynb`) can be **rerun without any changes** - just replace `Dataset.csv` with the real DWSIM-generated data and execute all cells.

---

## 7. Machine Learning Models - Why These Four

I compared four models that represent a deliberate spectrum from simplest to most flexible:

### Linear Regression - the baseline
Every ML study needs a simple baseline to establish what "no sophistication" gives you. Distillation is inherently nonlinear (Fenske equation is exponential in stages, Underwood is a root equation), so linear regression is expected to underfit. Its inclusion lets us quantify how much the nonlinear models actually help.

### Random Forest - robust nonlinear ensemble
Random Forest is one of the most reliable off-the-shelf models for tabular data. It handles nonlinearity and feature interactions naturally, requires minimal hyperparameter tuning, provides built-in feature importance, and is resistant to outliers. It also has a natural upper bound on overfitting due to averaging across many trees.

### XGBoost - gradient boosted trees
XGBoost is widely considered the best-performing model on structured/tabular data in practice. It builds trees sequentially, each correcting the previous one's errors. Compared to Random Forest, it typically achieves lower bias at the cost of being slightly more sensitive to hyperparameters. It serves as a strong benchmark between Random Forest and ANN.

### ANN (MLP) - the flexible approximator
Neural networks are universal function approximators. A 3-layer MLP (256->128->64 neurons, ReLU activation) with input and output standardization can capture complex, highly nonlinear relationships. It was the best performer here (R²=0.9971).

### Why NOT other models?

- **SVM (Support Vector Machine):** SVMs struggle to scale to datasets of ~1400 points with multi-output regression. The kernel trick that makes SVMs powerful also makes them computationally expensive (O(n²) to O(n³)), and multi-output SVM requires wrapping in `MultiOutputRegressor` which adds further overhead. For this problem size, XGBoost and ANN achieve better results with less tuning effort.

- **KNN (K-Nearest Neighbors):** KNN is a lazy learner - it stores all training data and does a nearest-neighbor search at prediction time. It works poorly in higher-dimensional spaces (7 features here) due to the curse of dimensionality - distances become meaningless and all points appear equidistant. It also does not generalize or extrapolate beyond the training data range.

- **Perceptron:** A single-layer perceptron is equivalent to linear regression for regression tasks. It would perform no better than Linear Regression and adds no value over the baseline we already have.

- **K-Medoids / Clustering methods:** These are unsupervised methods - they group data without using output labels. They are not regression models and cannot directly predict xD, xB, QC, QR from operating conditions.

The four chosen models cover the full range of model complexity, interpretability, and nonlinearity - which is exactly what a fair surrogate model comparison requires.

---

## 8. Physical Consistency Validation

Yes - physical trend validation was explicitly performed in the notebook (Section 8). The best model (ANN) was tested against four known distillation behaviors:

| Trend Tested | Expected Behavior | Model Behavior |
|---|---|---|
| xD vs Reflux Ratio R | Monotonic increase (more reflux -> higher purity) | ✅ Correctly increases |
| xD vs Number of Stages N | Increases, then plateaus (diminishing returns) | ✅ Correctly increases and flattens |
| QC vs Reflux Ratio R | Monotonic increase (more reflux -> more condenser load) | ✅ Correctly increases |
| xD vs Feed Composition Z | Increases (more benzene in feed -> easier to purify) | ✅ Correctly increases |

The model respects all four physical trends, confirming it has not just memorized data but learned the underlying physics of the separation.

---

## 9. Final Results

**Best Model: ANN (MLP)**

| Target | MAE | RMSE | R² |
|---|---|---|---|
| xD | 0.0082 | 0.0106 | 0.9976 |
| xB | 0.0084 | 0.0107 | 0.9974 |
| QC_kW | 74.81 | 92.02 | 0.9983 |
| QR_kW | 79.83 | 103.09 | 0.9981 |
| **Mean** | | | **0.9978** |

Train R² = 0.9978, Test R² = 0.9988 -> **gap of only 0.0010** - excellent generalization, no meaningful overfitting.

**Notable observation:** Random Forest performed surprisingly poorly on xD and xB (R²≈0.32), while performing well on QC and QR (R²≈0.99). This suggests the composition predictions have complex interactions that tree-based averaging struggles to resolve, while the energy targets have smoother, more monotonic relationships with the inputs.

---

## 10. Assumptions Made

1. **Saturated liquid feed (q = 1)** throughout - feed enters as liquid at its bubble point
2. **Total condenser** - all overhead vapor is condensed to liquid
3. **100% tray efficiency** - stages are theoretical (no Murphree efficiency correction)
4. **Constant molar overflow (CMO)** - vapor and liquid flows are constant within each column section
5. **Feed flow rate fixed at 100 mol/s** - B and D are derived from this (D = 100 - B)
6. **Raoult's law for VLE** - valid for this near-ideal benzene/toluene system
7. **Antoine equation for vapor pressures** - coefficients from NIST/Reid-Prausnitz-Poling
8. **Peng-Robinson EOS** - used in DWSIM; approximated via relative volatility in the generator
9. **No heat losses** - adiabatic column assumed
10. **No pressure drop across stages** - uniform pressure throughout the column

---

## 11. File Structure

```
DWSIM SURROGATE MODEL/
│
├── README.md                          ← This file
├── DWSIM_Flowsheet_File.dwxmz         ← DWSIM flowsheet (Benzene/Toluene column)
├── Dataset.csv                        ← Generated dataset (inputs/outputs labeled)
├── Results_Summary.txt                ← Best model metrics + sample predictions
│
├── code/
│   ├── dwsim_automation.py            ← Physics-based synthetic data generator
│   └── surrogate_model.ipynb          ← Main ML notebook (EDA + training + plots)
│
└── plots/                             ← All generated plots (created on notebook run)
    ├── eda_feature_distributions.png
    ├── eda_target_distributions.png
    ├── eda_correlation_heatmap.png
    ├── eda_scatter_key_pairs.png
    ├── eda_boxplots.png
    ├── overfitting_check.png
    ├── predicted_vs_actual.png
    ├── r2_comparison.png
    ├── rmse_comparison.png
    ├── mae_comparison.png
    ├── residuals.png
    ├── feature_importance.png
    └── trend_validation.png
```

---

## 12. How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy jupyter
```

Or inside Jupyter, run this as the first cell:

```python
import sys
!{sys.executable} -m pip install xgboost scikit-learn pandas numpy matplotlib seaborn scipy
```

### Step 1 - Regenerate the dataset (recommended for verification)

To verify the dataset is fully reproducible, run the physics-based generator from inside the `code/` folder:

```bash
cd "DWSIM Surrogate Model"
python code/dwsim_automation.py
# optionally add --plot for diagnostic plots:
python dwsim_automation.py --plot
```

This regenerates `Dataset.csv` in the root folder (~1408 rows). The output is deterministic - running it multiple times produces identical results due to the fixed seed (`SEED = 4`).

### Step 2 - Train the models and reproduce all results

Open the notebook from inside the `code/` folder:

```bash
cd code
jupyter notebook code/surrogate_model.ipynb
```

Then: **Kernel -> Restart & Run All**

The notebook will:
1. Load `Dataset.csv` from the root folder
2. Run full EDA (distributions, correlations, scatter plots, boxplots)
3. Split data 80/20 (train/test)
4. Train all four models
5. Evaluate and compare on MAE, RMSE, R²
6. Generate all 13 plots into the `../plots/` folder
7. Save `../Results_Summary.txt`

### Step 3 - Check results

Open `Results_Summary.txt` in the root folder for a clean text summary of all metrics, the best model, and sample predictions vs actual values.

---

## 13. How to Open the DWSIM File

1. Download and install **DWSIM** (free, open source): https://dwsim.org/
2. Open DWSIM
3. Go to **File -> Open**
4. Navigate to and select `DWSIM_Flowsheet_File.dwxmz` in the root folder
5. Click **Solve** (the green play button in the toolbar) to run the simulation
6. The flowsheet shows: Feed stream -> Distillation_Column -> Top (distillate) + Bottom streams, with E1 (condenser) and E2 (reboiler) energy streams

The DWSIM file is configured with:
- **Property Package:** Peng-Robinson (PR)
- **Components:** Benzene, Toluene
- **Anchor condition:** T=350K, P=101325 Pa, Z=0.5, N=10, Feed Stage=5, R=2, B=50 mol/s

---

## 14. How to Reproduce Results

To reproduce exactly the results in `Results_Summary.txt`:

1. Ensure `Dataset.csv` is in the root folder (or regenerate it via Step 1 above)
2. Open `code/surrogate_model.ipynb`
3. Run **Kernel -> Restart & Run All**
4. All random seeds are fixed (`RANDOM_STATE = 4`) - results will be identical every run

If you want to use real DWSIM data instead of synthetic:
1. Export your DWSIM parametric study results as a CSV
2. Rename columns to match: `T, P, Z, N, F_Stage, R, B, xD, xB, QC_kW, QR_kW, status`
3. Set `status = "done"` for all rows
4. Replace `Dataset.csv` in the root folder with your file
5. Run the notebook - **no other changes needed**