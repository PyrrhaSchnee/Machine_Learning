# 🤖 Machine Learning — Classification Pipeline

A complete **end-to-end machine learning pipeline** built from scratch in Python, covering every stage from exploratory data analysis to model evaluation. This project demonstrates proficiency in data preprocessing, feature engineering, visualization, and classification using industry-standard libraries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Highlights](#project-highlights)
- [Pipeline Stages](#pipeline-stages)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This repository implements a **supervised classification pipeline** designed to classify knight types using tabular CSV data. The project walks through the full data-science workflow:

1. **Data Exploration** — histograms, scatter plots, correlation analysis, and heatmaps
2. **Data Preprocessing** — standardization, normalization, and train/validation splitting
3. **Feature Engineering** — multi-collinearity detection via Variance Inflation Factor (VIF)
4. **Model Training** — K-Nearest Neighbours (KNN) and Random Forest classifiers
5. **Model Evaluation** — confusion matrix, precision, recall, F1-score, and accuracy metrics

All scripts are self-contained, well-documented, and designed to be run independently or orchestrated through the included `Makefile`.

---

## Project Highlights

| Aspect | Details |
|---|---|
| **Classification Models** | K-Nearest Neighbours (KNN), Random Forest |
| **F1-Score Target** | ≥ 92% (KNN), ≥ 90% (Decision Tree / Random Forest) |
| **Feature Selection** | Variance Inflation Factor (VIF) with iterative removal |
| **Preprocessing** | Z-score standardization, Min-Max normalization |
| **Visualization** | Histograms, scatter plots, correlation heatmaps, decision tree plots, confusion matrices |
| **Evaluation** | Per-class precision, recall, F1-score, overall accuracy, confusion matrix |

---

## Pipeline Stages

### 1. Exploratory Data Analysis

- **`Histogram.py`** — Generates histograms for feature distribution analysis across classes.
- **`Correlation.py`** — Computes and visualizes pairwise feature correlations.
- **`Heatmap.py`** — Produces a full correlation heatmap to identify relationships between features.
- **`points.py`** — Scatter plot visualization for selected feature pairs, with class-based color coding.

### 2. Data Preprocessing

- **`standardization.py`** — Applies Z-score standardization (`z = (x - μ) / σ`) to center data around 0 with unit variance.
- **`Normalization.py`** — Applies Min-Max normalization (`x_norm = (x - min) / (max - min)`) to scale features to [0, 1].
- **`split.py`** — Splits the training dataset into training and validation subsets for model evaluation.

### 3. Feature Engineering

- **`Feature_Selection.py`** — Performs multi-collinearity detection using Variance Inflation Factor (VIF). Iteratively removes the most collinear feature until all VIF values fall below 5, ensuring a lean and informative feature set.
- **`variances.py`** — Analyzes feature variances to support feature selection decisions.

### 4. Model Training & Prediction

- **`KNN.py`** — Implements a K-Nearest Neighbours classifier with:
  - Hyperparameter tuning across multiple `k` values
  - PCA-based dimensionality reduction
  - Validation via separate validation set or automatic train-test split
  - Outputs predictions to `KNN.txt`

- **`Tree.py`** — Trains a Random Forest classifier (400 estimators) with:
  - Balanced class weighting
  - Visual tree plot export (`Tree.png`)
  - Outputs predictions to `Tree.txt`

### 5. Model Evaluation

- **`Confusion_Matrix.py`** — Builds and visualizes a confusion matrix from prediction and ground-truth files, computing per-class and overall metrics:
  - **Precision** — `TP / (TP + FP)`
  - **Recall** — `TP / (TP + FN)`
  - **F1-Score** — Harmonic mean of precision and recall
  - **Accuracy** — `(TP + TN) / (TP + TN + FP + FN)`

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (KNN, Random Forest, PCA, metrics) |
| **Visualization** | Matplotlib |
| **Scientific Computing** | SciPy |
| **Build Automation** | Makefile, Shell script |

---

## Getting Started

### Prerequisites

- Python 3.10+
- GNU Make

### Installation

```bash
# Clone the repository
git clone https://github.com/PyrrhaSchnee/Machine_Learning.git
cd Machine_Learning

# Set up virtual environment and install dependencies
make prep
source ./.venv/bin/activate
```

This will:
1. Create a Python virtual environment
2. Install all dependencies from `requirements.txt`
3. Download the required dataset CSV files

---

## Usage

Each stage of the pipeline can be run independently via `make` targets:

```bash
# Data Exploration
make histogram        # Generate feature histograms
make correlation      # Compute pairwise correlations
make heatmap          # Generate correlation heatmap
make points           # Scatter plot visualization

# Preprocessing
make standard         # Apply Z-score standardization
make normal           # Apply Min-Max normalization
make split            # Split data into training/validation sets

# Feature Engineering
make variance         # VIF-based feature selection

# Model Training
make tree             # Train Random Forest classifier
# KNN is run directly: ./KNN.py <train_file> <test_file> [validation_file]

# Evaluation
make matrix           # Generate confusion matrix from predictions

# Cleanup
make clean            # Remove generated outputs (plots, predictions)
make fclean           # Full clean (also removes virtual env and data files)
```

---

## Project Structure

```
Machine_Learning/
├── Confusion_Matrix.py     # Confusion matrix & evaluation metrics
├── Correlation.py          # Pairwise correlation analysis
├── Feature_Selection.py    # VIF-based feature selection
├── Heatmap.py              # Correlation heatmap visualization
├── Histogram.py            # Feature distribution histograms
├── KNN.py                  # K-Nearest Neighbours classifier
├── Normalization.py        # Min-Max normalization
├── Tree.py                 # Random Forest classifier
├── points.py               # Scatter plot visualization
├── split.py                # Train/validation data splitting
├── standardization.py      # Z-score standardization
├── variances.py            # Feature variance analysis
├── Makefile                # Build automation
├── prep_python.sh          # Virtual environment setup script
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── .gitignore              # Git ignore rules
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with 🐍 Python · scikit-learn · Matplotlib · Pandas · NumPy</sub>
</p>
