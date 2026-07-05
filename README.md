# Credit Risk Prediction Pipeline

A comprehensive, production-ready machine learning pipeline for credit risk classification using a stacked ensemble approach. Combines multiple state-of-the-art algorithms with proper preprocessing, feature engineering, and imbalance handling.

## Project Overview

This project implements a **Stacked Ensemble Credit Risk Classifier** that predicts whether a credit applicant poses a "good" (0) or "bad" (1) credit risk based on multiple datasets:

- **German Credit Dataset**
- **Taiwanese Credit Card Dataset**
- **Australian Credit Approval Dataset

### Key Features

**Unified Data Pipeline**: Loads and standardizes three credit datasets into a common schema

**Advanced Preprocessing**: Handles missing values, categorical encoding, and feature scaling via sklearn Pipeline

**Feature Engineering**: 
- Correlation-based filtering (removes |r| > 0.85)
- Variance thresholding
- Mutual information & chi-square ranking

**Imbalance Handling**: SMOTE + class weights for dealing with imbalanced datasets

**Stacked Ensemble Architecture**:
- **Base Learners**: RandomForest, GradientBoosting, XGBoost, KNeighbors, Neural Network (ANN)
- **Meta-Learner**: XGBoost or LogisticRegression
- **Out-of-fold generation** to avoid data leakage

**Comprehensive Evaluation**: Accuracy, AUC-ROC, Precision, Recall, F1, Confusion Matrix, ROC curves, Feature Importance

**Modular Code Structure**: Separated concerns across data_loader, preprocessing, feature_selection, models, evaluate modules

**Next-Step Hooks**: Stubs for SHAP explainability, cost-sensitive optimization, and FastAPI deployment

---

## Project Structure

```
credit-risk-pipeline/
├── config.py                 # Centralized config (hyperparameters, paths, schemas)
├── data_loader.py            # Load & unify datasets
├── preprocessing.py          # Handle missing values, encoding, scaling
├── feature_selection.py      # Correlation, variance, MI filtering
├── imbalance.py              # SMOTE & class weight handling
├── models.py                 # Base learners + stacking ensemble
├── evaluate.py               # Metrics, ROC curves, feature importance
├── next_steps.py             # Stubs for SHAP, cost matrix, FastAPI
├── main.py                   # Orchestration script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── raw/                  # Raw datasets (to be downloaded)
│   └── processed/            # Processed data (generated)
├── models/                   # Trained models & artifacts (generated)
└── logs/                     # Pipeline logs (generated)
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shruti222005/CredWise.git
   cd CredWise
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

Download the three datasets and place them in `data/raw/`:

- [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) → `german_credit.csv`
- [Taiwanese Credit](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) → `taiwanese_credit.csv`
- [Australian Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)) → `australian_credit.csv`

---

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will:
1. Load and unify datasets
2. Preprocess features
3. Perform feature selection
4. Apply SMOTE for imbalance handling
5. Train base learners and stacking ensemble
6. Evaluate on test set
7. Generate visualizations (ROC curves, confusion matrix, feature importance)
8. Save models and artifacts

### Output

After running, check:
- **Logs**: `credit_risk_pipeline.log`
- **Models**: `models/` directory with:
  - `stacking_ensemble_*.pkl`
  - `RandomForest_*.pkl`, `GradientBoosting_*.pkl`, etc.
  - `preprocessor_*.pkl`
  - `feature_metadata_*.pkl`
- **Visualizations**: 
  - `roc_curves_*.png`
  - `confusion_matrix_*.png`
  - `feature_importance_*.png`
  - `threshold_optimization_*.png`

---

## Configuration

Edit `config.py` to customize:

### Model Hyperparameters
```python
BASE_LEARNERS_CONFIG = {
    "RandomForest": {"n_estimators": 100, "max_depth": 15, ...},
    "GradientBoosting": {...},
    "XGBoost": {...},
    ...
}
```

### Preprocessing
```python
MISSING_VALUE_STRATEGY = {"numeric": "median", "categorical": "most_frequent"}
SCALING_METHOD = "standard"  # or 'minmax', 'robust'
```

### Feature Selection
```python
CORRELATION_THRESHOLD = 0.85
VARIANCE_THRESHOLD = 0.01
TOP_N_FEATURES = None  # Keep all; or specify a number
```

### Imbalance Handling
```python
IMBALANCE_STRATEGY = "smote"  # or 'class_weight', 'both'
SMOTE_SAMPLING_STRATEGY = 0.8
```

---

## Pipeline Stages

### 1. Data Loading & Unification
- Loads 3 datasets with different schemas
- Standardizes column names and target values
- Merges into a single training set
- Logs class distribution

### 2. Preprocessing
- **Imputation**: Median for numeric, mode for categorical
- **Encoding**: OneHotEncoder for categorical features
- **Scaling**: StandardScaler for numerical features
- **Train/Val/Test Split**: 70/10/20 with stratification
- Uses sklearn Pipeline for reproducibility

### 3. Feature Selection
- **Correlation Filtering**: Removes pairs with |r| > 0.85, keeping the one more correlated with target
- **Variance Filtering**: Removes low-variance features
- **Ranking**: By mutual information or chi-square; optionally keep top-N

### 4. Imbalance Handling
- **SMOTE**: Oversamples minority class (applied only to training folds during CV)
- **Class Weights**: Alternatively, use `class_weight='balanced'` in tree models
- Avoids test set leakage

### 5. Model Training
- **Base Learners**:
  - RandomForestClassifier
  - GradientBoostingClassifier
  - XGBClassifier
  - KNeighborsClassifier
  - Keras ANN (3 hidden layers, dropout, early stopping)

- **Stacking Ensemble**:
  - Uses StratifiedKFold to generate out-of-fold predictions
  - Meta-learner (XGBoost or LogisticRegression) trained on base predictions
  - Avoids data leakage through proper CV strategy

### 6. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations**: ROC curves, confusion matrices, feature importance
- **Threshold Optimization**: Precision-recall curve analysis
- **Targets**: ~86% accuracy, ~0.94 AUC-ROC

---

## Performance Targets

- **Accuracy**: ≥ 0.86 (86%)
- **ROC-AUC**: ≥ 0.94
- **Precision**: Minimize false positive rate (mistakenly approved risky loans)
- **Recall**: Minimize false negative rate (missed bad loans)

---

## References

- [sklearn StackingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [UCI ML Repository - Credit Datasets](https://archive.ics.uci.edu/)

---

## License

MIT License

**Last Updated**: July 4, 2024
