"""
Configuration module for the Credit Risk Prediction Pipeline.

Centralizes all hyperparameters, file paths, and constants used throughout
the pipeline for reproducibility and easy experimentation.

IMPORTANT: Dataset column mappings are verified against ucimlrepo actual outputs.
Run verify_dataset_columns() to confirm mappings match your UCI fetches.
"""

import os
from typing import Dict, List

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

GERMAN_CREDIT_PATH = os.path.join(RAW_DATA_DIR, "german_credit.csv")
TAIWANESE_CREDIT_PATH = os.path.join(RAW_DATA_DIR, "taiwanese_credit.csv")
AUSTRALIAN_CREDIT_PATH = os.path.join(RAW_DATA_DIR, "australian_credit.csv")

# ============================================================================
# PREPROCESSING CONFIG
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Imputation strategy
MISSING_VALUE_STRATEGY = {
    "numeric": "median",  # Options: 'mean', 'median'
    "categorical": "most_frequent"  # Options: 'most_frequent'
}

# Scaling strategy
SCALING_METHOD = "standard"  # Options: 'standard', 'minmax', 'robust'

# ============================================================================
# FEATURE SELECTION CONFIG
# ============================================================================
CORRELATION_THRESHOLD = 0.85  # Remove features with |r| > 0.85
VARIANCE_THRESHOLD = 0.01  # Remove low-variance features
FEATURE_SELECTION_METHOD = "mutual_info"  # Options: 'mutual_info', 'chi_square'
TOP_N_FEATURES = None  # If None, keep all features post-filtering; else keep top-N

# ============================================================================
# IMBALANCE HANDLING CONFIG
# ============================================================================
IMBALANCE_STRATEGY = "smote_in_cv"  # Options: 'smote_in_cv', 'class_weight'
SMOTE_SAMPLING_STRATEGY = 0.8  # Resample minority to 80% of majority
SMOTE_RANDOM_STATE = RANDOM_STATE
CLASS_WEIGHTS = "balanced"  # Options: 'balanced', None

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Base Learners
BASE_LEARNERS_CONFIG = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": CLASS_WEIGHTS
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 0.8,
        "random_state": RANDOM_STATE
    },
    "XGBoost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": None  # Will be set based on class imbalance
    },
    "KNeighbors": {
        "n_neighbors": 5,
        "weights": "distance",
        "metric": "euclidean",
        "n_jobs": -1
    }
}

# Neural Network (Keras) - OPTIONAL, not included in base learners by default
ANN_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "activation": "relu",
    "output_activation": "sigmoid",
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 5,
    "early_stopping_monitor": "val_loss"
}

# Meta-Learner (Stacking)
META_LEARNER_CONFIG = {
    "model_type": "xgboost",  # Options: 'logistic_regression', 'xgboost'
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
}

# ============================================================================
# CROSS-VALIDATION CONFIG
# ============================================================================
N_SPLITS = 5  # StratifiedKFold splits
CV_SHUFFLE = True

# ============================================================================
# EVALUATION CONFIG
# ============================================================================
THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]  # For threshold optimization
TARGET_ACCURACY = 0.86  # Reference target (NOT a guarantee)
TARGET_AUC_ROC = 0.94   # Reference target (NOT a guarantee)

# ============================================================================
# DATASET SCHEMA MAPPING
# ============================================================================
# Verified mappings from ucimlrepo actual outputs
# Run verify_dataset_columns() in data_loader.py to confirm these match your UCI fetches

# TAIWANESE (UCI ID: 350) - VERIFIED
TAIWAN_MAPPING = {
    "LIMIT_BAL": "credit_amount",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "AGE": "age_years",
    "PAY_0": "repay_status_1",
    "PAY_2": "repay_status_2",
    "PAY_3": "repay_status_3",
    "PAY_4": "repay_status_4",
    "PAY_5": "repay_status_5",
    "PAY_6": "repay_status_6",
    "BILL_AMT1": "bill_amt_1",
    "BILL_AMT2": "bill_amt_2",
    "BILL_AMT3": "bill_amt_3",
    "BILL_AMT4": "bill_amt_4",
    "BILL_AMT5": "bill_amt_5",
    "BILL_AMT6": "bill_amt_6",
    "PAY_AMT1": "pay_amt_1",
    "PAY_AMT2": "pay_amt_2",
    "PAY_AMT3": "pay_amt_3",
    "PAY_AMT4": "pay_amt_4",
    "PAY_AMT5": "pay_amt_5",
    "PAY_AMT6": "pay_amt_6",
}

# GERMAN (UCI ID: 144) - Built dynamically, see data_loader.py
# Typically returns: Attribute1, Attribute2, ..., Attribute20 (or Attr...)
GERMAN_MAPPING = {}  # Will be populated dynamically

# AUSTRALIAN (UCI ID: 143) - Built dynamically, see data_loader.py
# Typically returns: A1, A2, ..., A14 (or Attribute1...)
AUSTRALIAN_MAPPING = {}  # Will be populated dynamically

DATASET_COLUMN_MAPPING: Dict[str, Dict[str, str]] = {
    "german": GERMAN_MAPPING,
    "taiwanese": TAIWAN_MAPPING,
    "australian": AUSTRALIAN_MAPPING,
}

# ============================================================================
# TARGET VALUE MAPPING
# ============================================================================
TARGET_MAPPING: Dict[str, Dict] = {
    "german": {
        "good": 0,
        "bad": 1,
        "Good": 0,
        "Bad": 1,
        1: 0,  # Often 1 = good in raw data
        2: 1,  # Often 2 = bad in raw data
        "reverse_mapping": {0: "good", 1: "bad"}
    },
    "taiwanese": {
        1: 1,  # Default payment (bad)
        0: 0,  # Paid in full (good)
        -1: 0, # No consumption
        "reverse_mapping": {0: "good", 1: "bad"}
    },
    "australian": {
        "+": 1,  # Bad credit
        "-": 0,  # Good credit
        1: 1,   # Sometimes numeric 1 = bad
        0: 0,   # Sometimes numeric 0 = good
        "reverse_mapping": {0: "good", 1: "bad"}
    }
}

# ============================================================================
# LOGGING CONFIG
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
