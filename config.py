"""
Configuration module for the Credit Risk Prediction Pipeline.

Centralizes all hyperparameters, file paths, and constants used throughout
the pipeline for reproducibility and easy experimentation.
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
IMBALANCE_STRATEGY = "smote"  # Options: 'smote', 'class_weight', 'both'
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

# Neural Network (Keras)
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
TARGET_ACCURACY = 0.86
TARGET_AUC_ROC = 0.94

# ============================================================================
# DATASET SCHEMA MAPPING
# ============================================================================
# Map dataset-specific column names to unified schema
DATASET_COLUMN_MAPPING: Dict[str, Dict[str, str]] = {
    "german": {
        "status": "account_status",
        "duration": "credit_duration_months",
        "credit_history": "credit_history",
        "purpose": "credit_purpose",
        "amount": "credit_amount",
        "savings": "savings_account",
        "employment": "employment_status",
        "installment_rate": "installment_rate",
        "personal_status": "personal_status",
        "debtors": "debtors_guarantors",
        "residence": "residence_duration_years",
        "property": "property",
        "age": "age_years",
        "other_plans": "other_installment_plans",
        "housing": "housing",
        "existing_credits": "num_existing_credits",
        "job": "job_type",
        "dependents": "num_dependents",
        "telephone": "has_telephone",
        "foreign_worker": "is_foreign_worker",
        "target": "credit_risk"
    },
    "taiwanese": {
        "ID": "customer_id",
        "LIMIT_BAL": "credit_limit",
        "SEX": "sex",
        "EDUCATION": "education_level",
        "MARRIAGE": "marital_status",
        "AGE": "age_years",
        "PAY_1": "repayment_status_sep",
        "BILL_AMT1": "bill_amount_sep",
        "PAY_AMT1": "payment_amount_sep",
        "target": "credit_risk"
    },
    "australian": {
        "A1": "age_group",
        "A2": "sex",
        "A3": "employment_status",
        "A4": "credit_history",
        "A5": "credit_purpose",
        "A6": "credit_amount",
        "A7": "savings_account",
        "A8": "employment_duration_years",
        "A9": "installment_rate",
        "A10": "personal_status",
        "A11": "debtors_guarantors",
        "A12": "residence_duration_years",
        "A13": "property",
        "A14": "age_years",
        "A15": "other_installment_plans",
        "A16": "housing",
        "A17": "num_existing_credits",
        "A18": "job_type",
        "A19": "num_dependents",
        "A20": "has_telephone",
        "target": "credit_risk"
    }
}

# ============================================================================
# TARGET VALUE MAPPING
# ============================================================================
TARGET_MAPPING: Dict[str, Dict] = {
    "german": {
        "good": 0,
        "bad": 1,
        "reverse_mapping": {0: "good", 1: "bad"}
    },
    "taiwanese": {
        1: 1,  # Default payment (bad)
        0: 0,  # Paid in full (good)
        "reverse_mapping": {0: "good", 1: "bad"}
    },
    "australian": {
        "+": 1,  # Bad credit
        "-": 0,  # Good credit
        "reverse_mapping": {0: "good", 1: "bad"}
    }
}

# ============================================================================
# LOGGING CONFIG
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
