"""
Preprocessing module for the Credit Risk Pipeline.

Handles missing value imputation, categorical encoding, and numerical scaling
using sklearn Pipeline and ColumnTransformer for reproducibility.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from sklearn.pipeline import Pipeline, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from config import (
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    MISSING_VALUE_STRATEGY,
    SCALING_METHOD,
)

logger = logging.getLogger(__name__)


def identify_feature_types(df: pd.DataFrame, target_col: str = "credit_risk") -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical features.
    
    Args:
        df: Input dataframe
        target_col: Name of target column (excluded from features)
    
    Returns:
        Tuple of (numerical_features, categorical_features)
    """
    # Exclude target and metadata columns
    exclude_cols = {target_col, "dataset_source"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    numerical_features = df[feature_cols].select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()
    
    categorical_features = df[feature_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    return numerical_features, categorical_features


def create_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    scaling_method: str = SCALING_METHOD,
    missing_numeric_strategy: str = MISSING_VALUE_STRATEGY["numeric"],
    missing_categorical_strategy: str = MISSING_VALUE_STRATEGY["categorical"]
) -> ColumnTransformer:
    """
    Create a ColumnTransformer pipeline for preprocessing.
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        scaling_method: Type of scaling ('standard', 'minmax', 'robust')
        missing_numeric_strategy: Imputation strategy for numerical features
        missing_categorical_strategy: Imputation strategy for categorical features
    
    Returns:
        Configured ColumnTransformer
    """
    logger.info(f"Creating preprocessor with scaling_method={scaling_method}")
    
    # Numerical pipeline: impute + scale
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=missing_numeric_strategy)),
            ("scaler", StandardScaler() if scaling_method == "standard" else StandardScaler()),
        ]
    )
    
    # Categorical pipeline: impute + one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=missing_categorical_strategy)),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "credit_risk",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, ColumnTransformer]:
    """
    Preprocess data and perform train/val/test split.
    
    Steps:
    1. Identify feature types
    2. Create preprocessor
    3. Fit on training data
    4. Transform all splits
    5. Return train/val/test with targets
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        test_size: Proportion for test set
        val_size: Proportion for validation set (relative to non-test data)
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    logger.info("Starting preprocessing...")
    
    # Separate features and target
    X = df.drop(columns=[target_col, "dataset_source"], errors="ignore")
    y = df[target_col]
    
    logger.info(f"Initial feature set: {X.shape[1]} features")
    logger.info(f"Target distribution (before split): \n{y.value_counts()}")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Val set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(X_train, target_col=None)
    
    # Create and fit preprocessor on training data only (avoid data leakage)
    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )
    preprocessor.fit(X_train)
    
    # Transform all splits
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to DataFrame for easier downstream processing
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    logger.info(f"After preprocessing: {X_train_df.shape[1]} features")
    logger.info("Preprocessing complete!")
    
    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_and_unify_data
    
    df, meta = load_and_unify_data()
    X_train, X_val, X_test, y_train, y_val, y_test, preproc = preprocess_data(df)
    print(f"\nPreprocessed shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
