"""
Feature selection module for the Credit Risk Pipeline.

Implements:
1. Correlation-based filtering (remove |r| > 0.85)
2. Variance-based filtering (low-variance features)
3. Feature ranking (mutual information, chi-square)
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    chi2,
    SelectKBest,
)
from scipy.stats import pearsonr
from config import (
    CORRELATION_THRESHOLD,
    VARIANCE_THRESHOLD,
    FEATURE_SELECTION_METHOD,
    TOP_N_FEATURES,
)

logger = logging.getLogger(__name__)


def remove_high_correlation_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = CORRELATION_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with high pairwise correlation (|r| > threshold).
    Keeps the feature more correlated with the target.
    
    Args:
        X: Feature matrix
        y: Target vector
        threshold: Correlation threshold
    
    Returns:
        Tuple of (filtered_X, removed_features)
    """
    logger.info(f"Removing features with |correlation| > {threshold}...")
    
    # Calculate target correlation for each feature
    target_corr = X.corrwith(y).abs()
    
    # Calculate pairwise correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find features to remove
    removed_features = []
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    for i in range(len(corr_matrix.columns)):
        if corr_matrix.columns[i] in removed_features:
            continue
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.columns[j] in removed_features:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                # Remove the one less correlated with target
                if target_corr.iloc[i] < target_corr.iloc[j]:
                    removed_features.append(corr_matrix.columns[i])
                else:
                    removed_features.append(corr_matrix.columns[j])
    
    removed_features = list(set(removed_features))
    X_filtered = X.drop(columns=removed_features)
    
    logger.info(f"Removed {len(removed_features)} highly correlated features: {removed_features}")
    logger.info(f"Remaining features: {X_filtered.shape[1]}")
    
    return X_filtered, removed_features


def remove_low_variance_features(
    X: pd.DataFrame,
    threshold: float = VARIANCE_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with variance below threshold.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
    
    Returns:
        Tuple of (filtered_X, removed_features)
    """
    logger.info(f"Removing features with variance < {threshold}...")
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    X_filtered = X[X.columns[selector.get_support()]]
    removed_features = X.columns[~selector.get_support()].tolist()
    
    logger.info(f"Removed {len(removed_features)} low-variance features: {removed_features}")
    logger.info(f"Remaining features: {X_filtered.shape[1]}")
    
    return X_filtered, removed_features


def rank_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = FEATURE_SELECTION_METHOD,
    top_n: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank features by mutual information or chi-square score.
    Optionally keep only top-N features.
    
    Args:
        X: Feature matrix (should contain only non-negative values for chi2)
        y: Target vector
        method: 'mutual_info' or 'chi_square'
        top_n: If specified, keep only top N features
    
    Returns:
        Tuple of (filtered_X, feature_scores_df)
    """
    logger.info(f"Ranking features by {method}...")
    
    if method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == "chi_square":
        # Ensure non-negative values for chi2
        X_nonneg = X - X.min() + 1e-6
        scores = chi2(X_nonneg, y)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create scores dataframe
    scores_df = pd.DataFrame({
        "feature": X.columns,
        "score": scores
    }).sort_values("score", ascending=False).reset_index(drop=True)
    
    logger.info(f"Top 10 features:\n{scores_df.head(10)}")
    
    # Filter by top-N if specified
    if top_n is not None and top_n < len(X.columns):
        top_features = scores_df.head(top_n)["feature"].tolist()
        X_filtered = X[top_features]
        logger.info(f"Kept top {top_n} features")
    else:
        X_filtered = X
        logger.info(f"Kept all {X.shape[1]} features")
    
    return X_filtered, scores_df


def select_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    remove_correlation: bool = True,
    remove_variance: bool = True,
    rank_by_importance: bool = True,
    top_n: int = TOP_N_FEATURES
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main feature selection pipeline.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train: Training target
        remove_correlation: Whether to remove correlated features
        remove_variance: Whether to remove low-variance features
        rank_by_importance: Whether to rank by importance
        top_n: Keep top-N features (if rank_by_importance=True)
    
    Returns:
        Tuple of (X_train_fs, X_val_fs, X_test_fs, metadata)
    """
    logger.info("Starting feature selection...")
    logger.info(f"Initial features: {X_train.shape[1]}")
    
    metadata = {
        "initial_features": X_train.shape[1],
        "removed_by_correlation": [],
        "removed_by_variance": [],
        "ranking_scores": None,
        "final_features": 0
    }
    
    # Step 1: Remove high-correlation features
    if remove_correlation:
        X_train, removed_corr = remove_high_correlation_features(X_train, y_train)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        metadata["removed_by_correlation"] = removed_corr
    
    # Step 2: Remove low-variance features
    if remove_variance:
        X_train, removed_var = remove_low_variance_features(X_train)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        metadata["removed_by_variance"] = removed_var
    
    # Step 3: Rank and filter by importance
    if rank_by_importance:
        X_train, scores_df = rank_features_by_importance(X_train, y_train, top_n=top_n)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        metadata["ranking_scores"] = scores_df
    
    metadata["final_features"] = X_train.shape[1]
    
    logger.info(f"Feature selection complete. Final features: {X_train.shape[1]}")
    logger.info(f"Features kept: {X_train.columns.tolist()}")
    
    return X_train, X_val, X_test, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from preprocessing import preprocess_data
    from data_loader import load_and_unify_data
    
    df, _ = load_and_unify_data()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)
    X_tr_fs, X_val_fs, X_te_fs, meta = select_features(X_train, X_val, X_test, y_train)
    print("\nFeature Selection Metadata:")
    for key, val in meta.items():
        if key != "ranking_scores":
            print(f"{key}: {val}")
