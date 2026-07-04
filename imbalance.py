"""
Imbalance handling module for the Credit Risk Pipeline.

Implements SMOTE and class weight strategies for handling imbalanced datasets.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

logger = logging.getLogger(__name__)


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to training data.
    
    IMPORTANT: SMOTE should only be applied to training data to avoid data leakage.
    This function is designed to be called on training folds during cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        sampling_strategy: Desired ratio of minority to majority class
        random_state: Random seed
    
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not SMOTE_AVAILABLE:
        logger.warning("imbalanced-learn not installed. SMOTE not applied.")
        return X, y
    
    logger.info(f"Applying SMOTE with sampling_strategy={sampling_strategy}...")
    
    initial_dist = y.value_counts().to_dict()
    logger.info(f"Initial distribution: {initial_dist}")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)
    
    final_dist = y_resampled.value_counts().to_dict()
    logger.info(f"After SMOTE distribution: {final_dist}")
    
    return X_resampled, y_resampled


def apply_class_weight(
    class_weight: str = "balanced"
) -> dict:
    """
    Return class weights for imbalanced classification.
    
    For sklearn models that support class_weight parameter.
    
    Args:
        class_weight: Type of class weight ('balanced' or 'balanced_subsample')
    
    Returns:
        Class weight specification (usually just the string parameter)
    """
    logger.info(f"Using class_weight='{class_weight}'")
    return class_weight


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Imbalance handling module ready")
