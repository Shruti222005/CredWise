"""
Updated models.py with SMOTE removed from here (moved to within CV).
Base learners: RandomForest, GradientBoosting, XGBoost, KNeighbors (4 models)
ANN is optional and NOT included by default.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config import (
    BASE_LEARNERS_CONFIG,
    META_LEARNER_CONFIG,
    N_SPLITS,
    RANDOM_STATE,
    CV_SHUFFLE,
)

logger = logging.getLogger(__name__)


def build_base_learners() -> Dict[str, Any]:
    """
    Build 4 base learner models (RandomForest, GradientBoosting, XGBoost, KNeighbors).
    ANN is NOT included here (optional, requires TensorFlow).
    
    Returns:
        Dictionary of 4 base learner models
    """
    logger.info("Building base learners...")
    
    base_learners = {}
    
    # Random Forest
    base_learners["RandomForest"] = RandomForestClassifier(
        **BASE_LEARNERS_CONFIG["RandomForest"]
    )
    logger.info("  ✓ RandomForest")
    
    # Gradient Boosting
    base_learners["GradientBoosting"] = GradientBoostingClassifier(
        **BASE_LEARNERS_CONFIG["GradientBoosting"]
    )
    logger.info("  ✓ GradientBoosting")
    
    # XGBoost
    base_learners["XGBoost"] = XGBClassifier(
        **BASE_LEARNERS_CONFIG["XGBoost"],
        eval_metric="logloss",
        verbosity=0
    )
    logger.info("  ✓ XGBoost")
    
    # K-Nearest Neighbors
    base_learners["KNeighbors"] = KNeighborsClassifier(
        **BASE_LEARNERS_CONFIG["KNeighbors"]
    )
    logger.info("  ✓ KNeighbors")
    
    logger.info(f"Built {len(base_learners)} base learners")
    return base_learners


def build_stacking_classifier(
    base_learners: Dict[str, Any],
    meta_learner_config: Dict = None,
) -> StackingClassifier:
    """
    Build a stacking classifier with balanced SMOTE applied inside CV folds.
    
    SMOTE is NOT applied here; it is applied inside each CV fold via imblearn Pipeline.
    See main.py for details.
    
    Args:
        base_learners: Dictionary of base learners
        meta_learner_config: Configuration for meta-learner
    
    Returns:
        Configured StackingClassifier (not yet fit)
    """
    logger.info("Building stacking classifier...")
    
    if meta_learner_config is None:
        meta_learner_config = META_LEARNER_CONFIG
    
    # Prepare base estimators list
    base_estimators = [(name, model) for name, model in base_learners.items()]
    
    # Build meta-learner
    if meta_learner_config["model_type"] == "xgboost":
        final_estimator = XGBClassifier(**meta_learner_config["xgboost"])
    else:
        final_estimator = LogisticRegression(**meta_learner_config["logistic_regression"])
    
    logger.info(f"  Base learners: {[name for name, _ in base_estimators]}")
    logger.info(f"  Meta-learner: {meta_learner_config['model_type']}")
    
    # Build stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)
    )
    
    logger.info("Stacking classifier built")
    return stacking_clf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_learners = build_base_learners()
    for name, model in base_learners.items():
        print(f"{name}: {type(model).__name__}")
