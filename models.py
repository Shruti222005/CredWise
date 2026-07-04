"""
Models module for the Credit Risk Pipeline.

Implements:
1. Base learners (RF, GB, XGB, KNN, ANN)
2. Stacking ensemble with meta-learner
3. Out-of-fold prediction generation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

from config import (
    BASE_LEARNERS_CONFIG,
    ANN_CONFIG,
    META_LEARNER_CONFIG,
    N_SPLITS,
    RANDOM_STATE,
    CV_SHUFFLE,
)

logger = logging.getLogger(__name__)


def build_base_learners() -> Dict[str, Any]:
    """
    Build all base learner models.
    
    Returns:
        Dictionary of base learner models
    """
    logger.info("Building base learners...")
    
    base_learners = {}
    
    # Random Forest
    base_learners["RandomForest"] = RandomForestClassifier(
        **BASE_LEARNERS_CONFIG["RandomForest"]
    )
    
    # Gradient Boosting
    base_learners["GradientBoosting"] = GradientBoostingClassifier(
        **BASE_LEARNERS_CONFIG["GradientBoosting"]
    )
    
    # XGBoost
    base_learners["XGBoost"] = XGBClassifier(
        **BASE_LEARNERS_CONFIG["XGBoost"],
        eval_metric="logloss",
        verbosity=0
    )
    
    # K-Nearest Neighbors
    base_learners["KNeighbors"] = KNeighborsClassifier(
        **BASE_LEARNERS_CONFIG["KNeighbors"]
    )
    
    logger.info(f"Built {len(base_learners)} base learners")
    return base_learners


def build_ann(
    input_dim: int,
    hidden_layers: List[int] = None,
    dropout_rate: float = 0.3,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    loss: str = "binary_crossentropy",
    optimizer: str = "adam"
) -> Any:
    """
    Build an Artificial Neural Network using Keras.
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate
        activation: Activation function for hidden layers
        output_activation: Activation for output layer
        loss: Loss function
        optimizer: Optimizer
    
    Returns:
        Compiled Keras model
    """
    if not KERAS_AVAILABLE:
        logger.warning("TensorFlow/Keras not available. Skipping ANN.")
        return None
    
    if hidden_layers is None:
        hidden_layers = ANN_CONFIG["hidden_layers"]
    
    logger.info(f"Building ANN with hidden layers: {hidden_layers}")
    
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    for hidden_size in hidden_layers[1:]:
        model.add(Dense(hidden_size, activation=activation))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation=output_activation))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    
    return model


class ANNWrapper:
    """
    Wrapper to make Keras models compatible with sklearn's StackingClassifier.
    """
    def __init__(self, input_dim: int, **kwargs):
        self.input_dim = input_dim
        self.kwargs = kwargs
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = build_ann(self.input_dim, **self.kwargs)
        early_stop = EarlyStopping(
            monitor=ANN_CONFIG["early_stopping_monitor"],
            patience=ANN_CONFIG["early_stopping_patience"],
            restore_best_weights=True
        )
        self.model.fit(
            X, y,
            epochs=ANN_CONFIG["epochs"],
            batch_size=ANN_CONFIG["batch_size"],
            validation_split=ANN_CONFIG["validation_split"],
            callbacks=[early_stop],
            verbose=0
        )
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def generate_out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    base_learners: Dict[str, Any],
    n_splits: int = N_SPLITS
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate out-of-fold predictions for stacking.
    
    Uses StratifiedKFold to create training data for the meta-learner
    while avoiding data leakage.
    
    Args:
        X: Feature matrix
        y: Target vector
        base_learners: Dictionary of base learner models
        n_splits: Number of CV folds
    
    Returns:
        Tuple of (meta_features, oof_scores_dict)
    """
    logger.info(f"Generating out-of-fold predictions with {n_splits} folds...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)
    
    meta_features = np.zeros((X.shape[0], len(base_learners)))
    oof_scores = {name: [] for name in base_learners.keys()}
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        logger.info(f"  Processing fold {fold + 1}/{n_splits}")
        
        X_train_fold = X_arr[train_idx]
        X_val_fold = X_arr[val_idx]
        y_train_fold = y_arr[train_idx]
        y_val_fold = y_arr[val_idx]
        
        for idx, (name, model) in enumerate(base_learners.items()):
            logger.debug(f"    Training {name}...")
            
            # Clone model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            proba = model_clone.predict_proba(X_val_fold)[:, 1]
            meta_features[val_idx, idx] = proba
            
            # Store OOF score
            oof_auc = roc_auc_score(y_val_fold, proba)
            oof_scores[name].append(oof_auc)
    
    logger.info("Out-of-fold prediction generation complete")
    
    return meta_features, oof_scores


def build_stacking_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_learners: Dict[str, Any],
    meta_learner_config: Dict = None,
    use_oob: bool = False
) -> StackingClassifier:
    """
    Build a stacking classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target
        base_learners: Dictionary of base learners
        meta_learner_config: Configuration for meta-learner
        use_oob: Whether to use out-of-bag predictions
    
    Returns:
        Fitted StackingClassifier
    """
    logger.info("Building stacking classifier...")
    
    if meta_learner_config is None:
        meta_learner_config = META_LEARNER_CONFIG
    
    # Prepare base estimators list
    base_estimators = [
        (name, model) for name, model in base_learners.items()
    ]
    
    # Build meta-learner
    if meta_learner_config["model_type"] == "xgboost":
        final_estimator = XGBClassifier(**meta_learner_config["xgboost"])
    else:
        final_estimator = LogisticRegression(**meta_learner_config["logistic_regression"])
    
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
        print(f"{name}: {model}")
