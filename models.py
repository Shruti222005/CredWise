"""
Updated models.py with proper Keras handling and imports.
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

# Try to import Keras/TensorFlow
KERAS_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.callbacks import EarlyStopping
        KERAS_AVAILABLE = True
    except ImportError:
        pass

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
        Compiled Keras model or None if not available
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
        if not KERAS_AVAILABLE:
            logger.warning("Keras not available. Returning unfitted wrapper.")
            return self
        
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
        if not KERAS_AVAILABLE or self.model is None:
            return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])
        
        proba = self.model.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


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
