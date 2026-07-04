"""
Evaluation module for the Credit Risk Pipeline.

Implements:
1. Evaluation metrics (accuracy, AUC, precision, recall, F1, confusion matrix)
2. ROC curve plotting
3. Feature importance visualization
4. Threshold optimization based on precision-recall
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a model and return multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for positive class)
        model_name: Name of the model
    
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    
    logger.info(f"{model_name} Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    ax: plt.Axes = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        ax: Matplotlib axis
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_xticklabels(["Good (0)", "Bad (1)"])
    ax.set_yticklabels(["Good (0)", "Bad (1)"])
    
    return fig


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to (y_true, y_pred_proba) tuples
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, (y_true, y_pred_proba) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curves saved to {output_path}")
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    output_path: str = None
) -> plt.Figure:
    """
    Plot feature importance from tree-based models.
    
    Args:
        model: Trained model (RandomForest, GradientBoosting, or XGBoost)
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to display
        output_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}")
    ax.grid(True, alpha=0.3, axis="x")
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance saved to {output_path}")
    
    return fig


def optimize_threshold_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = None
) -> Tuple[float, float, float, plt.Figure]:
    """
    Optimize threshold based on precision-recall curve.
    Maximizes F1-score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save figure
    
    Returns:
        Tuple of (optimal_threshold, precision, recall, figure)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, label="PR Curve")
    ax.plot(optimal_recall, optimal_precision, "ro", markersize=8,
            label=f"Optimal (threshold={optimal_threshold:.3f}, F1={optimal_f1:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve with Optimal Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Precision-recall curve saved to {output_path}")
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"  Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}, F1: {optimal_f1:.4f}")
    
    return optimal_threshold, optimal_precision, optimal_recall, fig


def print_evaluation_summary(
    results_dict: Dict[str, Dict[str, float]],
    target_accuracy: float = 0.86,
    target_auc: float = 0.94
) -> None:
    """
    Print evaluation summary comparing all models.
    
    Args:
        results_dict: Dictionary mapping model names to metric dictionaries
        target_accuracy: Target accuracy threshold
        target_auc: Target AUC-ROC threshold
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    for model_name, metrics in results_dict.items():
        logger.info(f"\n{model_name}:")
        for metric_name, metric_value in metrics.items():
            marker = ""
            if metric_name == "accuracy" and metric_value >= target_accuracy:
                marker = " ✓ (meets target)"
            elif metric_name == "roc_auc" and metric_value >= target_auc:
                marker = " ✓ (meets target)"
            logger.info(f"  {metric_name}: {metric_value:.4f}{marker}")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module ready")
