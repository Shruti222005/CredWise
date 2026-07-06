"""
Updated main.py with proper SMOTE handling inside CV folds (BUG FIX #4).
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from datetime import datetime

# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("credit_risk_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from config import (
    RANDOM_STATE,
    IMBALANCE_STRATEGY,
    SMOTE_SAMPLING_STRATEGY,
    SMOTE_RANDOM_STATE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET_ACCURACY,
    TARGET_AUC_ROC,
)
from data_loader import load_and_unify_data
from preprocessing import preprocess_data
from feature_selection import select_features
from models import build_base_learners, build_stacking_classifier
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    optimize_threshold_pr_curve,
    print_evaluation_summary,
)

# Try to import imblearn for SMOTE-in-CV
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE in CV will be skipped.")


def make_balanced_estimator(base_model):
    """
    Wrap a base learner in an imblearn Pipeline with SMOTE.
    This applies SMOTE fresh inside each CV fold, avoiding data leakage.
    
    Args:
        base_model: Unfitted base learner (sklearn estimator)
    
    Returns:
        ImbPipeline with SMOTE + classifier
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("imblearn not available; returning base model without SMOTE")
        return base_model
    
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=SMOTE_RANDOM_STATE)),
        ("clf", base_model)
    ])


def wrap_base_learners_with_smote(base_learners: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap each base learner with SMOTE in a pipeline.
    
    Args:
        base_learners: Dict of base learners
    
    Returns:
        Dict of base learners wrapped with SMOTE
    """
    logger.info("Wrapping base learners with SMOTE (applied inside CV folds)...")
    
    wrapped = {}
    for name, model in base_learners.items():
        wrapped[name] = make_balanced_estimator(model)
    
    return wrapped


def save_artifacts(
    preprocessor: Any,
    feature_selector_metadata: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    timestamp: str
) -> None:
    """
    Save preprocessing and feature selection artifacts for future inference.
    """
    logger.info(f"Saving artifacts to {MODELS_DIR}...")
    
    import joblib
    
    # Save preprocessor
    joblib.dump(
        preprocessor,
        os.path.join(MODELS_DIR, f"preprocessor_{timestamp}.pkl")
    )
    
    # Save feature metadata
    feature_metadata = {
        "feature_names": X_train.columns.tolist(),
        "feature_selection_metadata": feature_selector_metadata,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(
        feature_metadata,
        os.path.join(MODELS_DIR, f"feature_metadata_{timestamp}.pkl")
    )
    
    logger.info("Artifacts saved successfully")


def save_final_models(
    models_dict: Dict[str, Any],
    timestamp: str
) -> None:
    """
    Save trained models for inference.
    """
    logger.info(f"Saving models to {MODELS_DIR}...")
    
    import joblib
    
    for model_name, model in models_dict.items():
        joblib.dump(
            model,
            os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")
        )
    
    logger.info("Models saved successfully")


def save_results_to_json(results: Dict, timestamp: str) -> None:
    """
    Save evaluation results to JSON for reproducibility.
    BUG FIX #5: Save actual metrics, not reference targets.
    """
    logger.info("Saving results to JSON...")
    
    results_path = os.path.join(MODELS_DIR, f"results_{timestamp}.json")
    
    # Flatten dict to make it JSON-serializable
    json_results = {
        "timestamp": results["timestamp"],
        "status": results["status"],
        "metrics": results["metrics"],
    }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


def run_pipeline() -> Dict[str, Any]:
    """
    Execute the complete credit risk prediction pipeline with proper SMOTE handling.
    
    Returns:
        Dictionary containing pipeline results and metadata
    """
    logger.info("\n" + "="*80)
    logger.info("CREDIT RISK PREDICTION PIPELINE")
    logger.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "status": "running",
        "metrics": {},
        "models": {},
    }
    
    try:
        # ====================================================================
        # 1. DATA LOADING & UNIFICATION
        # ====================================================================
        logger.info("\n[STEP 1] Loading and unifying datasets...")
        df, data_metadata = load_and_unify_data()
        results["data_metadata"] = data_metadata
        logger.info(f"Unified dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # ====================================================================
        # 2. PREPROCESSING
        # ====================================================================
        logger.info("\n[STEP 2] Preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data(df)
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution (train): {y_train.value_counts().to_dict()}")
        
        # ====================================================================
        # 3. FEATURE SELECTION
        # ====================================================================
        logger.info("\n[STEP 3] Feature selection...")
        X_train_fs, X_val_fs, X_test_fs, fs_metadata = select_features(
            X_train.copy(),
            X_val.copy(),
            X_test.copy(),
            y_train
        )
        results["feature_selection_metadata"] = fs_metadata
        logger.info(f"Features after selection: {X_train_fs.shape[1]}")
        
        # ====================================================================
        # 4. MODEL TRAINING (with SMOTE inside CV folds)
        # ====================================================================
        logger.info("\n[STEP 4] Building and training models with SMOTE in CV...")
        
        # Build base learners
        base_learners = build_base_learners()
        
        # BUG FIX #4: Wrap base learners with SMOTE (applied inside CV)
        logger.info("Wrapping base learners with SMOTE...")
        if IMBLEARN_AVAILABLE:
            balanced_learners = wrap_base_learners_with_smote(base_learners)
        else:
            logger.warning("imblearn not available; using base learners without SMOTE")
            balanced_learners = base_learners
        
        # Build stacking classifier (SMOTE is now inside CV via wrapped learners)
        logger.info("Building stacking classifier...")
        stacking_clf = build_stacking_classifier(balanced_learners)
        
        # Train stacking classifier on ORIGINAL (unbalanced) training data
        # SMOTE will be applied fresh inside each CV fold
        logger.info("Training stacking classifier (SMOTE applied inside CV folds)...")
        stacking_clf.fit(X_train_fs, y_train)
        results["models"]["stacking_ensemble"] = stacking_clf
        
        # ====================================================================
        # 5. EVALUATION
        # ====================================================================
        logger.info("\n[STEP 5] Evaluating models on test set...")
        
        # Prepare evaluation results
        all_results = {}
        roc_data = {}
        
        # Evaluate each base learner (retrained on full training set)
        logger.info("\nEvaluating base learners...")
        individual_models = {}
        for name, model in base_learners.items():
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_fs, y_train)  # No SMOTE here; only for stacking CV
            individual_models[name] = model_clone
            
            y_pred = model_clone.predict(X_test_fs)
            y_pred_proba = model_clone.predict_proba(X_test_fs)[:, 1]
            
            metrics = evaluate_model(y_test.values, y_pred, y_pred_proba, name)
            all_results[name] = metrics
            roc_data[name] = (y_test.values, y_pred_proba)
        
        # Evaluate stacking ensemble
        logger.info("\nEvaluating stacking ensemble...")
        y_pred_stack = stacking_clf.predict(X_test_fs)
        y_pred_proba_stack = stacking_clf.predict_proba(X_test_fs)[:, 1]
        
        metrics_stack = evaluate_model(
            y_test.values,
            y_pred_stack,
            y_pred_proba_stack,
            "Stacking Ensemble"
        )
        all_results["Stacking Ensemble"] = metrics_stack
        roc_data["Stacking Ensemble"] = (y_test.values, y_pred_proba_stack)
        
        results["metrics"] = all_results
        
        # ====================================================================
        # 6. VISUALIZATION & ANALYSIS
        # ====================================================================
        logger.info("\n[STEP 6] Generating visualizations...")
        
        # ROC curves
        logger.info("Plotting ROC curves...")
        fig_roc = plot_roc_curves(
            roc_data,
            output_path=os.path.join(MODELS_DIR, f"roc_curves_{timestamp}.png")
        )
        plt.close(fig_roc)
        
        # Confusion matrices
        logger.info("Plotting confusion matrices...")
        fig_cm_stack, ax_cm_stack = plt.subplots(figsize=(6, 5))
        plot_confusion_matrix(
            y_test.values,
            y_pred_stack,
            "Stacking Ensemble",
            ax=ax_cm_stack
        )
        fig_cm_stack.savefig(
            os.path.join(MODELS_DIR, f"confusion_matrix_stack_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig_cm_stack)
        
        # Feature importance (from Random Forest)
        if "RandomForest" in individual_models:
            logger.info("Plotting feature importance...")
            fig_fi = plot_feature_importance(
                individual_models["RandomForest"],
                X_train_fs.columns.tolist(),
                "Random Forest",
                top_n=20,
                output_path=os.path.join(MODELS_DIR, f"feature_importance_{timestamp}.png")
            )
            if fig_fi:
                plt.close(fig_fi)
        
        # Threshold optimization
        logger.info("Optimizing threshold via precision-recall curve...")
        opt_threshold, opt_precision, opt_recall, fig_pr = optimize_threshold_pr_curve(
            y_test.values,
            y_pred_proba_stack,
            output_path=os.path.join(MODELS_DIR, f"threshold_optimization_{timestamp}.png")
        )
        plt.close(fig_pr)
        results["optimal_threshold"] = opt_threshold
        
        # ====================================================================
        # 7. SUMMARY & ARTIFACTS
        # ====================================================================
        logger.info("\n[STEP 7] Saving artifacts and generating summary...")
        
        print_evaluation_summary(all_results, TARGET_ACCURACY, TARGET_AUC_ROC)
        
        # BUG FIX #5: Save actual results to JSON
        save_results_to_json(results, timestamp)
        
        # Save preprocessing artifacts
        save_artifacts(
            preprocessor,
            fs_metadata,
            X_train_fs,
            y_train,  # Original training labels (not SMOTE-balanced)
            timestamp
        )
        
        # Save models
        save_final_models(
            {"stacking_ensemble": stacking_clf, **individual_models},
            timestamp
        )
        
        results["status"] = "completed"
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results saved to: {os.path.join(MODELS_DIR, f'results_{timestamp}.json')}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    results = run_pipeline()
    
    logger.info(f"\nPipeline Results Summary:")
    logger.info(f"Status: {results['status']}")
    logger.info(f"Timestamp: {results['timestamp']}")
    
    if results['status'] == 'completed':
        logger.info(f"\n" + "="*80)
        logger.info("ACTUAL METRICS (not targets):")
        logger.info("="*80)
        for model_name, metrics in results['metrics'].items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
