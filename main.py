"""
Main orchestration script for the Credit Risk Prediction Pipeline.

Coordinates all steps:
1. Data loading and unification
2. Preprocessing
3. Feature selection
4. Imbalance handling
5. Model training (base learners + stacking)
6. Evaluation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from datetime import datetime

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
from imbalance import apply_smote, apply_class_weight
from models import build_base_learners, build_stacking_classifier, generate_out_of_fold_predictions
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    optimize_threshold_pr_curve,
    print_evaluation_summary,
)


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


def run_pipeline() -> Dict[str, Any]:
    """
    Execute the complete credit risk prediction pipeline.
    
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
        # 4. IMBALANCE HANDLING
        # ====================================================================
        logger.info("\n[STEP 4] Handling class imbalance...")
        
        X_train_balanced = X_train_fs.copy()
        y_train_balanced = y_train.copy()
        
        if IMBALANCE_STRATEGY in ["smote", "both"]:
            logger.info(f"Applying SMOTE (sampling_strategy={SMOTE_SAMPLING_STRATEGY})...")
            X_train_balanced, y_train_balanced = apply_smote(
                X_train_balanced,
                y_train_balanced,
                sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                random_state=SMOTE_RANDOM_STATE
            )
            logger.info(f"After SMOTE: {y_train_balanced.value_counts().to_dict()}")
        
        # ====================================================================
        # 5. MODEL TRAINING
        # ====================================================================
        logger.info("\n[STEP 5] Building and training models...")
        
        # Build base learners
        base_learners = build_base_learners()
        
        # Build stacking classifier
        logger.info("Building stacking classifier...")
        stacking_clf = build_stacking_classifier(
            X_train_balanced,
            y_train_balanced,
            base_learners
        )
        
        # Train stacking classifier
        logger.info("Training stacking classifier...")
        stacking_clf.fit(X_train_balanced, y_train_balanced)
        results["models"]["stacking_ensemble"] = stacking_clf
        
        # Train individual base learners for evaluation
        logger.info("Training individual base learners...")
        individual_models = {}
        for name, model in base_learners.items():
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train_balanced, y_train_balanced)
            individual_models[name] = model_clone
            results["models"][name] = model_clone
        
        # ====================================================================
        # 6. EVALUATION
        # ====================================================================
        logger.info("\n[STEP 6] Evaluating models on test set...")
        
        # Prepare evaluation results
        all_results = {}
        roc_data = {}
        
        # Evaluate individual base learners
        logger.info("\nEvaluating base learners...")
        for name, model in individual_models.items():
            y_pred = model.predict(X_test_fs)
            y_pred_proba = model.predict_proba(X_test_fs)[:, 1]
            
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
        # 7. VISUALIZATION & ANALYSIS
        # ====================================================================
        logger.info("\n[STEP 7] Generating visualizations...")
        
        # ROC curves
        logger.info("Plotting ROC curves...")
        fig_roc = plot_roc_curves(
            roc_data,
            output_path=os.path.join(MODELS_DIR, f"roc_curves_{timestamp}.png")
        )
        
        # Confusion matrices
        logger.info("Plotting confusion matrices...")
        fig_cm_stack, ax_cm_stack = plt.subplots()
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
        
        # Feature importance (from Random Forest)
        if "RandomForest" in individual_models:
            logger.info("Plotting feature importance...")
            plot_feature_importance(
                individual_models["RandomForest"],
                X_train_fs.columns.tolist(),
                "Random Forest",
                top_n=20,
                output_path=os.path.join(MODELS_DIR, f"feature_importance_{timestamp}.png")
            )
        
        # Threshold optimization
        logger.info("Optimizing threshold via precision-recall curve...")
        opt_threshold, opt_precision, opt_recall, fig_pr = optimize_threshold_pr_curve(
            y_test.values,
            y_pred_proba_stack,
            output_path=os.path.join(MODELS_DIR, f"threshold_optimization_{timestamp}.png")
        )
        results["optimal_threshold"] = opt_threshold
        
        # ====================================================================
        # 8. SUMMARY & ARTIFACTS
        # ====================================================================
        logger.info("\n[STEP 8] Saving artifacts and generating summary...")
        
        print_evaluation_summary(all_results, TARGET_ACCURACY, TARGET_AUC_ROC)
        
        # Save preprocessing artifacts
        save_artifacts(
            preprocessor,
            fs_metadata,
            X_train_fs,
            y_train_balanced,
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
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')  # Use non-interactive backend
    
    results = run_pipeline()
    
    logger.info(f"\nPipeline Results Summary:")
    logger.info(f"Status: {results['status']}")
    logger.info(f"Timestamp: {results['timestamp']}")
    
    if results['status'] == 'completed':
        logger.info(f"\nBase Model Performance:")
        for model_name, metrics in results['metrics'].items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
