"""
Data loading module with UCI ML Repo integration and dynamic schema discovery.

Automatically fetches datasets from UCI ML Repository and dynamically discovers
column names instead of relying on hardcoded guesses.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from config import (
    DATASET_COLUMN_MAPPING,
    TARGET_MAPPING,
    TAIWAN_MAPPING,
    PROCESSED_DATA_DIR,
)

try:
    from ucimlrepo import fetch_ucirepo
    UCIMLREPO_AVAILABLE = True
except ImportError:
    UCIMLREPO_AVAILABLE = False

logger = logging.getLogger(__name__)


def verify_dataset_columns():
    """
    One-time verification: print actual column names from UCI datasets.
    Run this once manually to verify config.py mappings are correct.
    """
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Cannot verify columns.")
        return
    
    logger.info("\n" + "="*80)
    logger.info("DATASET COLUMN VERIFICATION")
    logger.info("="*80)
    
    for name, id_ in [("german", 144), ("taiwanese", 350), ("australian", 143)]:
        try:
            d = fetch_ucirepo(id=id_)
            features_cols = list(d.data.features.columns)
            targets_cols = list(d.data.targets.columns)
            logger.info(f"\n{name.upper()} (ID: {id_})")
            logger.info(f"  Feature columns: {features_cols}")
            logger.info(f"  Target column: {targets_cols}")
        except Exception as e:
            logger.error(f"Error fetching {name}: {str(e)}")


def load_german_credit() -> Tuple[pd.DataFrame, list]:
    """
    Load German Credit dataset from UCI ML Repository (ID: 144).
    Dynamically discovers column names and builds mapping.
    
    Returns:
        Tuple of (DataFrame, list of feature column names)
    """
    logger.info("Fetching German Credit dataset from UCI ML Repo (ID: 144)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        german = fetch_ucirepo(id=144)
        X = german.data.features
        y = german.data.targets
        
        feature_cols = list(X.columns)
        target_col = list(y.columns)[0]
        
        logger.info(f"German Credit feature columns: {feature_cols}")
        logger.info(f"German Credit target column: {target_col}")
        
        df = X.copy()
        df['credit_risk'] = y.iloc[:, 0]
        
        logger.info(f"German Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target distribution: {df['credit_risk'].value_counts().to_dict()}")
        
        return df, feature_cols
    except Exception as e:
        logger.error(f"Error loading German Credit dataset: {str(e)}")
        raise


def load_taiwanese_credit() -> Tuple[pd.DataFrame, list]:
    """
    Load Taiwanese Credit dataset from UCI ML Repository (ID: 350).
    Uses verified mapping for Taiwanese dataset.
    
    Returns:
        Tuple of (DataFrame, list of feature column names)
    """
    logger.info("Fetching Taiwanese Credit dataset from UCI ML Repo (ID: 350)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        taiwan = fetch_ucirepo(id=350)
        X = taiwan.data.features
        y = taiwan.data.targets
        
        feature_cols = list(X.columns)
        target_col = list(y.columns)[0]
        
        logger.info(f"Taiwanese Credit feature columns: {feature_cols}")
        logger.info(f"Taiwanese Credit target column: {target_col}")
        
        # Apply verified mapping
        rename_dict = {k: v for k, v in TAIWAN_MAPPING.items() if k in feature_cols}
        X_renamed = X.rename(columns=rename_dict)
        
        df = X_renamed.copy()
        df['credit_risk'] = y.iloc[:, 0]
        
        logger.info(f"Taiwanese Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target distribution: {df['credit_risk'].value_counts().to_dict()}")
        
        return df, list(X_renamed.columns)
    except Exception as e:
        logger.error(f"Error loading Taiwanese Credit dataset: {str(e)}")
        raise


def load_australian_credit() -> Tuple[pd.DataFrame, list]:
    """
    Load Australian Credit dataset from UCI ML Repository (ID: 143).
    Dynamically discovers and maps column names.
    
    Returns:
        Tuple of (DataFrame, list of feature column names)
    """
    logger.info("Fetching Australian Credit dataset from UCI ML Repo (ID: 143)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        australian = fetch_ucirepo(id=143)
        X = australian.data.features
        y = australian.data.targets
        
        feature_cols = list(X.columns)
        target_col = list(y.columns)[0]
        
        logger.info(f"Australian Credit feature columns: {feature_cols}")
        logger.info(f"Australian Credit target column: {target_col}")
        
        # Build a generic mapping (A1->attr_1, A2->attr_2, etc.)
        dynamic_mapping = {col: f"attr_{i+1}" for i, col in enumerate(feature_cols)}
        X_renamed = X.rename(columns=dynamic_mapping)
        
        df = X_renamed.copy()
        df['credit_risk'] = y.iloc[:, 0]
        
        logger.info(f"Australian Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target distribution: {df['credit_risk'].value_counts().to_dict()}")
        
        return df, list(X_renamed.columns)
    except Exception as e:
        logger.error(f"Error loading Australian Credit dataset: {str(e)}")
        raise


def standardize_targets(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Standardize target values to 0 (good) and 1 (bad).
    
    Args:
        df: DataFrame with 'credit_risk' column
        dataset_name: 'german', 'taiwanese', or 'australian'
    
    Returns:
        DataFrame with standardized target
    """
    if 'credit_risk' not in df.columns:
        logger.warning(f"No 'credit_risk' column found in {dataset_name} data")
        return df
    
    df = df.copy()
    target_map = TARGET_MAPPING[dataset_name]
    
    # Try mapping each unique value
    for val in df['credit_risk'].unique():
        if val in target_map:
            df.loc[df['credit_risk'] == val, 'credit_risk'] = target_map[val]
    
    logger.info(f"Standardized {dataset_name} target: {df['credit_risk'].value_counts().to_dict()}")
    return df


def unify_datasets(
    german_df: pd.DataFrame,
    taiwanese_df: pd.DataFrame,
    australian_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge standardized datasets into a unified training set.
    Keeps only common columns across all datasets.
    
    Args:
        german_df: Standardized German Credit DataFrame
        taiwanese_df: Standardized Taiwanese Credit DataFrame
        australian_df: Standardized Australian Credit DataFrame
    
    Returns:
        Unified DataFrame with common features and target
    """
    logger.info("Unifying datasets...")
    
    # Get common columns
    cols_german = set(german_df.columns)
    cols_taiwanese = set(taiwanese_df.columns)
    cols_australian = set(australian_df.columns)
    
    common_cols = cols_german & cols_taiwanese & cols_australian
    
    logger.info(f"German columns: {len(cols_german)}")
    logger.info(f"Taiwanese columns: {len(cols_taiwanese)}")
    logger.info(f"Australian columns: {len(cols_australian)}")
    logger.info(f"Common columns: {len(common_cols)}")
    
    # BUG FIX #2: Assert loudly if common columns are too few
    assert len(common_cols) >= 10, (
        f"Only {len(common_cols)} common columns found after mapping. "
        f"This indicates DATASET_COLUMN_MAPPING is broken. "
        f"Check actual column names from ucimlrepo against config.py. "
        f"Common cols: {sorted(common_cols)}"
    )
    
    # Select and concatenate
    unified_df = pd.concat(
        [
            german_df[list(common_cols)],
            taiwanese_df[list(common_cols)],
            australian_df[list(common_cols)]
        ],
        axis=0,
        ignore_index=True
    )
    
    logger.info(f"Unified dataset shape: {unified_df.shape}")
    logger.info(f"Target distribution: {unified_df['credit_risk'].value_counts().to_dict()}")
    logger.info(f"Common feature columns kept: {sorted(common_cols)}")
    
    return unified_df


def load_and_unify_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load and unify all datasets from UCI ML Repository.
    
    Returns:
        Tuple of (unified_df, metadata_dict)
    """
    logger.info("\n" + "="*80)
    logger.info("DATA LOADING AND UNIFICATION")
    logger.info("="*80)
    
    try:
        # Load individual datasets
        german, german_cols = load_german_credit()
        taiwanese, taiwanese_cols = load_taiwanese_credit()
        australian, australian_cols = load_australian_credit()
        
        logger.info(f"\nSuccessfully loaded datasets:")
        logger.info(f"  German: {german.shape}")
        logger.info(f"  Taiwanese: {taiwanese.shape}")
        logger.info(f"  Australian: {australian.shape}")
        
        # Standardize targets
        german = standardize_targets(german, "german")
        taiwanese = standardize_targets(taiwanese, "taiwanese")
        australian = standardize_targets(australian, "australian")
        
        # Unify
        unified = unify_datasets(german, taiwanese, australian)
        
        # Metadata
        metadata = {
            "total_rows": len(unified),
            "total_columns": len(unified.columns),
            "target_distribution": unified["credit_risk"].value_counts().to_dict(),
            "class_distribution_pct": (unified["credit_risk"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "missing_values": unified.isnull().sum().to_dict(),
            "feature_columns": [col for col in unified.columns if col != "credit_risk"],
        }
        
        logger.info("\nData loading and unification complete!")
        return unified, metadata
        
    except Exception as e:
        logger.error(f"Error during data loading and unification: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # First, verify dataset columns
    logger.info("Running column verification...")
    verify_dataset_columns()
    
    # Then load and unify
    logger.info("\nLoading and unifying datasets...")
    df, meta = load_and_unify_data()
    print("\n" + "="*80)
    print("UNIFIED DATASET PREVIEW")
    print("="*80)
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMetadata:")
    for key, val in meta.items():
        print(f"{key}: {val}")
