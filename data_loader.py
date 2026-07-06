"""
Data loading module with UCI ML Repo integration.

Automatically fetches German Credit, Taiwanese Credit, and Australian Credit
datasets from UCI ML Repository using ucimlrepo library.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from config import (
    DATASET_COLUMN_MAPPING,
    TARGET_MAPPING,
    PROCESSED_DATA_DIR,
)

try:
    from ucimlrepo import fetch_ucirepo
    UCIMLREPO_AVAILABLE = True
except ImportError:
    UCIMLREPO_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_german_credit() -> pd.DataFrame:
    """
    Load German Credit dataset from UCI ML Repository.
    Dataset ID: 144 (or 522 for South German Credit - corrected version)
    Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    
    Returns:
        DataFrame with features and target combined
    """
    logger.info("Fetching German Credit dataset from UCI ML Repo (ID: 144)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        german = fetch_ucirepo(id=144)
        X = german.data.features
        y = german.data.targets
        
        df = X.copy()
        df['target'] = y.iloc[:, 0]  # Extract target column
        
        logger.info(f"German Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading German Credit dataset: {str(e)}")
        raise


def load_taiwanese_credit() -> pd.DataFrame:
    """
    Load Taiwanese Credit dataset from UCI ML Repository.
    Dataset ID: 350
    Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    
    Returns:
        DataFrame with features and target combined
    """
    logger.info("Fetching Taiwanese Credit dataset from UCI ML Repo (ID: 350)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        taiwan = fetch_ucirepo(id=350)
        X = taiwan.data.features
        y = taiwan.data.targets
        
        df = X.copy()
        df['target'] = y.iloc[:, 0]  # Extract target column
        
        logger.info(f"Taiwanese Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading Taiwanese Credit dataset: {str(e)}")
        raise


def load_australian_credit() -> pd.DataFrame:
    """
    Load Australian Credit dataset from UCI ML Repository.
    Dataset ID: 143
    Source: https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval
    
    Returns:
        DataFrame with features and target combined
    """
    logger.info("Fetching Australian Credit dataset from UCI ML Repo (ID: 143)...")
    
    if not UCIMLREPO_AVAILABLE:
        logger.error("ucimlrepo not installed. Install with: pip install ucimlrepo")
        raise ImportError("ucimlrepo library required. Install with: pip install ucimlrepo")
    
    try:
        australian = fetch_ucirepo(id=143)
        X = australian.data.features
        y = australian.data.targets
        
        df = X.copy()
        df['target'] = y.iloc[:, 0]  # Extract target column
        
        logger.info(f"Australian Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading Australian Credit dataset: {str(e)}")
        raise


def standardize_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize German Credit dataset schema.
    Maps original column names to unified schema.
    """
    logger.info("Standardizing German Credit schema...")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Rename columns based on mapping
    rename_dict = DATASET_COLUMN_MAPPING["german"]
    # Only rename columns that exist
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Map target variable: 'good' -> 0, 'bad' -> 1
    target_map = TARGET_MAPPING["german"]
    if 'credit_risk' in df.columns:
        df['credit_risk'] = df['credit_risk'].map(lambda x: target_map.get(x.lower() if isinstance(x, str) else x, 1))
    elif 'target' in df.columns:
        df = df.rename(columns={'target': 'credit_risk'})
        df['credit_risk'] = df['credit_risk'].map(lambda x: target_map.get(x.lower() if isinstance(x, str) else x, 1))
    
    # Add dataset source identifier
    df['dataset_source'] = 'german'
    
    logger.info(f"Standardized German Credit schema. Target distribution: {df['credit_risk'].value_counts().to_dict()}")
    return df


def standardize_taiwanese_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Taiwanese Credit dataset schema.
    Maps original columns and handles multiple repayment history columns.
    """
    logger.info("Standardizing Taiwanese Credit schema...")
    
    df = df.copy()
    
    # Rename columns based on mapping
    rename_dict = DATASET_COLUMN_MAPPING["taiwanese"]
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Extract key features from repayment history columns (PAY_1 to PAY_6 in original)
    # Aggregate repayment status if multiple PAY columns exist
    pay_cols = [col for col in df.columns if 'PAY' in col.upper() and col not in ['payment_amount_sep']]
    if len(pay_cols) > 1:
        logger.info(f"Aggregating {len(pay_cols)} repayment history columns...")
        df['avg_repayment_status'] = df[pay_cols].mean(axis=1)
        df['max_repayment_status'] = df[pay_cols].max(axis=1)
        df = df.drop(columns=pay_cols, errors="ignore")
    
    # Map target variable
    target_map = TARGET_MAPPING["taiwanese"]
    if 'credit_risk' in df.columns:
        df['credit_risk'] = df['credit_risk'].map(target_map)
    elif 'target' in df.columns:
        df = df.rename(columns={'target': 'credit_risk'})
        df['credit_risk'] = df['credit_risk'].map(target_map)
    
    # Add dataset source identifier
    df['dataset_source'] = 'taiwanese'
    
    logger.info(f"Standardized Taiwanese Credit schema. Target distribution: {df['credit_risk'].value_counts().to_dict()}")
    return df


def standardize_australian_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Australian Credit dataset schema.
    Maps original column names (A1, A2, etc.) to unified schema.
    """
    logger.info("Standardizing Australian Credit schema...")
    
    df = df.copy()
    
    # Rename columns based on mapping
    rename_dict = DATASET_COLUMN_MAPPING["australian"]
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    # Map target variable: '+' -> 1 (bad), '-' -> 0 (good)
    target_map = TARGET_MAPPING["australian"]
    if 'credit_risk' in df.columns:
        df['credit_risk'] = df['credit_risk'].map(lambda x: target_map.get(x, 0))
    elif 'target' in df.columns:
        df = df.rename(columns={'target': 'credit_risk'})
        df['credit_risk'] = df['credit_risk'].map(lambda x: target_map.get(x, 0))
    
    # Add dataset source identifier
    df['dataset_source'] = 'australian'
    
    logger.info(f"Standardized Australian Credit schema. Target distribution: {df['credit_risk'].value_counts().to_dict()}")
    return df


def unify_datasets(
    german_df: pd.DataFrame,
    taiwanese_df: pd.DataFrame,
    australian_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge standardized datasets into a unified training set.
    Keeps only common columns across all datasets to ensure consistency.
    
    Args:
        german_df: Standardized German Credit DataFrame
        taiwanese_df: Standardized Taiwanese Credit DataFrame
        australian_df: Standardized Australian Credit DataFrame
    
    Returns:
        Unified DataFrame with common features and target
    """
    logger.info("Unifying datasets...")
    
    # Get common columns (excluding dataset_source)
    cols_german = set(german_df.columns)
    cols_taiwanese = set(taiwanese_df.columns)
    cols_australian = set(australian_df.columns)
    
    # Common columns should include 'credit_risk' and 'dataset_source'
    common_cols = cols_german & cols_taiwanese & cols_australian
    logger.info(f"Common columns across all datasets: {len(common_cols)}")
    logger.info(f"Common columns: {sorted(common_cols)}")
    
    # If very few common columns, include important ones even if not in all datasets
    if len(common_cols) < 10:
        logger.warning("Very few common columns. Including key features from each dataset...")
        # Keep credit_risk and dataset_source mandatory
        mandatory = {'credit_risk', 'dataset_source'}
        
        # Select top features from each
        german_df = german_df[list(common_cols | mandatory)]
        taiwanese_df = taiwanese_df[list(common_cols | mandatory)]
        australian_df = australian_df[list(common_cols | mandatory)]
        
        common_cols = common_cols | mandatory
    
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
    logger.info(f"Target distribution:\n{unified_df['credit_risk'].value_counts()}")
    logger.info(f"Dataset source distribution:\n{unified_df['dataset_source'].value_counts()}")
    
    return unified_df


def load_and_unify_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load and unify all datasets from UCI ML Repository.
    
    Returns:
        Tuple of (unified_df, metadata_dict)
    """
    logger.info("\n" + "="*80)
    logger.info("Starting data loading and unification from UCI ML Repository...")
    logger.info("="*80)
    
    try:
        # Load individual datasets
        german = load_german_credit()
        taiwanese = load_taiwanese_credit()
        australian = load_australian_credit()
        
        # Standardize schemas
        german = standardize_german_credit(german)
        taiwanese = standardize_taiwanese_credit(taiwanese)
        australian = standardize_australian_credit(australian)
        
        # Unify
        unified = unify_datasets(german, taiwanese, australian)
        
        # Metadata
        metadata = {
            "total_rows": len(unified),
            "total_columns": len(unified.columns),
            "target_distribution": unified["credit_risk"].value_counts().to_dict(),
            "class_distribution_pct": (unified["credit_risk"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "dataset_source_distribution": unified["dataset_source"].value_counts().to_dict(),
            "missing_values": unified.isnull().sum().to_dict(),
            "feature_dtypes": unified.dtypes.to_dict(),
        }
        
        logger.info("Data loading and unification complete!")
        logger.info(f"Metadata: {metadata}")
        
        return unified, metadata
        
    except Exception as e:
        logger.error(f"Error during data loading and unification: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
