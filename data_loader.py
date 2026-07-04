"""
Data loading and unification module for the Credit Risk Pipeline.

Loads German Credit, Taiwanese Credit, and Australian Credit datasets,
standardizes schemas, and merges them into a unified training set.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from config import (
    GERMAN_CREDIT_PATH,
    TAIWANESE_CREDIT_PATH,
    AUSTRALIAN_CREDIT_PATH,
    DATASET_COLUMN_MAPPING,
    TARGET_MAPPING,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


def load_german_credit() -> pd.DataFrame:
    """
    Load German Credit dataset from UCI ML Repository.
    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    """
    logger.info("Loading German Credit dataset...")
    try:
        df = pd.read_csv(GERMAN_CREDIT_PATH)
        logger.info(f"German Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"German Credit dataset not found at {GERMAN_CREDIT_PATH}")
        raise


def load_taiwanese_credit() -> pd.DataFrame:
    """
    Load Taiwanese Credit dataset from UCI ML Repository.
    Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    logger.info("Loading Taiwanese Credit dataset...")
    try:
        df = pd.read_csv(TAIWANESE_CREDIT_PATH)
        logger.info(f"Taiwanese Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Taiwanese Credit dataset not found at {TAIWANESE_CREDIT_PATH}")
        raise


def load_australian_credit() -> pd.DataFrame:
    """
    Load Australian Credit dataset from UCI ML Repository.
    Source: https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
    """
    logger.info("Loading Australian Credit dataset...")
    try:
        df = pd.read_csv(AUSTRALIAN_CREDIT_PATH)
        logger.info(f"Australian Credit: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Australian Credit dataset not found at {AUSTRALIAN_CREDIT_PATH}")
        raise


def standardize_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize German Credit dataset schema.
    """
    logger.info("Standardizing German Credit schema...")
    df = df.rename(columns=DATASET_COLUMN_MAPPING["german"])
    
    # Map target variable
    target_map = TARGET_MAPPING["german"]
    df["credit_risk"] = df["credit_risk"].map(target_map)
    
    # Add dataset source identifier
    df["dataset_source"] = "german"
    
    return df


def standardize_taiwanese_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Taiwanese Credit dataset schema.
    Note: Taiwanese dataset uses multiple repayment history columns; 
    for simplicity, we extract a subset of key features.
    """
    logger.info("Standardizing Taiwanese Credit schema...")
    
    # Map columns
    df = df.rename(columns=DATASET_COLUMN_MAPPING["taiwanese"])
    
    # Extract key features from repayment history (PAY_1 to PAY_6)
    # Aggregate repayment status
    pay_cols = [col for col in df.columns if col.startswith("PAY_") and col[-1].isdigit()]
    if pay_cols:
        df["avg_repayment_status"] = df[pay_cols].mean(axis=1)
        df["max_repayment_status"] = df[pay_cols].max(axis=1)
        df = df.drop(columns=pay_cols, errors="ignore")
    
    # Map target variable
    target_map = TARGET_MAPPING["taiwanese"]
    df["credit_risk"] = df["credit_risk"].map(target_map)
    
    # Add dataset source identifier
    df["dataset_source"] = "taiwanese"
    
    return df


def standardize_australian_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Australian Credit dataset schema.
    """
    logger.info("Standardizing Australian Credit schema...")
    df = df.rename(columns=DATASET_COLUMN_MAPPING["australian"])
    
    # Map target variable
    target_map = TARGET_MAPPING["australian"]
    df["credit_risk"] = df["credit_risk"].map(target_map)
    
    # Add dataset source identifier
    df["dataset_source"] = "australian"
    
    return df


def unify_datasets(
    german_df: pd.DataFrame,
    taiwanese_df: pd.DataFrame,
    australian_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge standardized datasets into a unified training set.
    Keeps only common columns across all datasets to ensure consistency.
    """
    logger.info("Unifying datasets...")
    
    # Get common columns
    common_cols = set(german_df.columns) & set(taiwanese_df.columns) & set(australian_df.columns)
    logger.info(f"Common columns across datasets: {len(common_cols)}")
    
    # Select and concatenate
    unified_df = pd.concat(
        [
            german_df[common_cols],
            taiwanese_df[common_cols],
            australian_df[common_cols]
        ],
        axis=0,
        ignore_index=True
    )
    
    logger.info(f"Unified dataset shape: {unified_df.shape}")
    logger.info(f"Target distribution:\n{unified_df['credit_risk'].value_counts()}")
    
    return unified_df


def load_and_unify_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load and unify all datasets.
    
    Returns:
        Tuple of (unified_df, metadata_dict)
    """
    logger.info("Starting data loading and unification...")
    
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
        "missing_values": unified.isnull().sum().to_dict(),
    }
    
    logger.info("Data loading and unification complete!")
    return unified, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, meta = load_and_unify_data()
    print(df.head())
    print("\nMetadata:")
    for key, val in meta.items():
        print(f"{key}: {val}")
