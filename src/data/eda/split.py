# src/data/eda/split.py
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
    score_col: str = "mos",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test sets with stratified sampling.

    Args:
        df: DataFrame with MOS scores
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_state: Random seed
        score_col: Column name for MOS scores

    Returns:
        (train_df, val_df, test_df)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    logger.info(f"Split: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

    stratify_labels = create_stratified_labels(df, score_col=score_col)

    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_ratio, random_state=random_state, stratify=stratify_labels
    )

    # Second split: separate val from train_val
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_val_stratify = stratify_labels.iloc[train_val.index]

    train, val = train_test_split(
        train_val,
        test_size=relative_val_ratio,
        random_state=random_state,
        stratify=train_val_stratify,
    )

    logger.info(f"Split complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test


def create_stratified_labels(df: pd.DataFrame, bins: int = 10, score_col: str = "mos"):
    """
    Create stratified labels for train_test_split.

    Uses quantile-based binning, falls back to uniform binning if quantiles fail.
    """
    try:
        return pd.qcut(df[score_col], q=bins, labels=False, duplicates="drop")
    except Exception:
        logger.debug("Quantile binning failed, falling back to uniform binning")
        return pd.cut(df[score_col], bins=bins, labels=False)


def check_fold_distribution(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    score_col: str = "mos",
    verbose: bool = True,
) -> List[Dict]:
    """
    Check distribution of K-fold cross-validation splits.

    Returns:
        List of statistics per fold: fold number, train/val mean and std
    """
    if df is None or df.empty:
        return []

    stratify_labels = create_stratified_labels(df, score_col=score_col)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        train_mean = df.iloc[train_idx][score_col].mean()
        val_mean = df.iloc[val_idx][score_col].mean()
        train_std = df.iloc[train_idx][score_col].std()
        val_std = df.iloc[val_idx][score_col].std()

        fold_stats.append({
            "fold": fold + 1,
            "train_mean": train_mean,
            "val_mean": val_mean,
            "train_std": train_std,
            "val_std": val_std,
        })

    if verbose:
        logger.info(f"Stratified {n_splits}-Fold distribution checked")
        for stat in fold_stats:
            logger.debug(
                f"  Fold {stat['fold']}: Train Mean={stat['train_mean']:.3f}±{stat['train_std']:.3f}, "
                f"Val Mean={stat['val_mean']:.3f}±{stat['val_std']:.3f}"
            )

    return fold_stats