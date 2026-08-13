# src/data/eda/split.py
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility for older sklearn
    StratifiedGroupKFold = None


GROUP_COLUMN_CANDIDATES = (
    "group_id",
    "reference_id",
    "ref_id",
    "source_id",
    "content_id",
    "original_id",
    "image_id",
    "video_id",
)


def _normalize_group_value(value) -> str:
    if pd.isna(value):
        return ""
    return Path(str(value).strip()).stem.lower()


def _infer_group_from_sample_id(sample_id, dataset_name: Optional[str] = None) -> str:
    stem = _normalize_group_value(sample_id)
    dataset_key = str(dataset_name or "").strip().lower()

    if dataset_key == "tid2013" or re.match(r"^i\d{1,3}_\d{2}_\d+$", stem):
        match = re.match(r"^(i\d{1,3})_\d{2}_\d+$", stem)
        if match:
            return match.group(1).lower()

    return stem


def infer_group_labels(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    sample_col: str = "sample_id",
) -> pd.Series:
    """Infer leakage-safe group labels for distorted-content datasets."""
    if sample_col not in df.columns:
        raise KeyError(f"Missing sample id column: {sample_col}")

    fallback_groups = df[sample_col].apply(
        lambda sample_id: _infer_group_from_sample_id(sample_id, dataset_name)
    )

    candidate_cols = []
    if group_col:
        candidate_cols.append(group_col)
    candidate_cols.extend(col for col in GROUP_COLUMN_CANDIDATES if col not in candidate_cols)

    for col in candidate_cols:
        if col not in df.columns:
            continue

        groups = df[col].apply(_normalize_group_value)
        if groups.ne("").any():
            return groups.where(groups.ne(""), fallback_groups)

    return fallback_groups


def find_group_split_overlaps(
    df: pd.DataFrame,
    split_col: str = "split",
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[Tuple[str, str], List[str]]:
    """Return group ids that appear in more than one named split."""
    if split_col not in df.columns or df.empty:
        return {}

    groups = infer_group_labels(df, group_col=group_col, dataset_name=dataset_name)
    split_groups = {
        str(split_name): set(groups[df[split_col] == split_name].dropna())
        for split_name in sorted(df[split_col].dropna().unique())
    }

    overlaps = {}
    split_names = list(split_groups)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1:]:
            shared = sorted(split_groups[left_name] & split_groups[right_name])
            if shared:
                overlaps[(left_name, right_name)] = shared

    return overlaps


def validate_group_split_isolation(
    df: pd.DataFrame,
    split_col: str = "split",
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    context: str = "split",
) -> bool:
    """Log and return whether each group is isolated to one split."""
    overlaps = find_group_split_overlaps(
        df=df,
        split_col=split_col,
        group_col=group_col,
        dataset_name=dataset_name,
    )
    if not overlaps:
        return True

    for (left_name, right_name), shared in overlaps.items():
        preview = ", ".join(shared[:10])
        suffix = "..." if len(shared) > 10 else ""
        logger.error(
            f"Group leakage in {context}: {left_name} vs {right_name}; "
            f"{len(shared)} shared groups: {preview}{suffix}"
        )

    return False


def _safe_stratified_labels(df: pd.DataFrame, score_col: str) -> pd.Series:
    labels = create_stratified_labels(df, score_col=score_col)
    return pd.Series(labels, index=df.index).astype("float").fillna(-1).astype(int)


def _group_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float,
    groups: pd.Series,
    stratify_labels: pd.Series,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Need at least two groups for a group-aware split")

    if StratifiedGroupKFold is not None:
        n_splits = max(2, round(1.0 / holdout_ratio))
        n_splits = min(n_splits, unique_groups)
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            candidates = []
            for train_idx, holdout_idx in splitter.split(df, stratify_labels, groups):
                ratio_gap = abs((len(holdout_idx) / len(df)) - holdout_ratio)
                candidates.append((ratio_gap, train_idx, holdout_idx))
            if candidates:
                _, train_idx, holdout_idx = min(candidates, key=lambda item: item[0])
                return train_idx, holdout_idx
        except Exception as e:
            logger.warning(f"StratifiedGroupKFold holdout split failed, using GroupShuffleSplit: {e}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_ratio,
        random_state=random_state,
    )
    return next(splitter.split(df, groups=groups))


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_state: int = 42,
    score_col: str = "mos",
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test sets with group-aware sampling.

    Args:
        df: DataFrame with MOS scores
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_state: Random seed
        score_col: Column name for MOS scores
        group_col: Optional explicit group column
        dataset_name: Dataset name used for dataset-specific group inference

    Returns:
        (train_df, val_df, test_df)
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    logger.info(f"Split: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

    stratify_labels = _safe_stratified_labels(df, score_col=score_col)
    groups = infer_group_labels(df, group_col=group_col, dataset_name=dataset_name)

    if groups.nunique() >= 3:
        train_val_idx, test_idx = _group_holdout_split(
            df=df,
            holdout_ratio=test_ratio,
            groups=groups,
            stratify_labels=stratify_labels,
            random_state=random_state,
        )

        train_val = df.iloc[train_val_idx]
        test = df.iloc[test_idx]

        train_val_groups = groups.iloc[train_val_idx]
        train_val_labels = stratify_labels.iloc[train_val_idx]
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)

        train_rel_idx, val_rel_idx = _group_holdout_split(
            df=train_val,
            holdout_ratio=relative_val_ratio,
            groups=train_val_groups,
            stratify_labels=train_val_labels,
            random_state=random_state + 1,
        )

        train = train_val.iloc[train_rel_idx]
        val = train_val.iloc[val_rel_idx]

        split_df = pd.concat(
            [
                train.assign(_split_check="train"),
                val.assign(_split_check="val"),
                test.assign(_split_check="test"),
            ]
        )
        if not validate_group_split_isolation(
            split_df,
            split_col="_split_check",
            group_col=group_col,
            dataset_name=dataset_name,
            context="train/val/test split",
        ):
            raise RuntimeError("Group-aware train/val/test split failed isolation validation")

        logger.info(
            "Group-aware split complete: "
            f"Train={len(train)} ({infer_group_labels(train, group_col, dataset_name).nunique()} groups), "
            f"Val={len(val)} ({infer_group_labels(val, group_col, dataset_name).nunique()} groups), "
            f"Test={len(test)} ({infer_group_labels(test, group_col, dataset_name).nunique()} groups)"
        )
        return train, val, test

    raise ValueError(
        "Need at least three distinct groups for leakage-safe train/val/test split; "
        f"found {groups.nunique()}."
    )


def create_stratified_labels(df: pd.DataFrame, bins: int = 10, score_col: str = "mos"):
    """
    Create score-bin labels for stratified splitters.

    Uses quantile-based binning, falls back to uniform binning if quantiles fail.
    """
    try:
        return pd.qcut(df[score_col], q=bins, labels=False, duplicates="drop")
    except Exception:
        logger.debug("Quantile binning failed, falling back to uniform binning")
        return pd.cut(df[score_col], bins=bins, labels=False)


def make_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    score_col: str = "mos",
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """Create leakage-safe K-fold splits, preferring stratified group folds."""
    if df is None or df.empty:
        return [], "empty"

    requested_splits = int(n_splits)
    if requested_splits < 2:
        raise ValueError(f"k-fold cross-validation requires at least 2 folds; got {n_splits}")

    groups = infer_group_labels(df, group_col=group_col, dataset_name=dataset_name)
    unique_groups = groups.nunique()

    if unique_groups >= 2:
        actual_splits = min(requested_splits, unique_groups)
        if actual_splits < requested_splits:
            logger.warning(
                f"Requested {requested_splits} folds but only {unique_groups} groups exist; "
                f"using {actual_splits} folds."
            )

        labels = _safe_stratified_labels(df, score_col=score_col)
        if StratifiedGroupKFold is not None:
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=actual_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                return list(splitter.split(df, labels, groups)), "StratifiedGroupKFold"
            except Exception as e:
                logger.warning(f"StratifiedGroupKFold failed, using GroupKFold: {e}")

        splitter = GroupKFold(n_splits=actual_splits)
        return list(splitter.split(df, groups=groups)), "GroupKFold"

    raise ValueError(
        "Need at least two distinct groups for leakage-safe cross-validation; "
        f"found {unique_groups}."
    )


def check_fold_distribution(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    score_col: str = "mos",
    group_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Check distribution of K-fold cross-validation splits.

    Returns:
        List of statistics per fold: fold number, train/val mean and std
    """
    if df is None or df.empty:
        return []

    splits, strategy = make_group_kfold_splits(
        df=df,
        n_splits=n_splits,
        random_state=random_state,
        score_col=score_col,
        group_col=group_col,
        dataset_name=dataset_name,
    )
    groups = infer_group_labels(df, group_col=group_col, dataset_name=dataset_name)

    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        if overlap:
            preview = ", ".join(sorted(overlap)[:10])
            raise RuntimeError(f"Group leakage detected in fold {fold + 1}: {preview}")

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
            "train_groups": groups.iloc[train_idx].nunique(),
            "val_groups": groups.iloc[val_idx].nunique(),
        })

    if verbose:
        logger.info(f"{strategy} {len(splits)}-Fold distribution checked")
        for stat in fold_stats:
            logger.debug(
                f"  Fold {stat['fold']}: Train Mean={stat['train_mean']:.3f}±{stat['train_std']:.3f}, "
                f"Val Mean={stat['val_mean']:.3f}±{stat['val_std']:.3f}, "
                f"Groups {stat['train_groups']}/{stat['val_groups']}"
            )

    return fold_stats
