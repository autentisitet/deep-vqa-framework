# src/data/eda/__init__.py
from .integrity import check_media_integrity, check_image_integrity, check_video_integrity
# Backward-compatible export. Plotting now belongs to src.visualization.
from src.visualization.training_plots import MetricsPlotter
from .split import (
    check_fold_distribution,
    create_stratified_labels,
    infer_group_labels,
    make_group_kfold_splits,
    split_train_val_test,
    validate_group_split_isolation,
)
from .statistics import (
    analyze_image_properties,
    analyze_video_properties,
    compute_mos_statistics,
)

_LAZY_IMPORTS = {
    "FeatureDistributionArtifact": ".feature_distribution",
    "fit_feature_distribution": ".feature_distribution",
}

__all__ = [
    "check_media_integrity",
    "check_image_integrity",
    "check_video_integrity",
    "MetricsPlotter",
    "split_train_val_test",
    "check_fold_distribution",
    "create_stratified_labels",
    "infer_group_labels",
    "make_group_kfold_splits",
    "validate_group_split_isolation",
    "analyze_image_properties",
    "analyze_video_properties",
    "compute_mos_statistics",
    "FeatureDistributionArtifact",
    "fit_feature_distribution",
]


def __getattr__(name):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    return getattr(module, name)
