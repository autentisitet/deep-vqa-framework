from pathlib import Path
import re

from loguru import logger


def checkpoint_score_from_name(path: Path, monitor: str) -> float | None:
    match = re.search(
        rf"{re.escape(monitor.lower())}(-?\d+(?:\.\d+)?)",
        path.name.lower(),
    )
    if match is None:
        return None
    return float(match.group(1))


def select_best_checkpoint(
    model_outputs_dir: Path,
    base_fn: str,
    monitor: str,
    mode: str,
    secondary_monitor: str = "val_plcc",
) -> Path:
    """Select the best checkpoint for one run across all folds."""
    mode = mode.lower()
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    if not model_outputs_dir.exists():
        raise FileNotFoundError(f"Model outputs directory does not exist: {model_outputs_dir}")

    prefix = f"{base_fn}_fold"
    best_epoch_paths = [
        path
        for path in model_outputs_dir.glob("*.pt")
        if path.name.startswith(prefix) and "_best_epoch" in path.name
    ]

    scored_paths: list[tuple[Path, float, float, float]] = []
    for path in best_epoch_paths:
        score = checkpoint_score_from_name(path, monitor)
        if score is not None:
            secondary_score = checkpoint_score_from_name(path, secondary_monitor)
            scored_paths.append(
                (
                    path,
                    score,
                    secondary_score if secondary_score is not None else float("-inf"),
                    path.stat().st_mtime,
                )
            )

    if scored_paths:
        if mode == "max":
            best_path, best_score, _, _ = max(scored_paths, key=lambda item: (item[1], item[2], item[3]))
        else:
            best_path, best_score, _, _ = min(scored_paths, key=lambda item: (item[1], -item[2], -item[3]))
        logger.info(f"Selected best checkpoint by {monitor}: {best_path.name} ({best_score:.4f})")
        return best_path

    fallback_paths = [
        path
        for path in model_outputs_dir.glob("*.pt")
        if path.name.startswith(prefix) and path.name.endswith("_best.pt")
    ]
    if fallback_paths:
        fallback_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        logger.warning(
            "No scored best checkpoint found for this run; "
            f"falling back to latest standard best file: {fallback_paths[0].name}"
        )
        return fallback_paths[0]

    raise FileNotFoundError(
        f"No best checkpoint found for current run prefix '{base_fn}' in {model_outputs_dir}"
    )
