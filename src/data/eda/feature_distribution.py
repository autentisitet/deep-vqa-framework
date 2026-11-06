"""Training-set feature distribution files for IQA/VQA models.

The workflow is deliberately separate from browser inference:

1. extract the trained model's quality embedding for the training split;
2. standardize and fit a two-dimensional PCA once;
3. save the embedding metadata and training projection;
4. project later samples with the saved parameters and report nearest-distance
   context in the same space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src import __version__
from src.core.trainer import ImageVideoDataset
from src.data.data_eda import DataEDA
from src.data.preprocessing import Preprocessor
from src.utils.checkpointing import load_checkpoint
from src.utils.file_loader import CaseInsensitiveAssetResolver

FILE_FORMAT_VERSION = 1


@dataclass
class FeatureDistributionFile:
    """Serializable PCA baseline fitted on one training feature distribution."""

    metadata: dict[str, object]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    train_projection: np.ndarray
    train_sample_ids: list[str]

    def save(self, path: Path) -> None:
        """Save numeric arrays in NPZ and human-readable metadata beside it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            train_projection=self.train_projection,
            train_sample_ids=np.asarray(self.train_sample_ids, dtype=str),
        )
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureDistributionFile":
        """Load a feature distribution from its NPZ and JSON metadata."""
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"Feature distribution file not found: {path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Feature distribution metadata not found: {metadata_path}")

        metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise ValueError("Feature distribution metadata must be a JSON object")
        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                metadata={key: value for key, value in metadata.items()},
                feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(arrays["feature_scale"], dtype=np.float32),
                pca_mean=np.asarray(arrays["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(arrays["pca_components"], dtype=np.float32),
                train_projection=np.asarray(arrays["train_projection"], dtype=np.float32),
                train_sample_ids=list(map(str, arrays["train_sample_ids"].tolist())),
            )

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project raw quality embeddings using the saved training transform."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.feature_mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: saved={self.feature_mean.shape[0]}, input={features.shape[1]}"
            )
        standardized = (features - self.feature_mean) / self.feature_scale
        projected: np.ndarray = (standardized - self.pca_mean) @ self.pca_components.T
        return projected

    def score_projection(self, projection: np.ndarray) -> list[dict[str, object]]:
        """Return nearest training samples and empirical distance percentile."""
        projection = np.asarray(projection, dtype=np.float32)
        if projection.ndim == 1:
            projection = projection[None, :]
        train = self.train_projection
        distances = np.sqrt(((projection[:, None, :] - train[None, :, :]) ** 2).sum(axis=2))
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(projection)), nearest_indices]
        train_nearest = np.sqrt(((train[:, None, :] - train[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(train_nearest, np.inf)
        baseline_distances = np.min(train_nearest, axis=1)
        results: list[dict[str, object]] = []
        for point, index, distance in zip(projection, nearest_indices, nearest_distances):
            percentile = float((baseline_distances <= distance).mean() * 100)
            results.append(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "nearest_distance": float(distance),
                    "nearest_sample_id": self.train_sample_ids[int(index)],
                    "nearest_distance_percentile": percentile,
                }
            )
        return results


def fit_feature_distribution(
    features: np.ndarray,
    sample_ids: Sequence[str],
    metadata: dict[str, object] | None = None,
) -> FeatureDistributionFile:
    """Fit standardization and two-component PCA on training embeddings."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError(f"Expected at least 2 samples and 2 features, got {features.shape}")
    if len(sample_ids) != features.shape[0]:
        raise ValueError("sample_ids length must match the number of feature rows")

    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features).astype(np.float32)
    pca = PCA(n_components=2).fit(standardized)
    projection = pca.transform(standardized).astype(np.float32)
    file_metadata = {
        "format_version": FILE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dim": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
        "projection_dim": 2,
        "explained_variance_ratio": list(map(float, pca.explained_variance_ratio_)),
        **(metadata or {}),
    }
    return FeatureDistributionFile(
        metadata=file_metadata,
        feature_mean=scaler.mean_.astype(np.float32),
        feature_scale=np.maximum(scaler.scale_, 1e-8).astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        train_projection=projection,
        train_sample_ids=list(map(str, sample_ids)),
    )


@torch.inference_mode()
def extract_quality_features(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: str,
) -> np.ndarray:
    """Extract the model's pooled image or temporally fused video embedding."""
    model = model.to(device).eval()
    extractor = getattr(model, "extract_quality_features", None)
    if extractor is None:
        raise AttributeError("Model does not expose extract_quality_features()")
    features: np.ndarray = extractor(data.to(device)).detach().cpu().numpy()
    return features


def extract_loader_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract embeddings and IDs from a deterministic DataLoader."""
    batches: list[np.ndarray] = []
    sample_ids: list[str] = []
    seen = 0
    for batch in loader:
        data = batch["data"]
        if max_samples is not None:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            data = data[:remaining]
            ids = list(map(str, batch["sample_id"][:remaining]))
        else:
            ids = list(map(str, batch["sample_id"]))
        batches.append(extract_quality_features(model, data, device))
        sample_ids.extend(ids)
        seen += len(ids)
    if not batches:
        raise ValueError("No samples were available for feature extraction")
    return np.concatenate(batches, axis=0), sample_ids


def save_projection_plot(
    distribution_file: FeatureDistributionFile,
    path: Path,
    query_projection: np.ndarray | None = None,
    query_labels: Sequence[str] | None = None,
) -> None:
    """Save a compact PCA scatter plot for human inspection."""
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(
        distribution_file.train_projection[:, 0],
        distribution_file.train_projection[:, 1],
        s=8,
        alpha=0.35,
        label="train",
    )
    if query_projection is not None:
        query_projection = np.asarray(query_projection)
        axis.scatter(
            query_projection[:, 0],
            query_projection[:, 1],
            c="crimson",
            s=42,
            label="query",
        )
        if query_labels:
            for point, label in zip(query_projection, query_labels):
                axis.annotate(str(label), (point[0], point[1]), fontsize=8, alpha=0.8)
    axis.set_xlabel("PCA-1")
    axis.set_ylabel("PCA-2")
    axis.set_title("Training feature distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


@dataclass(frozen=True)
class BuildDistributionOptions:
    """Inputs needed to fit a training feature distribution."""

    checkpoint: Path
    device: str
    dataset: str | None = None
    split: str = "train"
    output: Path | None = None
    batch_size: int = 32
    num_workers: int = 0
    max_samples: int | None = None


@dataclass(frozen=True)
class ProjectDistributionOptions:
    """Inputs needed to project media into a saved distribution."""

    checkpoint: Path
    device: str
    distribution_file: Path
    input: tuple[Path, ...]
    output: Path


def _build_distribution_file(options: BuildDistributionOptions) -> None:
    """Fit and save the training feature distribution for one split."""
    model, checkpoint_cfg = load_checkpoint(Path(options.checkpoint), options.device)
    dataset_name = options.dataset or checkpoint_cfg.dataset.registry_key or checkpoint_cfg.dataset.name
    # Use the checkpoint's exact preprocessing/model configuration so the
    # saved distribution is compatible with later projection queries.
    data_cfg = checkpoint_cfg
    eda = DataEDA(data_cfg, dataset_name)
    dataframe = eda.load_metadata()
    if dataframe.empty:
        raise RuntimeError(f"No metadata found for dataset {dataset_name}")
    eda.df = dataframe
    eda.ensure_split_column()
    selected = eda.df[eda.df["split"] == options.split].reset_index(drop=True)
    if selected.empty:
        raise RuntimeError(f"No samples found in split '{options.split}'")

    data_dir = data_cfg.paths.resolve_path(Path(data_cfg.dataset.paths.root) / data_cfg.dataset.paths.data)
    resolver = CaseInsensitiveAssetResolver(data_dir)
    dataset = ImageVideoDataset(selected, data_dir, data_cfg, resolver)
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=data_cfg.system.pin_memory,
    )
    features, sample_ids = extract_loader_features(model, loader, options.device, options.max_samples)
    output = (
        Path(options.output)
        if options.output
        else (data_cfg.paths.feature_distribution_dir(dataset_name) / f"{options.split}_pca.npz")
    )
    distribution_file = fit_feature_distribution(
        features,
        sample_ids,
        metadata={
            "dataset": str(dataset_name),
            "split": options.split,
            "checkpoint": str(Path(options.checkpoint).resolve()),
            "feature_extractor": "IQAVQANet.extract_quality_features",
            "task_type": str(checkpoint_cfg.task_type),
        },
    )
    distribution_file.save(output)
    save_projection_plot(distribution_file, output.with_name(f"{output.stem}.png"))
    logger.info("Saved feature distribution file: {}", output)


def _project_inputs(options: ProjectDistributionOptions) -> None:
    """Project new media into a saved distribution and write JSON and plot output."""
    distribution_file = FeatureDistributionFile.load(Path(options.distribution_file))
    model, model_cfg = load_checkpoint(Path(options.checkpoint), options.device)
    preprocessor = Preprocessor(
        num_frames=model_cfg.model.num_frames,
        input_size=model_cfg.model.input_size,
    )
    inputs = _collect_inputs(options.input)
    features = []
    labels = []
    for path in inputs:
        if path.suffix.lower() in {suffix.lower() for suffix in cfg_image_exts(model_cfg)}:
            data = preprocessor.process_image(path).unsqueeze(0)
        else:
            data = preprocessor.process_video(path).unsqueeze(0)
        features.append(extract_quality_features(model, data, options.device)[0])
        labels.append(path.name)

    projection = distribution_file.transform(np.stack(features))
    rows = distribution_file.score_projection(projection)
    for label, row in zip(labels, rows):
        row["input"] = label
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_projection_plot(distribution_file, output.with_suffix(".png"), projection, labels)
    logger.info("Saved feature distribution projections: {}", output)


def cfg_image_exts(model_cfg: object) -> set[str]:
    """Return the stable image extensions accepted for projection inputs."""
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_inputs(raw_inputs: Iterable[str | Path]) -> list[Path]:
    """Collect supported media files from the given paths."""
    extensions = cfg_image_exts(None) | {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    paths: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)
        elif path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
    return sorted(set(paths), key=lambda item: str(item).lower())


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without executing the command."""
    parser = argparse.ArgumentParser(
        description="Build and query IQA/VQA feature distribution files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python -m src.data.eda.feature_distribution build --checkpoint model.pt\n  python -m src.data.eda.feature_distribution project --checkpoint model.pt --distribution-file train.npz --input image.jpg --output projection.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    common_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. cpu or cuda:0"
    )

    build = subparsers.add_parser("build", parents=[common_parser], help="fit a training feature distribution")
    build.add_argument("--dataset", default=None)
    build.add_argument("--model", default="swin_vqa")
    build.add_argument("--split", default="train", choices=["train", "val", "test"])
    build.add_argument(
        "--output", type=Path, default=None, help="output .npz path (defaults under results/{dataset}/eda)"
    )
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--num-workers", type=int, default=0)
    build.add_argument("--max-samples", type=int, default=None)

    project = subparsers.add_parser(
        "project", parents=[common_parser], help="project new samples into a saved distribution"
    )
    project.add_argument("--distribution-file", type=Path, required=True)
    project.add_argument("--input", type=Path, required=True, nargs="+")
    project.add_argument("--output", type=Path, required=True, help="output JSON path")

    parser.add_argument("--version", action="version", version=f"Deep-VQA {__version__}")
    return parser


def parse_args(argv: list[str] | None = None) -> BuildDistributionOptions | ProjectDistributionOptions:
    """Produce validated options from argv (or the process command line)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "project" and not _collect_inputs(args.input):
        parser.error("No supported input files found")
    if args.command == "build":
        if args.batch_size <= 0 or args.num_workers < 0 or (args.max_samples is not None and args.max_samples <= 0):
            parser.error("batch size/max samples must be positive; num workers must be non-negative")
        return BuildDistributionOptions(
            checkpoint=args.checkpoint,
            device=args.device,
            dataset=args.dataset,
            split=args.split,
            output=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
    return ProjectDistributionOptions(
        checkpoint=args.checkpoint,
        device=args.device,
        distribution_file=args.distribution_file,
        input=tuple(args.input),
        output=args.output,
    )


def run(options: BuildDistributionOptions | ProjectDistributionOptions) -> None:
    """Execute a distribution operation without depending on argparse."""
    if isinstance(options, BuildDistributionOptions):
        _build_distribution_file(options)
    else:
        _project_inputs(options)


def main(argv: list[str] | None = None) -> None:
    """Parse command-line options and execute the command."""
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
