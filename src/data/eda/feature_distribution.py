"""Training-set feature distribution artifacts for IQA/VQA models.

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
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from deploy.core.model_loader import load_checkpoint
from deploy.core.preprocessor import Preprocessor
from src.core.trainer import ImageVideoDataset
from src.data.data_eda import DataEDA
from src.utils.file_loader import CaseInsensitiveAssetResolver


ARTIFACT_VERSION = 1


@dataclass
class FeatureDistributionArtifact:
    """Serializable PCA baseline fitted on one training feature distribution."""

    metadata: dict[str, Any]
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
    def load(cls, path: Path) -> "FeatureDistributionArtifact":
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"Feature distribution artifact not found: {path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Feature distribution metadata not found: {metadata_path}")

        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
                feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float32),
                feature_scale=np.asarray(arrays["feature_scale"], dtype=np.float32),
                pca_mean=np.asarray(arrays["pca_mean"], dtype=np.float32),
                pca_components=np.asarray(arrays["pca_components"], dtype=np.float32),
                train_projection=np.asarray(arrays["train_projection"], dtype=np.float32),
                train_sample_ids=[str(value) for value in arrays["train_sample_ids"].tolist()],
            )

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project raw quality embeddings using the saved training transform."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.feature_mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: artifact={self.feature_mean.shape[0]}, "
                f"input={features.shape[1]}"
            )
        standardized = (features - self.feature_mean) / self.feature_scale
        return (standardized - self.pca_mean) @ self.pca_components.T

    def score_projection(self, projection: np.ndarray) -> list[dict[str, Any]]:
        """Return nearest training samples and empirical distance percentile."""
        projection = np.asarray(projection, dtype=np.float32)
        if projection.ndim == 1:
            projection = projection[None, :]
        train = self.train_projection
        distances = np.sqrt(((projection[:, None, :] - train[None, :, :]) ** 2).sum(axis=2))
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(len(projection)), nearest_indices]
        train_nearest = np.sqrt(
            ((train[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
        )
        np.fill_diagonal(train_nearest, np.inf)
        baseline_distances = np.min(train_nearest, axis=1)
        results = []
        for index, distance in zip(nearest_indices, nearest_distances):
            percentile = float((baseline_distances <= distance).mean() * 100)
            results.append(
                {
                    "x": float(projection[len(results), 0]),
                    "y": float(projection[len(results), 1]),
                    "nearest_distance": float(distance),
                    "nearest_sample_id": self.train_sample_ids[int(index)],
                    "nearest_distance_percentile": percentile,
                }
            )
        return results


def fit_feature_distribution(
    features: np.ndarray,
    sample_ids: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> FeatureDistributionArtifact:
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
    artifact_metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dim": int(features.shape[1]),
        "sample_count": int(features.shape[0]),
        "projection_dim": 2,
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        **(metadata or {}),
    }
    return FeatureDistributionArtifact(
        metadata=artifact_metadata,
        feature_mean=scaler.mean_.astype(np.float32),
        feature_scale=np.maximum(scaler.scale_, 1e-8).astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        train_projection=projection,
        train_sample_ids=[str(value) for value in sample_ids],
    )


@torch.inference_mode()
def extract_quality_features(model: torch.nn.Module, data: torch.Tensor, device: str) -> np.ndarray:
    """Extract the model's pooled image or temporally fused video embedding."""
    model = model.to(device).eval()
    if not hasattr(model, "extract_quality_features"):
        raise AttributeError("Model does not expose extract_quality_features()")
    return model.extract_quality_features(data.to(device)).detach().cpu().numpy()


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
            ids = [str(value) for value in batch["sample_id"][:remaining]]
        else:
            ids = [str(value) for value in batch["sample_id"]]
        batches.append(extract_quality_features(model, data, device))
        sample_ids.extend(ids)
        seen += len(ids)
    if not batches:
        raise ValueError("No samples were available for feature extraction")
    return np.concatenate(batches, axis=0), sample_ids


def save_projection_plot(
    artifact: FeatureDistributionArtifact,
    path: Path,
    query_projection: np.ndarray | None = None,
    query_labels: Sequence[str] | None = None,
) -> None:
    """Save a compact PCA scatter plot for human inspection."""
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(artifact.train_projection[:, 0], artifact.train_projection[:, 1], s=8, alpha=0.35, label="train")
    if query_projection is not None:
        query_projection = np.asarray(query_projection)
        axis.scatter(query_projection[:, 0], query_projection[:, 1], c="crimson", s=42, label="query")
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


def _build_artifact(args: argparse.Namespace) -> None:
    model, checkpoint_cfg = load_checkpoint(Path(args.checkpoint), args.device)
    dataset_name = args.dataset or checkpoint_cfg.dataset.registry_key or checkpoint_cfg.dataset.name
    # Use the checkpoint's exact preprocessing/model configuration so the
    # saved distribution is compatible with later projection queries.
    data_cfg = checkpoint_cfg
    eda = DataEDA(data_cfg, dataset_name)
    dataframe = eda.load_metadata()
    if dataframe.empty:
        raise RuntimeError(f"No metadata found for dataset {dataset_name}")
    eda.df = dataframe
    eda.ensure_split_column()
    selected = eda.df[eda.df["split"] == args.split].reset_index(drop=True)
    if selected.empty:
        raise RuntimeError(f"No samples found in split '{args.split}'")

    data_dir = data_cfg.paths.resolve(Path(data_cfg.dataset.paths.root) / data_cfg.dataset.paths.data)
    resolver = CaseInsensitiveAssetResolver(data_dir)
    dataset = ImageVideoDataset(selected, data_dir, data_cfg, resolver)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=data_cfg.system.pin_memory,
    )
    features, sample_ids = extract_loader_features(model, loader, args.device, args.max_samples)
    output = Path(args.output) if args.output else (
        data_cfg.paths.feature_distribution_dir(dataset_name) / f"{args.split}_pca.npz"
    )
    artifact = fit_feature_distribution(
        features,
        sample_ids,
        metadata={
            "dataset": str(dataset_name),
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "feature_extractor": "IQAVQANet.extract_quality_features",
            "task_type": str(checkpoint_cfg.task_type),
        },
    )
    artifact.save(output)
    save_projection_plot(artifact, output.with_name(f"{output.stem}.png"))
    logger.info("Saved feature distribution artifact: {}", output)


def _project_inputs(args: argparse.Namespace) -> None:
    artifact = FeatureDistributionArtifact.load(Path(args.artifact))
    model, model_cfg = load_checkpoint(Path(args.checkpoint), args.device)
    preprocessor = Preprocessor(
        num_frames=model_cfg.model.num_frames,
        input_size=model_cfg.model.input_size,
    )
    inputs = _collect_inputs(args.input)
    features = []
    labels = []
    for path in inputs:
        if path.suffix.lower() in {suffix.lower() for suffix in cfg_image_exts(model_cfg)}:
            data = preprocessor.process_image(path).unsqueeze(0)
        else:
            data = preprocessor.process_video(path).unsqueeze(0)
        features.append(extract_quality_features(model, data, args.device)[0])
        labels.append(path.name)

    projection = artifact.transform(np.stack(features))
    rows = artifact.score_projection(projection)
    for label, row in zip(labels, rows):
        row["input"] = label
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_projection_plot(artifact, output.with_suffix(".png"), projection, labels)
    logger.info("Saved feature distribution projections: {}", output)


def cfg_image_exts(model_cfg: Any) -> set[str]:
    return {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_inputs(raw_inputs: Iterable[str]) -> list[Path]:
    extensions = cfg_image_exts(None) | {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    paths: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)
        elif path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
    return sorted(set(paths), key=lambda item: str(item).lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query IQA/VQA feature distribution artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="fit a training feature distribution")
    build.add_argument("--checkpoint", required=True)
    build.add_argument("--dataset", default=None)
    build.add_argument("--model", default="swin_vqa")
    build.add_argument("--split", default="train", choices=["train", "val", "test"])
    build.add_argument("--output", default=None, help="output .npz artifact path (defaults under results/{dataset}/eda)")
    build.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    build.add_argument("--batch-size", type=int, default=32)
    build.add_argument("--num-workers", type=int, default=0)
    build.add_argument("--max-samples", type=int, default=None)
    build.set_defaults(handler=_build_artifact)

    project = subparsers.add_parser("project", help="project new samples into a saved distribution")
    project.add_argument("--artifact", required=True)
    project.add_argument("--checkpoint", required=True)
    project.add_argument("--input", required=True, nargs="+")
    project.add_argument("--output", required=True, help="output JSON path")
    project.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    project.set_defaults(handler=_project_inputs)

    args = parser.parse_args()
    if args.command == "project" and not _collect_inputs(args.input):
        parser.error("No supported input files found")
    args.handler(args)


if __name__ == "__main__":
    main()
