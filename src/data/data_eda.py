# src/data/data_eda.py
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.data.eda.integrity import check_media_integrity
from src.data.eda.split import (
    check_fold_distribution,
    infer_group_labels,
    split_train_val_test,
    validate_group_split_isolation,
)
from src.data.eda.statistics import analyze_image_properties, analyze_video_properties, compute_mos_statistics
from src.data.dataset_loaders import MetadataLoaderFactory
from src.data.dataset_types import DatasetType
from src.config.schemas import Config
from src.utils.file_loader import CaseInsensitiveAssetResolver


class DataEDA:
    """
    Data exploration and analysis: loading metadata, cleaning samples,
    statistical analysis, and dataset partitioning.
    """

    def __init__(
        self,
        cfg: Config,
        dataset_name: str,
        data_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.dataset_name = cfg.dataset.registry_key or MetadataLoaderFactory.normalize_key(dataset_name)
        self.dataset_slug = self.cfg.paths.dataset_slug(self.dataset_name)
        self.df = None
        self.stats = {}

        self.dataset_info = cfg.dataset
        self.is_video = self.dataset_info.data_type == "video"

        self.file_extensions = list(DatasetType.all_extensions())

        if data_dir is None:
            data_dir = cfg.paths.resolve(Path(cfg.dataset.paths.root) / cfg.dataset.paths.data)
        self.data_dir = Path(data_dir).resolve()

        self.corrupted_dir = cfg.paths.corrupted_dir(self.dataset_slug)
        self.results_dir = cfg.paths.dataset_results_dir(self.dataset_slug)
        self.eda_dir = cfg.paths.eda_dir(self.dataset_slug)
        self.rejected_labels_dir = self.corrupted_dir / "labels"

        logger.info(f"Data directory: {self.data_dir}")
        plt.switch_backend("Agg")

        self.resolver = CaseInsensitiveAssetResolver(target_dir=self.data_dir)


    def load_metadata(self) -> pd.DataFrame:
        """Load metadata file using the appropriate loader."""
        meta_file = (
            self.cfg.paths.resolve(Path(self.dataset_info.paths.root))
            / self.dataset_info.paths.metadata
            / (self.dataset_info.metadata.mos_file or MetadataLoaderFactory.get_metadata_file(self.dataset_name))
        )

        if not meta_file.exists():
            logger.error(f"Metadata file not found: {meta_file}")
            return pd.DataFrame()

        try:
            loader = MetadataLoaderFactory.get_loader(self.dataset_name)
            df = loader.load(meta_file)
            self.df = df
            logger.info(f"Loaded {len(df)} samples from metadata")
            return df
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
            return pd.DataFrame()


    def ensure_split_column(self):
        """Ensure DataFrame has a 'split' column for train/val/test."""
        if "split" in self.df.columns and not self.df["split"].isnull().all():
            logger.info("Split column already exists, skipping")
            if "group_id" not in self.df.columns:
                self.df["group_id"] = infer_group_labels(self.df, dataset_name=self.dataset_name)
            if not validate_group_split_isolation(
                self.df,
                dataset_name=self.dataset_name,
                context="existing train/val/test split",
            ):
                raise RuntimeError("Group leakage detected in existing split column")
            return

        train_ratio = 0.8
        val_ratio = 0.1

        logger.info(f"Splitting: Train={train_ratio}, Val={val_ratio}")

        train_df, val_df, test_df = split_train_val_test(
            self.df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=42,
            dataset_name=self.dataset_name,
        )

        self.df["group_id"] = infer_group_labels(self.df, dataset_name=self.dataset_name)
        self.df["split"] = "train"
        self.df.loc[val_df.index, "split"] = "val"
        self.df.loc[test_df.index, "split"] = "test"

        if not validate_group_split_isolation(
            self.df,
            dataset_name=self.dataset_name,
            context="EDA train/val/test split",
        ):
            raise RuntimeError("Group leakage detected after EDA split assignment")

        logger.info(f"Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


    def basic_statistics(self):
        """Compute and log basic statistics."""
        if self.df is None or self.df.empty:
            logger.error("No data loaded. Run load_metadata() first.")
            return

        mos_stats = compute_mos_statistics(self.df)
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Dataset: {self.dataset_name} Basic Statistics")
        logger.info(f"{'=' * 50}")
        logger.info(f"  Total samples: {mos_stats['total_samples']}")
        logger.info(f"  MOS Range: [{mos_stats['range'][0]:.3f}, {mos_stats['range'][1]:.3f}]")
        logger.info(f"  MOS Mean: {mos_stats['mean']:.3f} | MOS Std: {mos_stats['std']:.3f}")

        self._analyze_media_properties()
        self.stats["total_samples"] = mos_stats["total_samples"]
        self.stats["range"] = mos_stats["range"]


    def _backup_rejected_labels(self, rejected_df: pd.DataFrame, reason_col: str = "reject_reason") -> None:
        """Persist labels for samples removed before analysis/training."""
        if rejected_df.empty:
            return

        self.rejected_labels_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.rejected_labels_dir / f"{self.dataset_slug}_rejected_labels_{timestamp}.csv"
        jsonl_path = self.rejected_labels_dir / f"{self.dataset_slug}_rejected_labels_{timestamp}.jsonl"

        rejected_df.to_csv(csv_path, index=False)
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for record in rejected_df.to_dict(orient="records"):
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        latest_csv_path = self.rejected_labels_dir / f"{self.dataset_slug}_rejected_labels_latest.csv"
        rejected_df.to_csv(latest_csv_path, index=False)

        reason_counts = rejected_df[reason_col].value_counts().to_dict() if reason_col in rejected_df else {}
        logger.warning(
            f"Backed up {len(rejected_df)} rejected labels: {csv_path} | reasons={reason_counts}"
        )


    def _analyze_media_properties(self):
        """Analyze media properties from disk."""
        media_paths = []
        for ext in self.file_extensions:
            ext = ext.lstrip(".")
            media_paths.extend(self.data_dir.rglob(f"*.{ext.lower()}"))
            media_paths.extend(self.data_dir.rglob(f"*.{ext.upper()}"))

        if not media_paths:
            logger.warning(f"No media files found in {self.data_dir}")
            return

        try:
            first_file = Path(media_paths[0]).name
            asset = self.resolver.resolve(first_file)
            dataset_type = asset.dataset_type
        except Exception:
            dataset_type = DatasetType.VIDEO if self.is_video else DatasetType.IMAGE

        if dataset_type == DatasetType.IMAGE:
            props = analyze_image_properties(media_paths)
            if props:
                logger.info("  Image Properties:")
                logger.info(f"    Files: {props['total_files']}")
                logger.info(
                    f"    Resolution: {props['width']['min']}x{props['height']['min']} ~ "
                    f"{props['width']['max']}x{props['height']['max']}"
                )
        elif dataset_type == DatasetType.VIDEO:
            props = analyze_video_properties(media_paths)
            if props and "error" not in props:
                logger.info("  Video Properties:")
                logger.info(f"    Files: {props.get('total_files', 0)}")
                w = props.get("resolution", {}).get("width", {}).get("mean", 0)
                h = props.get("resolution", {}).get("height", {}).get("mean", 0)
                fps = props.get("fps", {}).get("mean", 0)
                logger.info(f"    Size: {int(w)}x{int(h)} | FPS: {fps:.2f}")


    def check_integrity(self, max_samples: Optional[int] = None, skip_video_check: bool = False) -> Dict:
        """Check file integrity: missing, corrupted, repeated sample ids."""
        logger.info("Integrity check started...")

        df_to_check = self.df if max_samples is None else self.df.head(max_samples)

        missing = []
        corrupted = []
        valid_indices = []
        rejected_records = []

        self.corrupted_dir.mkdir(parents=True, exist_ok=True)

        for i, row in tqdm(df_to_check.iterrows(), total=len(df_to_check), desc="Checking files"):
            sid = str(row["sample_id"])

            try:
                asset = self.resolver.resolve(sid)
                target_path = asset.path
                is_video = asset.is_video
            except Exception:
                missing.append(sid)
                rejected_records.append({
                    **row.to_dict(),
                    "reject_reason": "missing",
                    "reject_error": "Unable to resolve media file",
                    "original_path": "",
                    "quarantine_path": "",
                })
                continue

            if not target_path.exists():
                missing.append(sid)
                rejected_records.append({
                    **row.to_dict(),
                    "reject_reason": "missing",
                    "reject_error": "Resolved media file does not exist",
                    "original_path": str(target_path),
                    "quarantine_path": "",
                })
                continue

            if skip_video_check and is_video:
                valid_indices.append(i)
                continue

            dataset_type = DatasetType.VIDEO if is_video else DatasetType.IMAGE
            is_ok, error, _ = check_media_integrity(target_path, dataset_type)

            if not is_ok:
                corrupted.append(sid)
                quarantine_path = ""
                if target_path.exists():
                    dest = self.corrupted_dir / target_path.name
                    if dest.exists():
                        dest = self.corrupted_dir / f"{target_path.stem}_corrupted{target_path.suffix}"
                    shutil.move(str(target_path), str(dest))
                    quarantine_path = str(dest)
                    logger.warning(f"Corrupted file quarantined: {target_path.name} ({error}) -> {dest}")
                rejected_records.append({
                    **row.to_dict(),
                    "reject_reason": "corrupted",
                    "reject_error": error or "Integrity check failed",
                    "original_path": str(target_path),
                    "quarantine_path": quarantine_path,
                })
                continue

            valid_indices.append(i)

        self.df = self.df.loc[valid_indices].reset_index(drop=True)

        sample_id_series = self.df["sample_id"].astype(str).str.strip().str.lower()
        repeated_sample_ids = (
            sample_id_series[sample_id_series.duplicated(keep=False)]
            .drop_duplicates()
            .tolist()
        )
        if repeated_sample_ids:
            preview = ", ".join(repeated_sample_ids[:10])
            suffix = "..." if len(repeated_sample_ids) > 10 else ""
            logger.warning(
                f"Detected {len(repeated_sample_ids)} repeated sample_id values; "
                f"kept for IQA/VQA semantics: {preview}{suffix}"
            )

        self._write_integrity_report(missing, corrupted, repeated_sample_ids)
        if rejected_records:
            self._backup_rejected_labels(pd.DataFrame(rejected_records))

        logger.info(
            f"Integrity complete: Retained {len(self.df)} | "
            f"Missing {len(missing)} | Corrupted {len(corrupted)} | Repeated sample_ids {len(repeated_sample_ids)}"
        )
        return {
            "corrupted": corrupted,
            "missing": missing,
            "repeated_sample_ids": repeated_sample_ids,
        }


    def _write_integrity_report(self, missing: List[str], corrupted: List[str], repeated_sample_ids: List[str]):
        """Write integrity report to file."""
        report_dir = self.cfg.paths.train_logs_dir(self.dataset_slug)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{self.dataset_name}_integrity_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- Integrity Audit Report: {self.dataset_name} ---\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write(
                f"Summary: {len(missing)} missing, "
                f"{len(corrupted)} corrupted, {len(repeated_sample_ids)} repeated sample ids.\n\n"
            )
            if missing:
                f.write("Missing Files:\n" + "\n".join(missing) + "\n\n")
            if corrupted:
                f.write("Corrupted Files:\n" + "\n".join(corrupted) + "\n\n")
            if repeated_sample_ids:
                f.write("Repeated Sample IDs (kept):\n" + "\n".join(repeated_sample_ids) + "\n")


    def check_filename_label_match(self) -> bool:
        """Check if filenames on disk match labels in metadata."""
        if self.df is None:
            return False

        physical_names = set()
        for ext in self.file_extensions:
            ext = ext.lstrip(".")
            physical_names.update({p.name.lower() for p in self.data_dir.rglob(f"*.{ext.lower()}")})
            physical_names.update({p.name.lower() for p in self.data_dir.rglob(f"*.{ext.upper()}")})

        label_names = {str(name).strip().lower() for name in self.df["sample_id"].astype(str)}

        if len(label_names) > 0 and "." not in list(label_names)[0]:
            physical_names = {Path(name).stem for name in physical_names}
            label_names = {Path(name).stem for name in label_names}

        match = physical_names == label_names
        logger.info(f"Filename-label match: {match}")
        if not match:
            logger.warning(f"Excess files on disk: {len(physical_names - label_names)}")
            logger.warning(f"Missing from labels: {len(label_names - physical_names)}")
        return match


    def visualize_mos_distribution(self, save_dir: Optional[Path] = None):
        """Plot MOS distribution."""
        if self.df is None or self.df.empty:
            return

        if save_dir is None:
            save_dir = self.eda_dir

        save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].hist(self.df["mos"], bins=25, edgecolor="black", alpha=0.75, color="steelblue")
        axes[0].set_xlabel("MOS Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"{self.dataset_name} - MOS Distribution", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].boxplot(
            self.df["mos"],
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="crimson", linewidth=1.5),
        )
        axes[1].set_ylabel("MOS Range")
        axes[1].set_title(f"{self.dataset_name} - Boxplot", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        output_path = save_dir / f"{self.dataset_name}_mos_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"MOS distribution saved: {output_path}")


    def normalize_scores(self) -> pd.DataFrame:
        """Normalize MOS scores to [0, 1]."""
        if "split" in self.df.columns:
            train_df = self.df[self.df["split"] == "train"]
            min_val = train_df["mos"].min()
            max_val = train_df["mos"].max()
        else:
            min_val = self.df["mos"].min()
            max_val = self.df["mos"].max()

        denom = (max_val - min_val) if max_val != min_val else 1.0
        self.df["normalized_score"] = (self.df["mos"] - min_val) / denom

        self.stats["mos_min"] = float(min_val)
        self.stats["mos_max"] = float(max_val)
        logger.info("MOS scores normalized to [0, 1]")
        return self.df


    def detect_outliers_3sigma(self) -> pd.DataFrame:
        """Detect outliers using 3-sigma rule."""
        mean = self.df["mos"].mean()
        std = self.df["mos"].std() if self.df["mos"].std() > 0 else 1.0
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        outliers = self.df[(self.df["mos"] < lower_bound) | (self.df["mos"] > upper_bound)]
        logger.info(f"Outliers detected: {len(outliers)} samples beyond 3σ")
        self.df["is_outlier"] = (self.df["mos"] < lower_bound) | (self.df["mos"] > upper_bound)
        return self.df


    def check_fold_score_distribution(self, n_splits: int = 5) -> List[Dict]:
        """Check K-fold cross-validation distribution."""
        fold_df = self.df
        if "split" in self.df.columns:
            fold_df = self.df[self.df["split"].isin(["train", "val"])].reset_index(drop=True)

        return check_fold_distribution(
            df=fold_df,
            n_splits=n_splits,
            random_state=42,
            dataset_name=self.dataset_name,
            verbose=True,
        )


    def run_full_eda(
        self,
        save_dir: Optional[Path] = None,
        skip_integrity: bool = False
    ) -> Dict:
        """Execute the complete data exploration and analysis pipeline."""
        if save_dir is None:
            save_dir = self.eda_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        self.df = self.load_metadata()
        if self.df is None or self.df.empty:
            logger.error(f"EDA pipeline initialization failed for: {self.dataset_name}")
            return {}

        if skip_integrity:
            integrity_res = {"corrupted": [], "missing": [], "repeated_sample_ids": []}
            logger.info("Skipping integrity check")
        else:
            integrity_res = self.check_integrity(skip_video_check=False)

        self.basic_statistics()
        self.ensure_split_column()
        self.check_filename_label_match()
        self.visualize_mos_distribution(save_dir)
        self.normalize_scores()
        self.detect_outliers_3sigma()
        fold_res = self.check_fold_score_distribution()

        self.stats.update({"integrity": integrity_res, "fold_variance": fold_res})
        logger.info(f"EDA complete for {self.dataset_name}")
        return self.stats
