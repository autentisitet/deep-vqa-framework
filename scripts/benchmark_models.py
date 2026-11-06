#!/usr/bin/env python3
"""Benchmark serving checkpoints on held-out test splits.

For each configured model this produces, from the dataset's held-out test
split rather than from the training logs:

- raw PLCC/SROCC/KROCC/RMSE/MAE and five-parameter-mapped PLCC/RMSE/MAE
- the same correlation calculations for a deterministic OpenCV heuristic
- first-inference and steady-state latency plus process RSS after model loading

Results are written as JSON plus a Markdown table ready to paste into the README
or a report. Nothing here is simulated: every number comes from real inference
against real media.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

from loguru import logger

from src import __version__
from src.config.loader import load_config
from src.models.metrics import MappedMetrics, MappingDetails, compute_mapped_metrics
from src.utils.file_loader import CaseInsensitiveAssetResolver

ROOT = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from deploy.inference.predictor import PredictionResult


class BaselineReport(TypedDict):
    name: str
    raw: dict[str, float]
    mapped: dict[str, float] | None
    mapping: MappingDetails
    delta_srocc: float
    delta_krocc: float
    delta_mapped_plcc: float | None


class ModelReport(TypedDict):
    model_id: str
    dataset: str
    test_samples: int
    parameters: int
    checkpoint_bytes: int
    load_seconds: float
    first_inference_ms: float
    warmup_runs: int
    rss_increase_mib: float | None
    accuracy: MappedMetrics
    baseline: BaselineReport | None
    latency: dict[str, float]
    failures: list[dict[str, str]]
    task_type: NotRequired[str]
    media_type: NotRequired[str]
    checkpoint: NotRequired[str]
    test_manifest: NotRequired[str]
    rss_after_load_mib: NotRequired[float | None]


class BenchmarkEnvironment(TypedDict):
    device: str
    python: str
    torch: str
    cpu_count: int | None
    threads: int


class BenchmarkReport(TypedDict):
    recorded_at: str
    environment: BenchmarkEnvironment
    models: list[ModelReport]
    process_rss_after_mib: float | None


MODEL_ROLES: dict[str, dict[str, str]] = {
    "iqa": {
        "dataset": "tid2013",
        "training_config": "swin_iqa",
        "checkpoint": "deploy/iqa-models/tid2013_best.pt",
    },
    "vqa": {
        "dataset": "konvid-1k",
        "training_config": "swin_vqa",
        "checkpoint": "deploy/vqa-models/konvid-1k_best.pt",
    },
}

BASELINE_NAME = "opencv-handcrafted-heuristic"


def _rss_mib() -> float | None:
    """Resident set size of this process in MiB, or None when unavailable."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def summarize_latency(samples_ms: list[float]) -> dict[str, float]:
    """Summarize latency with nearest-rank p95.

    The estimated serial rate is count divided by summed inference time. It is
    not a concurrent load-test result.
    """
    if not samples_ms:
        return {}
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    total_seconds = sum(ordered) / 1000.0
    return {
        "count": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "estimated_serial_items_per_second": (round(len(ordered) / total_seconds, 3) if total_seconds > 0 else 0.0),
    }


def find_test_manifest(dataset_slug: str) -> Path:
    """Return the newest held-out test manifest recorded for a dataset."""
    manifest_dir = ROOT / "results" / dataset_slug / "train_logs"
    candidates = sorted(manifest_dir.glob("*_test_manifest.csv"))
    if not candidates:
        raise FileNotFoundError(f"No test manifest under {manifest_dir}. Run training for '{dataset_slug}' first.")
    return candidates[-1]


def read_test_split(manifest_path: Path, limit: int | None) -> list[tuple[str, float]]:
    """Read (sample_id, true_mos) pairs from a test manifest."""
    import csv

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    pairs: list[tuple[str, float]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = (row.get("sample_id") or "").strip()
            raw_mos = (row.get("true_mos") or "").strip()
            if not sample_id or not raw_mos:
                continue
            pairs.append((sample_id, float(raw_mos)))
            if limit is not None and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError(f"No usable rows in {manifest_path}")
    return pairs


def load_dataset_expectations(dataset_name: str) -> tuple[float, float]:
    """Return the configured (mos_min, mos_max) for a dataset."""
    import yaml

    with (ROOT / "train-config" / "dataset_config.yaml").open(encoding="utf-8") as handle:
        datasets = yaml.safe_load(handle) or {}
    key = next((name for name in datasets if name.lower().strip() == dataset_name.lower()), None)
    if key is None:
        raise KeyError(f"Dataset '{dataset_name}' is not present in train-config/dataset_config.yaml")
    entry = datasets[key]
    return float(entry["mos_min"]), float(entry["mos_max"])


def benchmark_role(
    role: str,
    spec: dict[str, str],
    device: str,
    limit: int | None,
    include_baseline: bool,
    warmup_runs: int,
) -> ModelReport:
    """Evaluate one serving role on its held-out test split."""
    import torch

    from deploy.inference.model_loader import load_checkpoint
    from deploy.inference.predictor import predict_single
    from deploy.inference.technical_quality import assess_technical_quality

    cfg = load_config(
        config_dir=ROOT / "train-config",
        model_name=spec["training_config"],
        dataset_name=spec["dataset"],
    )
    data_dir = cfg.paths.resolve_path(Path(cfg.dataset.paths.root) / cfg.dataset.paths.data)
    checkpoint_path = ROOT / spec["checkpoint"]
    manifest_path = find_test_manifest(cfg.paths.dataset_slug(spec["dataset"]))
    split = read_test_split(manifest_path, limit)
    mos_min, mos_max = load_dataset_expectations(spec["dataset"])

    logger.info(f"[{role}] checkpoint={checkpoint_path}")
    logger.info(f"[{role}] manifest={manifest_path} samples={len(split)}")

    rss_before = _rss_mib()
    load_started = time.perf_counter()
    model, _ = load_checkpoint(checkpoint_path, device)
    load_seconds = round(time.perf_counter() - load_started, 3)
    model.eval()
    rss_after_load = _rss_mib()

    resolver = CaseInsensitiveAssetResolver(target_dir=data_dir)
    media_type = cfg.dataset.data_type

    def timed_inference(path: Path) -> tuple[PredictionResult, float]:
        """Run one inference and return its result and wall-clock latency."""
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = predict_single(model, path, cfg, device, mos_min, mos_max)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return prediction, (time.perf_counter() - started) * 1000.0

    probe_asset = None
    for sample_id, _ in split:
        try:
            probe_asset = resolver.find_asset(sample_id)
            break
        except Exception:  # noqa: BLE001 - the scored loop records individual failures
            continue
    if probe_asset is None:
        raise RuntimeError(f"[{role}] no held-out media could be resolved for latency warm-up")

    try:
        _, first_inference_ms = timed_inference(probe_asset.path)
        for _ in range(warmup_runs):
            timed_inference(probe_asset.path)
    except Exception as exc:  # noqa: BLE001 - setup failure needs the media/model cause
        raise RuntimeError(f"[{role}] latency warm-up failed: {exc}") from exc

    predicted: list[float] = []
    truth: list[float] = []
    baseline_scores: list[float] = []
    latencies_ms: list[float] = []
    failures: list[dict[str, str]] = []

    for sample_id, true_mos in split:
        try:
            asset = resolver.find_asset(sample_id)
        except Exception as exc:  # noqa: BLE001 - a missing asset must not abort the run
            failures.append({"sample_id": sample_id, "error": f"unresolved: {exc}"})
            continue

        try:
            result, latency_ms = timed_inference(asset.path)
            latencies_ms.append(latency_ms)
        except Exception as exc:  # noqa: BLE001 - report, do not abort
            failures.append({"sample_id": sample_id, "error": f"inference: {exc}"})
            continue

        mos_score = result.get("mos_score")
        if mos_score is None:
            failures.append({"sample_id": sample_id, "error": "no MOS score produced"})
            continue

        predicted.append(float(mos_score))
        truth.append(true_mos)

        if include_baseline:
            try:
                baseline_scores.append(float(assess_technical_quality(asset.path, media_type).score))
            except Exception as exc:  # noqa: BLE001 - baseline is optional
                failures.append({"sample_id": sample_id, "error": f"baseline: {exc}"})

    if not predicted:
        raise RuntimeError(f"[{role}] no samples were scored successfully")

    accuracy_report = compute_mapped_metrics(predicted, truth)
    raw_accuracy = {key: round(value, 4) for key, value in accuracy_report["raw"].items()}
    mapped_accuracy = (
        {key: round(value, 4) for key, value in accuracy_report["mapped"].items()}
        if accuracy_report["mapped"]
        else None
    )

    baseline: BaselineReport | None = None
    if include_baseline and baseline_scores and len(baseline_scores) == len(truth):
        baseline_report = compute_mapped_metrics(baseline_scores, truth)
        baseline_raw = baseline_report["raw"]
        baseline_mapped = baseline_report["mapped"]
        baseline = {
            "name": BASELINE_NAME,
            "raw": {key: round(baseline_raw[key], 4) for key in ("plcc", "srocc", "krocc")},
            "mapped": ({"plcc": round(baseline_mapped["plcc"], 4)} if baseline_mapped is not None else None),
            "mapping": baseline_report["mapping"],
            "delta_srocc": round(raw_accuracy["srocc"] - baseline_raw["srocc"], 4),
            "delta_krocc": round(raw_accuracy["krocc"] - baseline_raw["krocc"], 4),
            "delta_mapped_plcc": (
                round(mapped_accuracy["plcc"] - baseline_mapped["plcc"], 4)
                if mapped_accuracy is not None and baseline_mapped is not None
                else None
            ),
        }

    return {
        "model_id": role,
        "dataset": spec["dataset"],
        "task_type": cfg.dataset.task_type,
        "media_type": media_type,
        "checkpoint": spec["checkpoint"],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_samples": len(predicted),
        "test_manifest": str(manifest_path.relative_to(ROOT)),
        "accuracy": {
            "raw": raw_accuracy,
            "mapped": mapped_accuracy,
            "mapping": accuracy_report["mapping"],
        },
        "baseline": baseline,
        "latency": summarize_latency(latencies_ms),
        "first_inference_ms": round(first_inference_ms, 2),
        "warmup_runs": warmup_runs,
        "load_seconds": load_seconds,
        "rss_after_load_mib": rss_after_load,
        "rss_increase_mib": (
            round(rss_after_load - rss_before, 2) if rss_after_load is not None and rss_before is not None else None
        ),
        "failures": failures,
    }


def render_markdown(report: BenchmarkReport) -> str:
    """Render a paste-ready Markdown report."""
    environment = report["environment"]
    lines = [
        "# Deep-VQA benchmark report",
        "",
        f"Recorded: {report['recorded_at']}",
        "",
        "## Environment",
        "",
        f"- Device: `{environment['device']}` (no GPU used unless reported above)",
        f"- Python: `{environment['python']}`",
        f"- PyTorch: `{environment['torch']}`",
        f"- Host CPUs: {environment['cpu_count']}; torch threads pinned to {environment['threads']}",
        "",
        "Latency figures are single-process, single-request measurements with the",
        "thread count above. Re-measure on the deployment host before quoting",
        "latency for a specific machine.",
        "",
        "## Held-out test accuracy",
        "",
        "Raw values use model outputs directly. Mapped PLCC/RMSE use a",
        "five-parameter logistic function fitted on this test set. The mapping",
        "is for final reporting only and is not used for checkpoint selection.",
        "",
        "| Model | Dataset | N | SROCC | KROCC | Raw PLCC | Mapped PLCC | Raw RMSE | Mapped RMSE | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report["models"]:
        raw = entry["accuracy"]["raw"]
        mapped = entry["accuracy"]["mapped"] or {}
        lines.append(
            f"| {entry['model_id']} | {entry['dataset']} | {entry['test_samples']} | "
            f"{raw['srocc']:.4f} | {raw['krocc']:.4f} | {raw['plcc']:.4f} | "
            f"{mapped.get('plcc', float('nan')):.4f} | {raw['rmse']:.4f} | "
            f"{mapped.get('rmse', float('nan')):.4f} | {raw['mae']:.4f} |"
        )

    lines += [
        "",
        "## Deterministic baseline comparison",
        "",
        f"Baseline `{BASELINE_NAME}` scores the identical test samples with reproducible",
        "handcrafted OpenCV heuristics (Laplacian sharpness, median-residual noise, exposure/clipping,",
        "contrast, temporal stability). Both model and baseline receive the same",
        "five-parameter mapping before mapped PLCC is compared.",
        "",
        "| Model | Dataset | Model SROCC | Baseline SROCC | Delta | Model KROCC | Baseline KROCC | Delta | Model mapped PLCC | Baseline mapped PLCC | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report["models"]:
        baseline = entry.get("baseline")
        if not baseline:
            continue
        lines.append(
            f"| {entry['model_id']} | {entry['dataset']} | "
            f"{entry['accuracy']['raw']['srocc']:.4f} | {baseline['raw']['srocc']:.4f} | "
            f"{baseline['delta_srocc']:+.4f} | {entry['accuracy']['raw']['krocc']:.4f} | "
            f"{baseline['raw']['krocc']:.4f} | {baseline['delta_krocc']:+.4f} | "
            f"{(entry['accuracy']['mapped'] or {}).get('plcc', float('nan')):.4f} | "
            f"{(baseline['mapped'] or {}).get('plcc', float('nan')):.4f} | "
            f"{baseline['delta_mapped_plcc'] if baseline['delta_mapped_plcc'] is not None else float('nan'):+.4f} |"
        )

    lines += [
        "",
        "## Latency and footprint",
        "",
        "| Model | First inference (ms) | Warm-up runs | Steady median (ms) | p95 (ms) | Estimated serial items/s | Load (s) | Params (M) | Checkpoint (MiB) | RSS increase (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in report["models"]:
        latency = entry["latency"]
        growth = entry["rss_increase_mib"]
        lines.append(
            f"| {entry['model_id']} | {entry['first_inference_ms']:.2f} | {entry['warmup_runs']} | "
            f"{latency.get('median_ms', 0):.2f} | {latency.get('p95_ms', 0):.2f} | "
            f"{latency.get('estimated_serial_items_per_second', 0):.3f} | {entry['load_seconds']:.3f} | "
            f"{entry['parameters'] / 1e6:.3f} | {entry['checkpoint_bytes'] / 1048576:.1f} | "
            f"{growth if growth is not None else 'n/a'} |"
        )

    final_rss = report["process_rss_after_mib"]
    lines += [
        "",
        f"Process RSS with all selected models loaded: {final_rss if final_rss is not None else 'n/a'} MiB.",
        "",
        "All selected models share one FastAPI process, so",
        "the RSS increase per model is an estimate rather than an isolated allocation",
        "and the combined RSS is the deployment-relevant figure. Container-level limits",
        "are measured separately by `make docker-memory-measure`.",
        "",
    ]
    total_failures = sum(len(entry["failures"]) for entry in report["models"])
    lines += [f"Rejected/failed samples during the run: {total_failures}."]
    for entry in report["models"]:
        for failure in entry["failures"][:5]:
            lines.append(f"- `{entry['model_id']}` {failure['sample_id']}: {failure['error']}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class BenchmarkOptions:
    """Explicit execution inputs, independent of the CLI parser."""

    roles: tuple[str, ...] = ("iqa", "vqa")
    device: str = "cpu"
    threads: int = 2
    limit: int | None = None
    warmup_runs: int = 1
    skip_baseline: bool = False
    output: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without executing the command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python -m scripts.benchmark_models --roles iqa --limit 10\n  python -m scripts.benchmark_models --device cuda:0 --threads 2",
    )
    parser.add_argument("--roles", default="iqa,vqa", help="Comma-separated serving roles (default: iqa,vqa)")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Torch/OpenCV thread count; the default 2 emulates the documented 2-vCPU target",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N test samples per role")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Additional unmeasured runs after recording first-inference latency (default: 1)",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the OpenCV baseline comparison")
    parser.add_argument("--output", type=Path, default=None, help="Report stem (default: reports/benchmark/<utc>)")
    parser.add_argument("--version", action="version", version=f"Deep-VQA {__version__}")
    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkOptions:
    """Parse and validate CLI input, returning project-owned execution options."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")

    if args.threads <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--threads and --limit must be positive")
    selected = [role.strip() for role in args.roles.split(",")]
    if not selected or any(role not in MODEL_ROLES for role in selected):
        parser.error("--roles must contain iqa and/or vqa, separated by commas")
    return BenchmarkOptions(
        roles=tuple(selected),
        device=args.device,
        threads=args.threads,
        limit=args.limit,
        warmup_runs=args.warmup_runs,
        skip_baseline=args.skip_baseline,
        output=args.output,
    )


def run(options: BenchmarkOptions) -> int:
    """Consume parsed options and execute the command."""
    # Pin threads before torch/OpenCV do any work so latency reflects the target.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(options.threads)

    import cv2
    import torch

    torch.set_num_threads(options.threads)
    cv2.setNumThreads(options.threads)

    models: list[ModelReport] = []
    for role in options.roles:
        try:
            models.append(
                benchmark_role(
                    role,
                    MODEL_ROLES[role],
                    options.device,
                    options.limit,
                    include_baseline=not options.skip_baseline,
                    warmup_runs=options.warmup_runs,
                )
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            logger.error(f"[{role}] benchmark failed: {exc}")
            return 1

    report: BenchmarkReport = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "device": options.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "threads": options.threads,
        },
        "models": models,
        "process_rss_after_mib": _rss_mib(),
    }

    output_stem = options.output or (
        ROOT / "reports" / "benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    markdown_path = output_stem.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    logger.info(f"Benchmark JSON: {json_path}")
    logger.info(f"Benchmark report: {markdown_path}")
    for entry in models:
        logger.info(
            f"[{entry['model_id']}] N={entry['test_samples']} "
            f"SROCC={entry['accuracy']['raw']['srocc']:.4f} "
            f"raw_PLCC={entry['accuracy']['raw']['plcc']:.4f} "
            f"mapped_PLCC={(entry['accuracy']['mapped'] or {}).get('plcc', float('nan')):.4f} "
            f"median={entry['latency'].get('median_ms', 0):.1f}ms"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options and execute the command."""
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
