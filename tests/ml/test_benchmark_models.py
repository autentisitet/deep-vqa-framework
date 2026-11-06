import json
from pathlib import Path

from scripts import benchmark_models as benchmark


def test_summarize_latency_reports_stable_statistics() -> None:
    summary = benchmark.summarize_latency([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary["count"] == 5
    assert summary["mean_ms"] == 30.0
    assert summary["median_ms"] == 30.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["p95_ms"] == 50.0
    assert summary["estimated_serial_items_per_second"] > 0


def test_summarize_latency_handles_empty_and_single_samples() -> None:
    assert benchmark.summarize_latency([]) == {}

    single = benchmark.summarize_latency([12.5])
    assert single["count"] == 1
    assert single["p95_ms"] == 12.5
    assert single["median_ms"] == 12.5


def test_summarize_latency_uses_nearest_rank_p95() -> None:
    summary = benchmark.summarize_latency([float(value) for value in range(1, 31)])

    assert summary["p95_ms"] == 29.0


def test_read_test_split_honours_limit_and_skips_incomplete_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,true_mos,pred_mos\n"
        "i01_01_1.bmp,5.5139685,4.83\n"
        ",4.0,1.0\n"
        "i01_01_3.bmp,,2.0\n"
        "i01_01_4.bmp,3.25,3.1\n",
        encoding="utf-8",
    )

    assert benchmark.read_test_split(manifest, None) == [
        ("i01_01_1.bmp", 5.5139685),
        ("i01_01_4.bmp", 3.25),
    ]
    assert benchmark.read_test_split(manifest, 1) == [("i01_01_1.bmp", 5.5139685)]


def _report(with_baseline: bool = True) -> dict:
    return {
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "environment": {
            "device": "cpu",
            "python": "3.12.13",
            "torch": "2.13.0",
            "cpu_count": 24,
            "threads": 2,
        },
        "models": [
            {
                "model_id": "iqa",
                "dataset": "tid2013",
                "test_samples": 238,
                "parameters": 28_046_561,
                "checkpoint_bytes": 337_100_000,
                "load_seconds": 0.365,
                "first_inference_ms": 91.25,
                "warmup_runs": 1,
                "rss_increase_mib": 181.71,
                "accuracy": {
                    "raw": {
                        "plcc": 0.8446,
                        "srocc": 0.87,
                        "krocc": 0.6901,
                        "rmse": 1.191,
                        "mae": 0.9993,
                    },
                    "mapped": {"plcc": 0.9012, "rmse": 0.91, "mae": 0.73},
                    "mapping": {"applied": True, "method": "five_parameter_logistic", "parameters": [1, 2, 3, 4, 5]},
                },
                "baseline": (
                    {
                        "name": "opencv-handcrafted-heuristic",
                        "raw": {"plcc": 0.1199, "srocc": 0.1321, "krocc": 0.09},
                        "mapped": {"plcc": 0.2},
                        "mapping": {"applied": True, "method": "five_parameter_logistic"},
                        "delta_srocc": 0.7379,
                        "delta_krocc": 0.6001,
                        "delta_mapped_plcc": 0.7012,
                    }
                    if with_baseline
                    else None
                ),
                "latency": {
                    "count": 238,
                    "median_ms": 54.3,
                    "p95_ms": 56.62,
                    "estimated_serial_items_per_second": 18.414,
                },
                "failures": [],
            }
        ],
        "process_rss_after_mib": 886.79,
    }


def test_render_markdown_includes_accuracy_baseline_and_latency_tables() -> None:
    markdown = benchmark.render_markdown(_report())

    assert "| iqa | tid2013 | 238 | 0.8700 | 0.6901 | 0.8446 | 0.9012 |" in markdown
    assert "| iqa | tid2013 | 0.8700 | 0.1321 | +0.7379 | 0.6901 | 0.0900 | +0.6001 |" in markdown
    assert "| iqa | 91.25 | 1 | 54.30 | 56.62 | 18.414 |" in markdown
    assert "Process RSS with all selected models loaded: 886.79 MiB." in markdown
    assert "Rejected/failed samples during the run: 0." in markdown
    assert "torch threads pinned to 2" in markdown
    assert "First inference (ms)" in markdown
    assert "Estimated serial items/s" in markdown


def test_render_markdown_omits_baseline_table_rows_without_baseline() -> None:
    markdown = benchmark.render_markdown(_report(with_baseline=False))

    assert "Baseline SROCC" in markdown
    assert "| iqa | tid2013 | 0.8700 | 0.1321 |" not in markdown


def test_model_roles_cover_both_serving_profiles() -> None:
    assert set(benchmark.MODEL_ROLES) == {"iqa", "vqa"}
    for role in benchmark.MODEL_ROLES.values():
        assert role["checkpoint"].startswith("deploy/")
        assert role["dataset"]
        assert role["training_config"]


def test_report_json_round_trips_through_the_documented_shape() -> None:
    # The JSON file is the machine-readable contract for later reporting.
    payload = json.loads(json.dumps(_report()))
    assert set(payload) == {"recorded_at", "environment", "models", "process_rss_after_mib"}


def test_limited_manifest_does_not_parse_unused_rows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,true_mos\nfirst,1.5\nunused,not-a-number\n")
    assert benchmark.read_test_split(manifest, 1) == [("first", 1.5)]
