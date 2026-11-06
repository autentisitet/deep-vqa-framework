"""CLI contracts: parsing rejects invalid workloads before running them."""

import importlib
from pathlib import Path

import pytest

MODULES = (
    "src.main",
    "deploy.inference.cli",
    "deploy.web.manage_account",
    "src.data.eda.feature_distribution",
    "scripts.benchmark_models",
    "scripts.measure_container_memory",
    "scripts.sync_version",
)


@pytest.mark.parametrize("name", MODULES)
def test_help_and_version_exit_without_running_workloads(name, capsys, monkeypatch):
    module = importlib.import_module(name)

    def unexpected_run(*_args):
        pytest.fail("help/version must not execute the command")

    monkeypatch.setattr(module, "run", unexpected_run)
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as error:
            module.main([flag])
        assert error.value.code == 0
        output = capsys.readouterr().out
        assert "Examples:" in output if flag == "--help" else "Deep-VQA" in output


@pytest.mark.parametrize("action", ("enable", "disable", "reset-password"))
def test_account_parent_requires_and_normalizes_email(action):
    module = importlib.import_module("deploy.web.manage_account")
    args = module.parse_args([action, " User@Example.COM "])
    assert args == (action, "user@example.com")
    for argv in ([action], [action, "invalid-email"]):
        with pytest.raises(SystemExit) as error:
            module.parse_args(argv)
        assert error.value.code == 2


@pytest.mark.parametrize(
    "name,argv",
    [
        ("scripts.benchmark_models", ["--roles", ""]),
        ("scripts.benchmark_models", ["--roles", "iqa,unknown"]),
        ("scripts.benchmark_models", ["--threads", "0"]),
        ("scripts.benchmark_models", ["--limit", "-1"]),
        ("scripts.benchmark_models", ["--warmup-runs", "-1"]),
        ("scripts.measure_container_memory", ["--sample-interval", "nan"]),
        ("scripts.measure_container_memory", ["--timeout", "0"]),
        ("scripts.measure_container_memory", ["--settle-seconds", "-1"]),
        ("deploy.inference.cli", ["-i", "image.jpg", "--technical-baseline", "--visualize"]),
        ("src.data.eda.feature_distribution", ["build", "--checkpoint", "model.pt", "--batch-size", "0"]),
    ],
)
def test_invalid_workloads_are_usage_errors(name, argv):
    with pytest.raises(SystemExit) as error:
        importlib.import_module(name).parse_args(argv)
    assert error.value.code == 2


def test_distribution_parent_preserves_paths_device_and_operation_type(tmp_path):
    module = importlib.import_module("src.data.eda.feature_distribution")
    build = module.parse_args(["build", "--checkpoint", "model.pt", "--device", "cuda:0"])
    assert build.checkpoint == Path("model.pt")
    assert build.device == "cuda:0"
    assert isinstance(build, module.BuildDistributionOptions)
    image = tmp_path / "image.jpg"
    image.touch()
    project = module.parse_args(
        [
            "project",
            "--checkpoint",
            "model.pt",
            "--device",
            "cuda:0",
            "--distribution-file",
            "train.npz",
            "--input",
            str(image),
            "--output",
            "out.json",
        ]
    )
    assert project.checkpoint == build.checkpoint
    assert project.device == build.device
    assert isinstance(project, module.ProjectDistributionOptions)
    assert project.input == (image,)
    assert project.output == Path("out.json")


@pytest.mark.parametrize(
    "name,argv,option_type",
    [
        ("src.main", ["--dataset", "konvid-1k", "--smoke_test"], "TrainingOptions"),
        ("deploy.inference.cli", ["-i", "sample.jpg", "--visualize"], "InferenceOptions"),
        ("scripts.benchmark_models", ["--roles", "vqa,iqa", "--limit", "3"], "BenchmarkOptions"),
        ("scripts.measure_container_memory", ["--skip-video", "--timeout", "7"], "MemoryOptions"),
    ],
)
def test_main_passes_frozen_project_options_to_execution(name, argv, option_type, monkeypatch):
    from dataclasses import FrozenInstanceError, is_dataclass

    module = importlib.import_module(name)
    received = []
    monkeypatch.setattr(module, "run", lambda options: received.append(options))
    module.main(argv)
    assert len(received) == 1
    options = received[0]
    assert isinstance(options, getattr(module, option_type))
    assert is_dataclass(options)
    field = next(iter(options.__dataclass_fields__))
    with pytest.raises(FrozenInstanceError):
        setattr(options, field, "changed")
    if option_type == "BenchmarkOptions":
        assert options.roles == ("vqa", "iqa")
        assert options.limit == 3


def test_memory_options_do_not_expose_password_in_repr():
    module = importlib.import_module("scripts.measure_container_memory")
    options = module.parse_args(["--password", "test-password-only"])
    assert options.password == "test-password-only"
    assert "test-password-only" not in repr(options)


def test_distribution_execution_accepts_explicit_options_without_parser(monkeypatch):
    module = importlib.import_module("src.data.eda.feature_distribution")
    received = []
    monkeypatch.setattr(module, "_build_distribution_file", lambda options: received.append(("build", options)))
    monkeypatch.setattr(module, "_project_inputs", lambda options: received.append(("project", options)))
    monkeypatch.setattr(module, "build_parser", lambda: pytest.fail("execution must not construct a parser"))
    build = module.BuildDistributionOptions(checkpoint=Path("model.pt"), device="cpu")
    project = module.ProjectDistributionOptions(
        checkpoint=Path("model.pt"),
        device="cpu",
        distribution_file=Path("train.npz"),
        input=(Path("image.jpg"),),
        output=Path("out.json"),
    )
    module.run(build)
    module.run(project)
    assert received == [("build", build), ("project", project)]


def test_account_execution_uses_explicit_arguments_and_leaves_usage_errors_to_main(monkeypatch):
    module = importlib.import_module("deploy.web.manage_account")
    operations = []

    class Store:
        def __init__(self, *_args):
            pass

        def initialize(self):
            pass

        def set_disabled(self, email, disabled):
            operations.append((email, disabled))

    monkeypatch.setattr(module, "AccountStore", Store)
    monkeypatch.setattr(module, "build_parser", lambda: pytest.fail("execution must not construct a parser"))
    module.run("disable", "user@example.com")
    module.run("enable", "user@example.com")
    assert operations == [("user@example.com", True), ("user@example.com", False)]
    with pytest.raises(ValueError, match="Unknown account operation"):
        module.run("typo", "user@example.com")


def test_version_execution_returns_exit_status_without_exiting(monkeypatch):
    module = importlib.import_module("scripts.sync_version")
    calls = []
    monkeypatch.setattr(module, "project_version", lambda: "1.0.0")
    monkeypatch.setattr(module, "sync", lambda version, write=True: calls.append((version, write)) or True)
    assert module.run(check=True) == 1
    assert calls == [("1.0.0", False)]
    assert module.parse_args([]) is False
    assert module.parse_args(["--check"]) is True


def test_release_version_metadata_matches_package_lock_and_public_docs() -> None:
    import json
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = next(item for item in locked["package"] if item["name"] == project["name"])
    schema = json.loads((root / "docs/openapi.json").read_text(encoding="utf-8"))
    version = project["version"]
    assert version == package["version"] == schema["info"]["version"]
    for readme, label in (("README.md", "Version"), ("README_zh.md", "版本")):
        contents = (root / readme).read_text(encoding="utf-8")
        assert f"version-{version}-blue.svg" in contents
        assert f'alt="{label} {version}"' in contents
    assert f"deep-vqa-framework v{version} and later" in (root / "DISCLAIMER.md").read_text(encoding="utf-8")
