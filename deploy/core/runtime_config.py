from pathlib import Path

from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Deployment runtime defaults only."""

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    deploy_dir: Path = Path("deploy")
    reports_dir: Path = Path("reports")
    examples_dir: Path = Path("examples")

    iqa_model_path: Path = Field(default=Path("deploy/iqa-models/tid2013_best.pt"))
    vqa_model_path: Path = Field(default=Path("deploy/vqa-models/konvid-1k_best.pt"))

    num_frames: int = Field(default=8)
    input_size: int = Field(default=224)

    image_exts: set = Field(default={".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
    video_exts: set = Field(default={".mp4", ".avi", ".mov", ".mkv", ".wmv"})

    black_threshold: float = Field(default=8.0)
    white_threshold: float = Field(default=245.0)
    max_black_white_ratio: float = Field(default=0.3)
    max_frame_deviation: float = Field(default=0.10)

    default_device: str = Field(default="cuda" if __import__("torch").cuda.is_available() else "cpu")

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.project_root / candidate


cfg = InferenceConfig()


def ensure_runtime_dirs() -> None:
    for path in (
        cfg.resolve(cfg.examples_dir / "images"),
        cfg.resolve(cfg.examples_dir / "videos"),
        cfg.resolve(cfg.deploy_dir / "iqa-models"),
        cfg.resolve(cfg.deploy_dir / "vqa-models"),
        cfg.resolve(cfg.reports_dir / "iqa-test"),
        cfg.resolve(cfg.reports_dir / "vqa-test"),
    ):
        path.mkdir(parents=True, exist_ok=True)
