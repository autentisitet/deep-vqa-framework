from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InferenceConfig(BaseModel):
    """Deployment runtime defaults only."""

    iqa_model_path: Path = Field(default=Path("deploy/iqa-models/tid2013_best.pt"))
    vqa_model_path: Path = Field(default=Path("deploy/vqa-models/konvid_best.pt"))

    num_frames: int = Field(default=8)
    input_size: int = Field(default=224)

    image_exts: set = Field(default={".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
    video_exts: set = Field(default={".mp4", ".avi", ".mov", ".mkv", ".wmv"})

    black_threshold: float = Field(default=8.0)
    white_threshold: float = Field(default=245.0)
    max_black_white_ratio: float = Field(default=0.3)
    max_frame_deviation: float = Field(default=0.10)

    default_device: str = Field(default="cuda" if __import__("torch").cuda.is_available() else "cpu")


cfg = InferenceConfig()


def ensure_runtime_dirs() -> None:
    for path in (
        PROJECT_ROOT / "examples" / "images",
        PROJECT_ROOT / "examples" / "videos",
        PROJECT_ROOT / "deploy" / "iqa-models",
        PROJECT_ROOT / "deploy" / "vqa-models",
        PROJECT_ROOT / "reports" / "iqa-test",
        PROJECT_ROOT / "reports" / "vqa-test",
    ):
        path.mkdir(parents=True, exist_ok=True)
