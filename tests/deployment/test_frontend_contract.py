"""Frontend and backend machine-contract values must not drift apart."""

import re
from pathlib import Path

from src.data.dataset_types import DatasetType

ROOT = Path(__file__).resolve().parents[2]


def _js_set_entries(path: Path, variable: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"const {variable} = new Set\(\[([^\]]+)\]\)", text)
    assert match is not None, f"{variable} not found in {path.name}"
    return {item.strip().strip("\"'").lower() for item in match.group(1).split(",") if item.strip()}


def test_frontend_accept_lists_match_backend_stable_extensions() -> None:
    """The browser accept list must equal the serving-side stable extensions."""
    app_js = ROOT / "frontend/app.js"
    stable = DatasetType.stable_extensions()

    backend_image = {ext.lstrip(".").lower() for ext in stable if ext in DatasetType.extensions_for(DatasetType.IMAGE)}
    backend_video = {ext.lstrip(".").lower() for ext in stable if ext in DatasetType.extensions_for(DatasetType.VIDEO)}

    assert _js_set_entries(app_js, "IMAGE_EXTENSIONS") == backend_image
    assert _js_set_entries(app_js, "VIDEO_EXTENSIONS") == backend_video
