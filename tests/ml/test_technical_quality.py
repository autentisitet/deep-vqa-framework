from pathlib import Path

import cv2
import numpy as np

from deploy.inference.technical_quality import assess_technical_quality


def _write_image(path: Path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), image)


def test_technical_quality_rejects_black_and_noise(tmp_path) -> None:
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    noise = np.random.default_rng(4).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    black_path = tmp_path / "black.png"
    noise_path = tmp_path / "noise.png"
    _write_image(black_path, black)
    _write_image(noise_path, noise)

    black_result = assess_technical_quality(black_path, "image")
    noise_result = assess_technical_quality(noise_path, "image")

    assert black_result.score < 40
    assert noise_result.score < 60
    assert "blurred" in black_result.description or "poor exposure" in black_result.description
    assert "heavy noise" in noise_result.description


def test_sharp_pattern_scores_above_blurred_pattern(tmp_path) -> None:
    pattern = (80 + ((np.indices((256, 256)).sum(axis=0) // 8) % 2) * 120).astype(np.uint8)
    sharp_path = tmp_path / "sharp.png"
    blurred_path = tmp_path / "blurred.png"
    _write_image(sharp_path, cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR))
    blurred = cv2.GaussianBlur(pattern, (51, 51), 18)
    _write_image(blurred_path, cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))

    sharp = assess_technical_quality(sharp_path, "image")
    blurred_result = assess_technical_quality(blurred_path, "image")

    assert sharp.score > blurred_result.score
    assert sharp.sharpness > blurred_result.sharpness
