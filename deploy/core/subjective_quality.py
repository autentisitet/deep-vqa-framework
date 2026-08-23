"""Ollama-backed subjective quality descriptions and Bayesian score fusion."""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubjectiveQualityError(ValueError):
    """Raised when Ollama cannot produce a valid subjective-quality response."""


@dataclass(frozen=True)
class SubjectiveQualityAssessment:
    description: str
    llm_score: int
    bayesian_score: float
    credible_interval: tuple[float, float]
    model_used: str
    frames_sampled: int


# The durable system instructions and examples live in deploy-config/ollama/Modelfile. Keep
# the request itself deliberately small so the model's configured persona is
# not duplicated (or accidentally drifted) in application code.
_QUALITY_PROMPT = "Assess this media only by technical visual quality."


def _opencv() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SubjectiveQualityError("opencv-python is required for subjective media assessment") from exc
    return cv2


def _encode_frame(frame: Any, max_side: int = 768) -> str:
    cv2 = _opencv()
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise SubjectiveQualityError("Unable to encode media frame for Ollama")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _image_payload(path: Path) -> list[str]:
    cv2 = _opencv()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SubjectiveQualityError("Unable to read image for subjective quality assessment")
    return [_encode_frame(image)]


def _video_payload(path: Path, max_frames: int) -> list[str]:
    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SubjectiveQualityError("Unable to open video for subjective quality assessment")
    try:
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        sample_count = min(max_frames, frame_count)
        positions = sorted({round(index * (frame_count - 1) / max(sample_count - 1, 1)) for index in range(sample_count)})
        frames: list[str] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            success, frame = capture.read()
            if success:
                frames.append(_encode_frame(frame))
        if not frames:
            raise SubjectiveQualityError("No decodable video frames were available for assessment")
        return frames
    finally:
        capture.release()


def _parse_ollama_response(response: str) -> tuple[str, int]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match is None:
            raise SubjectiveQualityError("Ollama did not return a JSON quality assessment")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SubjectiveQualityError("Ollama returned malformed quality-assessment JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"description", "quality_score"}:
        raise SubjectiveQualityError("Ollama response must contain exactly description and quality_score")
    description = str(payload.get("description", "")).strip()
    try:
        score = float(payload["quality_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SubjectiveQualityError("Ollama response is missing a numeric quality_score") from exc
    if not description:
        raise SubjectiveQualityError("Ollama response is missing a quality description")
    if len(description) > 50:
        raise SubjectiveQualityError("Ollama quality description must be at most 50 characters")
    if not math.isfinite(score):
        raise SubjectiveQualityError("Ollama quality_score must be finite")
    # Language models may encode a valid score as 78, 78.0, 78.5, or "78".
    # Normalize defensively at the application boundary and clamp the result to
    # the API contract instead of trusting prompt-level type instructions.
    normalized_score = max(0, min(100, int(score)))
    return description, normalized_score


def _bayesian_quality_score(score: float, prior_alpha: float = 5.0, prior_beta: float = 5.0) -> tuple[float, tuple[float, float]]:
    """Fuse the subjective score into a neutral Beta prior and return a 95% interval."""
    normalized = score / 100.0
    evidence_weight = 10.0
    alpha = prior_alpha + normalized * evidence_weight
    beta = prior_beta + (1.0 - normalized) * evidence_weight
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1.0))
    margin = 1.96 * math.sqrt(variance)
    return mean * 100.0, (max(0.0, mean - margin) * 100.0, min(1.0, mean + margin) * 100.0)


def assess_subjective_quality(
    path: Path,
    media_type: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
    max_video_frames: int,
) -> SubjectiveQualityAssessment:
    """Ask a local Ollama vision model for quality text and score; never persist media or results."""
    images = _image_payload(path) if media_type == "image" else _video_payload(path, max_video_frames)
    request_body = json.dumps(
        {
            "model": model,
            "prompt": _QUALITY_PROMPT,
            "images": images,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - configurable local Ollama URL
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SubjectiveQualityError(f"Ollama quality model is unavailable: {exc}") from exc

    description, score = _parse_ollama_response(str(payload.get("response", "")))
    bayesian_score, interval = _bayesian_quality_score(score)
    return SubjectiveQualityAssessment(
        description=description,
        llm_score=int(score),
        bayesian_score=round(bayesian_score, 2),
        credible_interval=(round(interval[0], 2), round(interval[1], 2)),
        model_used=model,
        frames_sampled=len(images),
    )


__all__ = ["SubjectiveQualityAssessment", "SubjectiveQualityError", "assess_subjective_quality"]
