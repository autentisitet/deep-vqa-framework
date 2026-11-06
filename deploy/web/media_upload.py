"""Bounded, magic-validated media upload persistence."""

from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .exceptions import UploadLimitError, UploadTimeoutError, UploadTypeError


class AsyncUpload(Protocol):
    async def read(self, size: int) -> bytes:
        """Read up to ``size`` bytes from the async upload source."""
        ...


@dataclass(frozen=True)
class DetectedMedia:
    media_type: str
    suffix: str
    format_name: str


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    media: DetectedMedia
    size_bytes: int


_EXECUTABLE_MAGICS = (
    b"\x7fELF",
    b"MZ",
    b"#!",
    b"\x00asm",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
)

# Simple prefix signatures: matched in order, one row per format.
_MEDIA_MAGIC_RULES: tuple[tuple[tuple[bytes, ...], str, str, str], ...] = (
    ((b"\xff\xd8\xff",), "image", ".jpg", "jpeg"),
    ((b"\x89PNG\r\n\x1a\n",), "image", ".png", "png"),
    ((b"BM",), "image", ".bmp", "bmp"),
    ((b"II*\x00", b"MM\x00*"), "image", ".tiff", "tiff"),
    ((b"\x1aE\xdf\xa3",), "video", ".mkv", "matroska/webm"),
    ((b"0&\xb2u\x8ef\xcf\x11",), "video", ".wmv", "wmv/asf"),
)

_RIFF_SUBTYPES = {
    b"WEBP": ("image", ".webp", "webp"),
    b"AVI ": ("video", ".avi", "avi"),
}


def _ftyp_media(brand: bytes) -> DetectedMedia:
    """Map an ISO-BMFF ``ftyp`` brand to a QuickTime or MP4 detection."""
    is_mov = brand == b"qt  "
    return DetectedMedia("video", ".mov" if is_mov else ".mp4", "quicktime" if is_mov else "mp4")


def detect_media_magic(header: bytes) -> DetectedMedia:
    """Return a canonical media type from bytes, never from the request MIME."""
    if any(header.startswith(magic) for magic in _EXECUTABLE_MAGICS):
        raise UploadTypeError("Executable uploads are not accepted")

    for prefixes, media_type, suffix, format_name in _MEDIA_MAGIC_RULES:
        if header.startswith(prefixes):
            return DetectedMedia(media_type, suffix, format_name)

    if header.startswith(b"RIFF"):
        subtype = _RIFF_SUBTYPES.get(header[8:12])
        if subtype is not None:
            media_type, suffix, format_name = subtype
            return DetectedMedia(media_type, suffix, format_name)

    if len(header) >= 12 and header[4:8] == b"ftyp":
        return _ftyp_media(header[8:12])

    raise UploadTypeError("Unsupported media file magic")


async def _read_chunk(file: AsyncUpload, timeout_seconds: float) -> bytes:
    """Read one chunk from the upload, raising on a per-read timeout."""
    try:
        return await asyncio.wait_for(file.read(1024 * 1024), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise UploadTimeoutError(f"Upload read exceeded {timeout_seconds:g} seconds") from exc


async def save_validated_upload(
    file: AsyncUpload,
    max_bytes: int,
    timeout_seconds: float,
    upload_dir: Path | None = None,
) -> StoredUpload:
    """Validate the first bytes before persisting to a random non-executable path."""
    if max_bytes <= 0:
        raise ValueError("uploads.max_bytes must be positive")
    if timeout_seconds <= 0:
        raise ValueError("uploads.read_timeout_seconds must be positive")

    first_chunk = await _read_chunk(file, timeout_seconds)
    if not first_chunk:
        raise UploadTypeError("Uploaded media file is empty")
    media = detect_media_magic(first_chunk[:64])
    if len(first_chunk) > max_bytes:
        raise UploadLimitError(f"Uploaded file exceeds the {max_bytes} byte limit")

    root = upload_dir or Path(tempfile.gettempdir()) / "deep-vqa-uploads"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / f"{secrets.token_hex(32)}{media.suffix}"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    total_bytes = len(first_chunk)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(first_chunk)
            while True:
                chunk = await _read_chunk(file, timeout_seconds)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise UploadLimitError(f"Uploaded file exceeds the {max_bytes} byte limit")
                target.write(chunk)
        return StoredUpload(path=path, media=media, size_bytes=total_bytes)
    except Exception:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "DetectedMedia",
    "StoredUpload",
    "UploadLimitError",
    "UploadTimeoutError",
    "UploadTypeError",
    "detect_media_magic",
    "save_validated_upload",
]
