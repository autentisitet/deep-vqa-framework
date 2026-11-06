import asyncio
import os
from pathlib import Path

import pytest

from deploy.web.media_upload import (
    UploadLimitError,
    UploadTypeError,
    detect_media_magic,
    save_validated_upload,
)


class FakeUpload:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    async def read(self, _size):
        return next(self.chunks, b"")


def test_magic_detection_ignores_claimed_request_type() -> None:
    assert detect_media_magic(b"\x89PNG\r\n\x1a\n" + b"x" * 60).media_type == "image"
    assert detect_media_magic(b"\x00\x00\x00\x18ftypisom" + b"x" * 52).media_type == "video"


@pytest.mark.parametrize("header", [b"\x7fELFpayload", b"MZpayload", b"#!/bin/sh\n"])
def test_executable_magic_is_rejected_before_persistence(header: bytes) -> None:
    with pytest.raises(UploadTypeError, match="Executable"):
        detect_media_magic(header)


def test_validated_upload_uses_random_private_filename(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"payload"
    stored = asyncio.run(save_validated_upload(FakeUpload([png]), 1024, 1, tmp_path))
    try:
        assert stored.path.parent == tmp_path
        assert stored.path.suffix == ".png"
        assert len(stored.path.stem) == 64
        assert stored.path.name != "../../attack.png"
        assert stored.media.media_type == "image"
        assert stored.path.read_bytes() == png
        assert os.stat(stored.path).st_mode & 0o777 == 0o600
    finally:
        stored.path.unlink(missing_ok=True)


def test_limit_is_enforced_while_streaming_and_partial_file_is_removed(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\n"
    with pytest.raises(UploadLimitError):
        asyncio.run(save_validated_upload(FakeUpload([png, b"12345"]), len(png) + 3, 1, tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_unsupported_magic_is_never_persisted(tmp_path: Path) -> None:
    with pytest.raises(UploadTypeError):
        asyncio.run(save_validated_upload(FakeUpload([b"plain text"]), 1024, 1, tmp_path))

    assert list(tmp_path.iterdir()) == []
