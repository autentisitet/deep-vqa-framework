"""Serving defaults for the shared training/inference media preprocessor."""

from src.data.preprocessing import Preprocessor as MediaPreprocessor

from ..config import cfg


class Preprocessor(MediaPreprocessor):
    """Apply deployment defaults while keeping media processing in src/."""

    def __init__(self, num_frames: int | None = None, input_size: int | None = None):
        super().__init__(
            num_frames=cfg.num_frames if num_frames is None else num_frames,
            input_size=cfg.input_size if input_size is None else input_size,
        )
