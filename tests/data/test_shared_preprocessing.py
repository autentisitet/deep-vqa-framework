"""Training and serving must agree on temporal sampling and image transforms."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deploy.inference.preprocessor import Preprocessor as ServingPreprocessor
from src.core.trainer import ImageVideoDataset
from src.data import preprocessing


@pytest.mark.parametrize("frame_count,num_frames", [(10, 4), (2, 4)])
def test_training_and_serving_video_tensors_match_the_sampling_contract(monkeypatch, frame_count, num_frames):
    frames = np.stack([np.full((8, 12, 3), index * 20, dtype=np.uint8) for index in range(frame_count)])
    sampled = []

    class Batch:
        def __init__(self, indices):
            self.indices = indices

        def asnumpy(self):
            return frames[self.indices]

    class Reader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return frame_count

        def get_batch(self, indices):
            sampled.append(indices)
            return Batch(indices)

    monkeypatch.setattr(preprocessing, "VideoReader", Reader)
    dataset = ImageVideoDataset.__new__(ImageVideoDataset)
    dataset.num_frames = num_frames
    dataset.input_size = 8
    training = dataset._read_video("video.mp4")
    serving = ServingPreprocessor(num_frames=num_frames, input_size=8).process_video(Path("video.mp4"))
    indices = (
        np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()
        if frame_count >= num_frames
        else list(range(frame_count))
    )
    expected_frames = frames[indices]
    if frame_count < num_frames:
        expected_frames = np.concatenate(
            [expected_frames, np.repeat(expected_frames[-1:], num_frames - frame_count, axis=0)]
        )
    expected = preprocessing.rgb_video_array_to_imagenet_tensor(expected_frames, input_size=8)
    assert sampled == [indices, indices]
    assert training.shape == (num_frames, 3, 8, 8)
    torch.testing.assert_close(training, expected, rtol=0, atol=0)
    torch.testing.assert_close(serving, expected, rtol=0, atol=0)


def test_image_preprocessing_is_equal_and_training_transform_still_applies(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.full((12, 16, 3), 80, dtype=np.uint8)).save(path)
    dataset = ImageVideoDataset.__new__(ImageVideoDataset)
    dataset.input_size = 8
    dataset.transform = None
    serving = ServingPreprocessor(input_size=8).process_image(path)
    torch.testing.assert_close(dataset._read_image(str(path)), serving, rtol=0, atol=0)
    custom = torch.ones(3, 4, 4)
    dataset.transform = lambda _image: custom
    assert dataset._read_image(str(path)) is custom


def test_empty_video_is_rejected_before_requesting_frames(monkeypatch):
    class EmptyReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 0

        def get_batch(self, _indices):
            pytest.fail("empty video must not request a batch")

    monkeypatch.setattr(preprocessing, "VideoReader", EmptyReader)
    with pytest.raises(ValueError, match="Empty video"):
        preprocessing.Preprocessor().process_video(Path("empty.mp4"))


@pytest.mark.parametrize(
    ("convert", "shape"),
    [
        (preprocessing.rgb_array_to_imagenet_tensor, (8, 8, 3)),
        (preprocessing.rgb_video_array_to_imagenet_tensor, (2, 8, 8, 3)),
    ],
)
def test_read_only_media_arrays_convert_without_tensor_warnings(convert, shape):
    source = np.full(shape, 80, dtype=np.uint8)
    source.flags.writeable = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        actual = convert(source, input_size=8)
    expected = convert(source.copy(), input_size=8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not source.flags.writeable


def test_invalid_png_magic_without_decodable_image_is_client_input_error(tmp_path):
    image = tmp_path / "broken.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-image")
    with pytest.raises(ValueError, match="Image cannot be decoded"):
        preprocessing.Preprocessor(input_size=8).process_image(image)


def test_invalid_mp4_magic_without_decodable_video_is_client_input_error(tmp_path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 72)
    with pytest.raises(ValueError, match="Video cannot be decoded"):
        preprocessing.Preprocessor(num_frames=2, input_size=8).process_video(video)
