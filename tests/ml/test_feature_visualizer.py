import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.visualization.feature_visualizer import FeatureVisualizer, to_spatial_feature_map  # noqa: E402


class TinyImageQualityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_backbone = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1))

    def forward(self, image):
        return self.head(self.image_backbone(image))


def test_to_spatial_feature_map_converts_nhwc_and_tokens() -> None:
    nhwc = torch.randn(1, 5, 7, 3)
    swin = torch.randn(1, 7, 7, 768)
    nchw = torch.randn(1, 768, 7, 7)
    tokens = torch.randn(1, 16, 4)

    assert to_spatial_feature_map(nhwc).shape == (1, 3, 5, 7)
    assert to_spatial_feature_map(swin).shape == (1, 768, 7, 7)
    assert to_spatial_feature_map(nchw).shape == (1, 768, 7, 7)
    assert to_spatial_feature_map(tokens).shape == (1, 4, 4, 4)


def test_run_image_produces_gradcam_for_differentiable_backbone() -> None:
    model = TinyImageQualityModel()
    visualizer = FeatureVisualizer(model)

    result = visualizer.run_image(torch.randn(1, 3, 16, 16), layers=["image_backbone"], cam_layer="image_backbone")

    assert result.score == pytest.approx(result.score)
    assert result.activations["image_backbone"].shape == (1, 4, 16, 16)
    assert result.cam is not None
    assert result.cam.shape == (1, 16, 16)
    assert torch.isfinite(result.cam).all()


def test_invalid_later_layer_cleans_up_already_registered_hooks():
    model = TinyImageQualityModel()
    visualizer = FeatureVisualizer(model)
    with pytest.raises(KeyError, match="Unknown layer"):
        visualizer.run_image(torch.randn(1, 3, 16, 16), layers=["image_backbone", "missing_layer"])
    assert not model.image_backbone._forward_hooks
    result = visualizer.run_image(torch.randn(1, 3, 16, 16), layers=["image_backbone"])
    assert set(result.activations) == {"image_backbone"}
    assert not model.image_backbone._forward_hooks


def test_hook_closures_keep_each_layer_name_and_clean_up_on_forward_failure(monkeypatch):
    model = TinyImageQualityModel()
    visualizer = FeatureVisualizer(model)
    result = visualizer.run_image(torch.randn(1, 3, 16, 16), layers=["image_backbone", "head"])
    assert result.activations["image_backbone"].shape == (1, 4, 16, 16)
    assert result.activations["head"].shape == (1, 1)

    def fail(_image):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        visualizer.run_image(torch.randn(1, 3, 16, 16), layers=["image_backbone", "head"])
    assert not model.image_backbone._forward_hooks
    assert not model.head._forward_hooks
