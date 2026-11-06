# Image Samples Directory

Place your test images here for batch inference.

## Supported Formats

- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`
- `.webp`

---

## Usage

```bash
make docker-batch-test IMAGES=1
```

The inference path verifies that the file content can be decoded as an image;
the filename suffix is not treated as proof of media type. For a single image,
feature-map and Grad-CAM artifacts are written under `reports/iqa-test/`:

```bash
uv run python -m deploy.inference.cli -i examples/images/mmexport1776570035144.jpg --visualize
```
