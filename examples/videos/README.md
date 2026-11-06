# Video Samples Directory

Place your test videos here for batch inference.

## Supported Formats

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.wmv`

---

## Usage

```bash
make docker-batch-test VIDEOS=1
```

Video inputs are decoded with the configured OpenCV/Decord path before
inference. A forged or misleading filename suffix is rejected instead of being
trusted as the media identity.
