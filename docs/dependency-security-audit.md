# Dependency Security Audit

Scope: `feature/v0.9.6` dependency tree, generated from `uv.lock` /
`requirements.txt` after the CPU-only PyTorch switch and `uv lock --upgrade`.
The exact CVE list and severity for each entry lives on the GitHub Dependabot
page; this document records the package-level risk surface and what has been
done.

---

## Summary

- The 151 Dependabot alerts were dominated by the NVIDIA CUDA runtime that
  `torch` pulls in on Linux. The default deployment is CPU, so that chain was
  not reachable.
- PyTorch is now installed from the CPU-only index (`download.pytorch.org/whl/cpu`),
  which removed the entire `nvidia-*` / `cuda-*` / `triton` chain.
- Every other dependency was upgraded to the latest version compatible with
  the declared constraints, and `requirements.txt` was re-exported.

---

## Removed by this change (main risk surface)

| Package | Why removed |
| --- | --- |
| `cuda-toolkit`, `cuda-bindings`, `cuda-pathfinder` | CUDA toolchain, GPU-only |
| `nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13` | CUDA math/DNN/collective libs |
| `nvidia-cufft`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13` | CUDA math libs |
| `nvidia-cuda-runtime`, `nvidia-cuda-nvrtc`, `nvidia-cuda-cupti`, `nvidia-cufile`, `nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, `nvidia-nvtx` | CUDA runtime/tooling |
| `triton` | CUDA kernel compiler |

These 19 packages were the largest source of low/moderate Dependabot alerts
and are no longer installed on the default CPU deployment.

---

## Current runtime dependency inventory

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.16.0
av==19.0.2
beautifulsoup4==4.15.0
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
decord==0.6.0
fastapi==0.162.0
filelock==4.2.9
fonttools==4.68.0
fsspec==2026.9.0
gdown==6.5.1
h11==0.16.0
idna==3.20
jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
loguru==0.7.3
markupsafe==3.0.4
matplotlib==3.11.2
mpmath==1.3.0
narwhals==2.28.0
networkx==3.7
numpy==2.5.4
opencv-python==5.0.0.93
opentelemetry-api==1.45.2
packaging==26.3
pandas==3.1.1
pillow==13.0.0
pydantic==2.14.3
pydantic-core==2.50.4
pyparsing==3.3.3
pysocks==1.7.1
python-dateutil==4.0.0
python-multipart==0.0.33
pyyaml==6.0.3
requests==2.34.2
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
setuptools==84.2.1
soupsieve==4.0.1
starlette==1.10.0
sympy==1.14.0
threadpoolctl==3.8.0
torch==2.15.0+cpu
torchaudio==2.11.1+cpu
torchvision==0.29.2+cpu
tqdm==4.70.1
typing-extensions==4.16.0
typing-inspection==0.4.5
tzdata==2026.6
urllib3==2.8.0
uvicorn==0.54.0
win32-setctime==1.2.1
```

---

## Packages to keep watching

These are the remaining packages whose ecosystems historically attract CVEs.
They are already on recent versions; exact advisories should be read from the
Dependabot page rather than inferred here.

| Package | Version | Notes |
| --- | --- | --- |
| `opencv-python` | 5.0.0.93 | Image/video decode CVEs; used only for sharpness/noise heuristics and uploads are magic-validated. |
| `pillow` | 13.0.0 | Image parsing CVEs; decode path is server-side and bounded. |
| `torch` / `torchvision` / `torchaudio` | 2.15.0+cpu / 0.29.2+cpu / 2.11.1+cpu | CPU build removes the CUDA chain; PyTorch itself is occasionally flagged. |
| `urllib3` / `requests` | 2.8.0 / 2.34.2 | Used by `gdown` for checkpoint downloads. |
| `jinja2` | 3.1.6 | Templating; not exposed to user input in the current routes. |
| `setuptools` | 84.2.1 | Build-time dependency; not shipped in the runtime image. |
| `starlette` / `fastapi` | 1.10.0 / 0.162.0 | Web framework; watch for upstream advisories. |

---

## Remaining manual steps

1. Wait for Dependabot to re-scan after this branch is pushed.
2. Filter the page by `Critical` and `High`.
3. Dismiss alerts that are unreachable (CPU-only deployment, build-time deps).
4. For anything still actionable, upgrade or pin as indicated.

---

## Reproduce / refresh this inventory

```bash
UV_CACHE_DIR=/tmp/uv-cache uv tree
UV_CACHE_DIR=/tmp/uv-cache uv export --locked --format requirements-txt --output-file requirements.txt
grep -E '^[a-zA-Z0-9._-]+==' requirements.txt
```
