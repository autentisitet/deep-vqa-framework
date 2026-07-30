# Cloud GPU Platform Setup Guide

This guide covers renting and configuring a cloud GPU instance (AutoDL, Lambda Labs, RunPod) for the Deep-VQA-Framework.

## Instance Selection

> [!NOTE]
> The recommendations below are based on typical configurations. Actual usage depends on `batch_size`, `num_frames`, and model backbone. Profile your run to determine exact requirements.

| Component | IQA Only (Minimum) | VQA Training (Recommended) |
| :--- | :--- | :--- |
| **GPU** | RTX 3060 (12GB) / RTX 4060 Ti (16GB) | RTX 3090 / RTX 4090 |
| **VRAM** | 8GB – 16GB | 24GB |
| **Disk** | 100GB SSD | 200GB+ SSD |

> [!TIP]
> **AutoDL Users**: Select the "PyTorch 2.x + CUDA 12.x" base image. `setup_env.sh` pins Python to 3.12 via `uv`, regardless of the base image's Python version.

---

## Billing Strategy

Choose the billing method that matches your task duration:

| Method | Best For | Notes |
| :--- | :--- | :--- |
| **On-Demand** | Short tasks (debugging, smoke tests, < 3 hours) | Terminate immediately after use — on-demand charges accumulate quickly |
| **Prepaid (Daily/Weekly)** | Long-running training (24+ hours) | 30–50% discount vs. on-demand; ideal for overnight or multi-day runs |

---

## Storage: Data Disk vs System Disk

Cloud platforms typically provide a high-speed data disk separate from the system disk. **Always clone the repository and store datasets on the data disk** for better I/O performance.

**AutoDL example**:

```bash
# Example for AutoDL: Navigate to the data disk and clone the project
cd /root/autodl-tmp/
git clone https://github.com/autentisitet/deep-vqa-framework.git
```

> [!TIP]
> `cache_clean.sh` redirects HuggingFace/ModelScope caches to PROJECT_PARENT_DIR. If the repo is on the system disk, caches will also land there. Always clone to a data disk for optimal cache behavior and I/O performance.

**Self-check**:

```bash
df -h "$(pwd)" | grep -q "autodl-tmp" && echo "✅ On data disk" || echo "⚠️  Not on data disk — I/O will be slower"
```

---

## Additional Recommendations

* **SSH Security**: Use SSH keys instead of passwords. Change default passwords if using public instances.

* **Billing**: Set a reminder to stop/terminate instances after training. Auto-billing can accumulate quickly.

* **Decord Backend**: If OpenCV video codecs are missing on your instance, install Decord (`uv add decord`) as a fallback.

---

*For platform-specific issues, refer to your cloud provider's documentation or open an issue in the repository.*
