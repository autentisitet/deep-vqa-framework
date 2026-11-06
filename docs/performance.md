# Model memory and container memory

Deep-VQA runs one Uvicorn process and handles one model operation at a time.
Starting more Uvicorn worker processes does not share model memory: every
process loads its own IQA and VQA model copies. If several servers are used,
put a load balancer in front of them; each server still needs enough memory for
its own models.

Redis is only useful when several processes or servers need a shared request
queue. It does not reduce model memory usage.

## Models loaded at startup

The selected deployment configuration contains:

```yaml
inference_runtime:
  preload_models: true
  serialize_heavy_requests: true
```

When FastAPI starts, it validates and loads both checkpoints into memory once.
Evaluation requests reuse those model objects and do not call `torch.load`
again. Image evaluation, video evaluation, and Grad-CAM generation are handled
one at a time to limit peak CPU/GPU memory usage.

The models remain in memory until the FastAPI process exits. Restart the
process when a complete memory reset is required.

---

## Repeatable measurements

Use `make benchmark` for model accuracy and per-item inference time. It runs the
published checkpoints on each dataset's held-out test split and writes:

```text
reports/benchmark/<utc>.json
reports/benchmark/<utc>.md
```

The report separates checkpoint loading time, first inference time, and
steady-state median/p95 after warm-up. These values measure the local model
code path, not the complete browser-to-HTTP response time. Do not compare a CPU
steady-state median directly with the first CUDA HTTP request.

Use `make docker-memory-measure` to measure the complete running containers:

```bash
make docker-memory-measure
```

The host-side uploader streams media in 1 MiB chunks with an explicit
`Content-Length`, rather than loading and concatenating the whole multipart
body. Keep input files unchanged during a measurement. This bounds upload-body
buffering on the host; it does not reduce model or decoded-frame memory in the
containers. `make benchmark` also stops reading the manifest after the requested
number of usable rows when `--limit` is provided.

The command sends authenticated image and video requests while reading Docker
or Podman memory statistics every 0.05 seconds by default. It records memory
before requests, the largest sampled value during each request, and memory
after the request finishes. Because sampling is periodic, the sampled maximum
is a lower bound on the instantaneous peak. The report also records host
available memory and swap usage from `/proc/meminfo`. Results are written
to:

```text
reports/memory/latest.json
reports/memory/latest.md
```

Authentication follows the selected deployment mode: internal mode reads an
account's `DEEP_VQA_EMAIL` and `DEEP_VQA_PASSWORD` from the environment, while
the legacy `api_key` mode reuses `DEEP_VQA_API_KEY`. The values can also be
passed directly with `--email` / `--password`.

Reduced runs:

```bash
make docker-memory-measure MEMORY_ARGS="--skip-video"
make docker-memory-measure MEMORY_ARGS="--sample-interval 0.2 --output reports/memory/experiment-01.json"
```

Container image size mainly affects disk usage and download/build time. Model
parameters, intermediate tensors, video frames, and inference libraries are the
main sources of memory usage while the service is running.

---

## Small-machine configuration

On a 2-core, 4-GB machine, use one FastAPI process and one model request at a
time. Swap or zram can reduce the chance of an immediate out-of-memory failure,
but it may increase latency. Measure memory and latency on the actual target
machine because the CPU, operating system, container program, and PyTorch
allocator affect the result.
