# Landmines

Symptom -> cause -> fix. Every entry below was observed in this repository.

## 1. A missing file returns `200` with the homepage

- Symptom: a documentation link returned `200` and `10189` bytes.
- Cause: `try_files ... /index.html` treats any missing path as a client-side
  route. In the original case the bind mount also lacked `:z`, so the file was
  unreadable and the fallback hid it.
- Fix: link repository documentation to its GitHub URL and make the static
  Nginx fallback `=404` so missing assets/routes cannot serve `index.html`
  with HTTP 200. Keep a contract test for that behavior.

---

## 2. Login or save returns `403` only through Nginx

- Symptom: the same request succeeds on the API port and fails through the
  proxy.
- Cause: `proxy_set_header Host $host` drops the port, so the API compares the
  `Origin` header against a different origin.
- Fix: `Host $http_host;`, and verify in a browser. `curl` sends no `Origin`
  header, so a curl-only test never reaches the same-origin check.

---

## 3. Container is "running" but every probe reports `Connection refused`

- Symptom: `docker inspect --format .State.Health` shows only connection
  refusals; the healthcheck never recovers.
- Cause: the app raised during startup and exited before binding a socket, for
  example `DEEP_VQA_API_KEY must be set when auth.mode is api_key`.
- Fix: read `logs --tail`, fix the configuration, then recreate the container.
  `make env-secrets` generates the deployment secrets.

---

## 4. Editing a deployment profile changes nothing

- Symptom: `auth.mode` stays at its old value after editing the YAML.
- Cause: `deploy-config/` is copied into the image, and the process reads the
  baked copy when the bind mount is absent.
- Fix: the base Compose file mounts `deploy-config/` read-only; recreate the
  container after editing a profile, and keep the image copy in sync for hosts
  that do not mount it.

---

## 5. SELinux host: files inside a bind mount are invisible

- Cause: an unlabelled bind mount is unreadable to the container under an
  enforcing policy.
- Fix: add `:z` to the mount, or `:ro,z` when the container only reads it.

---

## 6. `unknown or invalid runtime name: nvidia`

- Cause: the GPU Compose overlay was used on a host without the NVIDIA
  container runtime.
- Fix: start without that overlay. `device: cpu` is correct on the 2 vCPU
  deployment host.

---

## 7. Compose tries to pull `docker.io/library/vqa-infer:latest`

- Cause: the image was built locally as `localhost/<name>`, so an unqualified
  service image resolves against Docker Hub.
- Fix: transfer the image as a tarball (`docker save | gzip` -> `rsync` ->
  `docker load`) and start with `up -d --pull never`.

---

## 8. `PermissionError` when writing `reports/`

- Cause: a container running as root created the directory; the host user then
  cannot write into it.
- Fix: `chown -R` the affected directories once, and keep host-side reports
  outside container-owned paths.

---

## 9. `make benchmark` fails with `ModuleNotFoundError: loguru`

- Cause: the benchmark runs on the host, not in the image, and needs the
  repository `.venv`.
- Fix: `make setup` on the host, or run the same script inside `vqa-dev`.

---

## 10. `rm -f /tmp/x.sqlite3*` aborts in zsh

- Cause: `nomatch` makes an unmatched glob a fatal error before `-f` applies.
- Fix: quote the pattern, list the files explicitly, or set `nonomatch` for the
  shell.

---

## 11. Latency numbers that cannot be compared

- `scripts/benchmark_models.py` measures in-process inference on a fixed thread
  count; `curl -w '%{time_total}'` measures the full HTTP path: upload, proxy,
  auth, inference, JSON serialization.
- The API rate-limits authentication attempts. Timed bursts that exceed the
  window turn into `429` or an implicit wait, which inflates the tail.
- Always state which of the two was measured, with the sample size and the
  percentile definition.

---

## 12. `docker compose down` runs `podman-compose` and fails

- Symptom: `no compose.yaml, docker-compose.yml or container-compose.yml file
  found`.
- Cause: on this host the `docker` CLI dispatches `compose` to podman-compose,
  while the Makefile always passes explicit `-f` files.
- Fix: use `make docker-stop` or `make docker-purge-all`, or pass every
  Compose file explicitly.

---

## 13. `make setup DEV=1 MIRROR=1` reports an unsatisfiable `tqdm`

- Symptom: uv finds `tqdm` on the CUDA wheel index but rejects the project's
  newer minimum version even though PyPI or TUNA provides it.
- Cause: uv's default first-index policy stops at the first index containing a
  package, and the PyTorch index intentionally contains only a partial package
  set.
- Fix: when a CUDA index is selected, `scripts/setup_env.sh` opts into
  `unsafe-best-match` across the explicitly trusted PyTorch and primary
  indexes. Keep that option scoped to this known pair of indexes.

---

## 14. `make preflight` says installed development tools are missing

- Symptom: `.venv/bin/ruff`, `.venv/bin/mypy`, and Python's pytest module run,
  but the Make targets report that those tools are not installed.
- Cause: the tool probe uses `uv pip show`; an unwritable uv cache makes that
  probe fail, and its output is suppressed by the Makefile.
- Fix: point uv at a writable cache for the check, for example
  `UV_CACHE_DIR=/tmp/deep-vqa-uv-cache make preflight`. This does not require
  reinstalling dependencies or changing the project environment.


---

## 15. A checkpoint containing string enums fails safe loading

- Symptom: a checkpoint saved from `Config.model_dump()` retains enum instances
  instead of primitive strings; `weights_only=True` cannot load those globals
  without additional allowlisting.
- Cause: `StrEnum` values still serialize as enum instances through pickle, and
  the training serializer previously converted paths but left enums untouched.
- Fix: convert enums to their values recursively before writing checkpoints.
  Keep a save/load regression using `weights_only=True`; do not allowlist enums
  as a workaround.

---

## 16. Invalid visualization layers leave hooks installed

- Symptom: after requesting a valid layer followed by an unknown layer, the
  valid layer retains a forward hook.
- Cause: hook registration happened before the `try/finally` cleanup scope.
- Fix: register hooks inside that scope and remove already registered handles
  when either layer resolution or the model forward pass fails.

---

## 17. CSV logs contain repeated headers and never rotate

- Symptom: a CSV header appears between log records and the size limit does
  not trigger rotation.
- Cause: the rotation callback wrote a header on every record and returned
  `None`, which Loguru treats as a decision not to rotate.
- Fix: use the 500 MB rotation threshold and a file opener that writes the
  header only when the file is empty. Verify both append and rotated files.


---

## 18. Image inference warns about a non-writable NumPy array

- Symptom: PyTorch reports that writing to a non-writable NumPy-backed tensor
  has undefined behavior when decoding an image with Pillow.
- Cause: `np.ascontiguousarray` reuses an already contiguous, read-only buffer
  returned by Pillow instead of making it writable.
- Fix: copy only read-only image/video buffers before `torch.from_numpy`; keep
  writable contiguous buffers on the zero-copy path. Regression tests treat the
  warning as an error for both image and video inputs.

---

## 19. A forged media header is reported as HTTP 500

- Symptom: a file starts with PNG/MP4 magic but Pillow/Decord cannot decode the remaining
  bytes; the service classifies the decoder exception as inference failure.
- Cause: checking magic bytes identifies a candidate format, not a valid image.
- Fix: normalize image/video decoder failures to a media `ValueError` at the decoder
  boundary, then convert that error to HTTP 422 in the existing API handler.
  Model execution failures still use HTTP 500.
