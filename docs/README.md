# Deep-VQA Documentation

This directory is the developer documentation. The root
READMEs remain the project overview, while `make help` remains a command index.
The YAML configuration files and Compose files are the machine-readable source of truth
when prose and implementation disagree.

Run `make` to display the project command index. `make help`, `make h`, and
`make ?` are equivalent; `make --help` displays GNU Make's own options.

## Contents

- [Architecture map](architecture.html)
- [Cross-file contracts](contracts.md)
- [Deployment and container configuration](deployment.md)
- [Frontend behavior and access control](frontend.md)
- [Model and container memory](performance.md)
- [Known pitfalls](pitfalls.md)
- [Generated OpenAPI document](openapi.json)
- [Shared deployment defaults](../deploy-config/basic.yaml)
- [Internal-mode configuration](../deploy-config/profiles/infer_deploy.internal.yaml)
- [Proxy-mode configuration](../deploy-config/profiles/infer_deploy.proxy.yaml)
- [Local-mode configuration](../deploy-config/profiles/infer_deploy.local.yaml)

---

## Containers

| Component | Container | Image | Required for browser access |
| --- | --- | --- | --- |
| FastAPI/model runtime | `vqa-infer` | `vqa-infer:latest` | Yes |
| Frontend and `/api/` forwarder | `vqa-nginx` | `nginx:alpine` | Yes |
| Training runtime | `vqa-train` | `vqa-train:latest` | No |
| Development shell | `vqa-dev` | `vqa-dev:latest` | No |

---

## Supported commands

```text
make docker-dev
make docker-train
make docker-infer
make docker-infer DEPLOYMENT_MODE=proxy
make docker-infer GPU=1
make local
make docker-batch-test
make docker-deploy-check
make docker-memory-measure
make account-disable EMAIL=user@example.com
make account-enable EMAIL=user@example.com
make account-reset-password EMAIL=user@example.com
make docker-stop
make docker-manage
make docker-purge-train
make docker-purge-deploy
make docker-purge-all
```

Targets prefixed with `_docker-` are internal Makefile commands. User-facing
Make targets use hyphens (`cache-clean`); shell scripts under `scripts/` keep
snake_case (`cache_clean.sh`).

Host-side tooling lives in `scripts/` and needs a repository `.venv`: shell
entry points and standard-library Python utilities only. Anything that must run
against a live deployment (for example the account operations above) lives in
the `deploy/` package instead, so it ships inside the inference image.


---

## Python command-line interfaces

Python entry points expose `build_parser()`, `parse_args(argv=None)`, and
`run(...)`. Namespace objects remain inside `parse_args`: complex commands
return frozen project-owned dataclasses (`TrainingOptions`, `InferenceOptions`,
`BenchmarkOptions`, `MemoryOptions`, and separate build/project distribution
options). Account commands return `(action, email)` and version synchronization
returns a `check` boolean. Execution functions consume these explicit inputs;
`main(argv=None)` handles CLI wiring and usage errors. Callers can construct
options directly without manipulating `sys.argv` or instantiating a parser.
Memory measurement options exclude the password from their representation.

Third-party types stay within the layer that needs them: Torch tensors and
NumPy arrays remain native to numerical computation, database connections stay
in storage code, and HTTP schemas stay in the web layer. Output parameters are
reserved for measured buffer-reuse needs, rather than used to hide return types.

Use `--help` for usage and examples, and `--version` for the release version.
For commands with subcommands, these flags belong before the subcommand.
Account commands share an email parent parser; feature-distribution commands
share the required checkpoint and device options through `parents`.
Required inputs remain mandatory, while configured defaults remain optional.
Paths use `Path` where applicable, fixed alternatives use `choices`, and
numeric workload limits are validated before execution. Torch devices remain
open strings to support values such as `cuda:0`.

```text
python -m src.main --help
python -m deploy.inference.cli --version
python -m deploy.web.manage_account --help
python -m src.data.eda.feature_distribution build --help
python -m scripts.benchmark_models --help
python scripts/measure_container_memory.py --help
python scripts/sync_version.py --check
```

`deploy.inference.cli` is an offline checkpoint loader, not an HTTP client.
For localhost tests of the deployed service, use `make docker-deploy-check`
for health, then send a representative authenticated request to the Nginx
`/api/v1` entry; see [HTTP responses and business codes](deployment.md#http-responses-and-business-codes).
Do not treat a successful CLI inference as proof of the Nginx → Uvicorn →
FastAPI path. No `httpx` dependency is needed for the current health probe.

Deployment commands still require a selected deployment profile, as described
in the [deployment guide](deployment.md).


---

## Python type and callable contracts

Prefer concrete project types and `TypedDict` for known field structures.
Use `object` for unknown inputs, then narrow with `isinstance` before accessing
attributes or doing numeric work. Pydantic models and string-keyed configuration
mappings are checked at the inference boundary; unknown config objects and
invalid preprocessing values raise `TypeError`. Explicit `Any` annotations are
rejected by the Ruff banned-API rule. Third-party untyped APIs still need boundary
checks; this setting does not make them fully typed.

Function decorators preserve argument and return types with `ParamSpec` and a
return type parameter, and retain runtime metadata with `wraps`. Training cleanup
runs in `finally`, including when a wrapped stage raises. Visualization hooks
bind each layer name and are removed even if registration of a later layer fails.

Metadata loader implementations inherit `BaseMetadataLoader` and implement its
abstract `load` method. Factories that access class registries use `classmethod`;
helpers independent of instance and class state use `staticmethod`. Computed
paths stay as ordinary properties so filename and directory changes are visible.
Do not cache those properties, mutable configuration, model tensors, or artifacts
with `cached_property` / `lru_cache` without an explicit invalidation and memory
budget. No such caches are currently used in the Python application.

Training checkpoints serialize enum values as primitive strings so
`torch.load(weights_only=True)` can read them. CSV logs rotate at 500 MB, with
one header written when an empty file is opened, including after rotation.
