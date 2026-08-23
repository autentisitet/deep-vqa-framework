# Ollama integration guide <a id="ollama-integration-guide"></a>

The optional containerized Ollama service for subjective-quality inference is
defined under `deploy-config/ollama/`. The normal choice for users is:

```bash
make docker-infer-internal-ollama
```

All commands in this guide are run from the repository root, not from the
`deploy-config/ollama/` directory. `make env-secrets` creates `.env` from the
repository defaults and generates missing internal-mode secrets; edit `.env`
only when changing ports, endpoints, or proxy settings:

```bash
make env-secrets
```

The internal profile requires the generated API key and cookie-signing secret.
For a trusted stateless demo without those secrets, use
`make docker-infer-public-ollama` instead.

Compose reads `.env` automatically, and the Python runtime reads the same file
when exported environment variables are not already present.

The Make targets pass the absolute `deploy-config/ollama/Modelfile` path when combining
project Compose files. This avoids Compose resolving `./Modelfile` against the
the Compose directory and accidentally creating a directory mount under Podman.

That command starts `vqa-ollama`, waits for its HTTP endpoint, downloads
`qwen2.5vl:3b`, and creates the `deep-vqa-subjective` model from `Modelfile`.
The Make target runs model initialization as a short-lived task from the same
`vqa-ollama` image. There is no separate model-init service or long-lived
initializer container.

Model downloads use `OLLAMA_HTTP_PROXY` and `OLLAMA_HTTPS_PROXY`, not the
generic host proxy variables. A host loopback proxy such as
`http://127.0.0.1:7897` is not reachable as-is from a container. For Podman,
use `host.containers.internal`; for Docker, use `host.docker.internal`:

```env
OLLAMA_HTTP_PROXY=http://host.containers.internal:7897
OLLAMA_HTTPS_PROXY=http://host.containers.internal:7897
```

Leave both variables empty when the container can access `registry.ollama.ai`
directly.

If Fedora/RHEL/Podman reports `Modelfile: permission denied`, this is usually
an SELinux mount-label issue. The Podman override applies `:Z` and disables
SELinux labeling for the local Ollama container, then reruns the short-lived
model setup task when you run an Ollama profile target.
The model is stored in the named `ollama-models` volume, so stopping the
container does not delete it.

Useful commands:

```bash
make docker-infer-internal-ollama  # internal profile + Ollama
make docker-infer-public-ollama    # public stateless profile + Ollama
make docker-deploy-check           # unified service health check
make docker-stop            # stop the project and Ollama containers
```

`make docker-infer` is the usual lightweight deployment and does not start
Ollama. Direct FastAPI debugging starts only FastAPI for API clients or debugging;
`make docker-dev` opens a development shell with the host source and tests
mounted; `make docker-train` runs the
configured training workflow. See `make help-*` for concise command guidance.

The compose file intentionally uses the official `ollama/ollama:latest`
image. Its service and container are named `vqa-ollama`. Model initialization
runs as a temporary task from that same image and is removed after completion.

Ollama is intentionally not exposed through Nginx. The browser calls FastAPI;
FastAPI calls Ollama over the private Compose network or the configured host
endpoint. This keeps Ollama model-management endpoints out of the public web surface.

For host-mode API deployment:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

The subjective-quality model name and timeout are read from
deployment profiles. Change that file when
using a different model; `.env` is only for the endpoint and container
connection settings.

For Docker Compose inference the default endpoint is
`http://host.docker.internal:11434`; Podman uses
`http://host.containers.internal:11434`. A successful Ollama profile target
proves that the endpoint answered and that model setup completed at that moment.
It does not guarantee model availability, API-to-Ollama connectivity, proxy
configuration, or continued network health. The Ollama profile targets validate
the model setup, while the inference API must still be tested end-to-end.
