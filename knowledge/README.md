# Deep-VQA Knowledge Base

This directory is the canonical developer-facing knowledge base. The root
READMEs remain product overviews, while `make help` remains a command index.
Use this directory for operational details: deployment explains profiles,
ports, networks, health checks, and failure modes; frontend explains browser
behavior; Ollama explains the optional model service. The YAML profiles and
Compose files remain the machine-readable source of truth when prose and
implementation ever disagree.

## Contents

- [Architecture map](architecture.html)
- [Deployment and container profiles](deployment.md)
- [Frontend behavior and access control](frontend.md)
- [Optional Ollama integration](ollama.md)
- [Generated OpenAPI document](../docs/openapi.json)
- [Internal deployment policy](../deploy-config/profiles/infer_deploy.internal.yaml)
- [Public deployment policy](../deploy-config/profiles/infer_deploy.public.yaml)

## Runtime inventory

| Component | Container | Image | Required |
| --- | --- | --- | --- |
| FastAPI/model runtime | `vqa-infer` | `vqa-infer:latest` | Yes |
| Frontend/reverse proxy | `vqa-nginx` | `nginx:alpine` | Optional |
| Subjective quality helper | `vqa-ollama` | `ollama/ollama:latest` | Optional |
| Training/development runtime | `vqa-train` | `vqa-train:latest` | Optional |
| Containerized development shell | `vqa-dev` | `vqa-dev:latest` | Optional |

## Supported user entry points

```text
make docker-dev                 # reproducible dev shell; source/tests are mounted
make docker-train
make docker-infer               # default internal profile
make docker-infer DEPLOYMENT_MODE=public
make docker-infer-internal-ollama
make docker-infer-public-ollama
make docker-deploy-check
make docker-stop
make docker-manage
make docker-purge-all
```

Targets prefixed with `_docker-` are Makefile implementation details.
