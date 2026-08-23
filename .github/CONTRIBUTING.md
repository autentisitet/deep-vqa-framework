# Contributing to deep-vqa-framework

Thank you for your interest! Please follow these guidelines.

---

## 1. Environment Setup

We use `uv` for dependency management. Run `make install` to install dependencies (uv will be installed automatically).

```bash
# Install core + dev + security tools (recommended for contributors)
make install DEV=1 SECURITY=1

# Verify environment
make info
```

> [!NOTE]
> If you're in China, add `--mirror` to use TUNA mirror for faster downloads:

```bash
make install MIRROR=1 DEV=1 SECURITY=1
make info
```

### Containerized Development (No apt/sudo required)

If you're not using a Debian/Ubuntu system or prefer isolated environments, use Docker or Podman:

```bash
# Enter development container (interactive shell; source and tests are mounted)
make docker-dev

# Run training in container
make docker-train

# Start the default internal inference API and Nginx stack
make docker-infer

# Stop all containers
make docker-stop

# Check container environment status
make docker-manage
```

> [!NOTE]
> The Makefile auto-detects your container runtime (Docker or Podman).
> No manual configuration needed.

### Optional Ollama workflow

The normal contributor workflow is `make docker-infer`; it defaults to the
internal profile and does not require
Ollama. To develop or test the subjective-quality endpoint, run these commands
from the repository root:

```bash
# Create the local configuration and required internal-mode secrets.
make env-secrets

# Start Ollama, initialize deep-vqa-subjective, and start the browser-facing stack.
make docker-infer-internal-ollama
```

Use `make docker-infer DEPLOYMENT_MODE=public` for the stateless public profile.
Deployment policy files are under `deploy-config/profiles/`; training and model
YAML files are under `train-config/`.

Ollama is called by FastAPI and is not exposed through Nginx. On Podman, a
host-loopback proxy must be made container-reachable, for example
`OLLAMA_HTTP_PROXY=http://host.containers.internal:7897`; see
[knowledge/ollama.md](../knowledge/ollama.md) for proxy and SELinux details.

---

## 2. Coding Standards

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before committing your code, please run:

```bash
make format-all     # Run all formatters together (black + isort + ruff)
make typecheck      # Run type checking
```

---

## 3. Development Workflow

* **Branching**: Please use descriptive branch names:

- `feature/<your-feature-name>` for new features
- `fix/<issue-description>` for bug fixes
- `docs/<update>` for documentation changes

* **Commits**: Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

```text
<type>(<scope>): <subject>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

* **Testing**: Before submitting a PR, please ensure the smoke test passes:

```bash
uv run python -m src.main --smoke_test
```

---

## 4. Dependencies

Dependencies are managed via `pyproject.toml` with three categories:

| Category | Group | Description | Install with |
| :--- | :--- | :--- | :--- |
| **Core** | `[project.dependencies]` | Runtime dependencies (PyTorch, FastAPI, OpenCV, etc.) | `make setup` |
| **Dev** | `[project.optional-dependencies]` | Development tools (pytest, ruff, mypy, black, isort) | `make setup DEV=1 MIRROR=1` |
| **Security** | `[project.optional-dependencies]` | Security scanners (pip-audit, cyclonedx-bom, safety) | `make setup SECURITY=1 MIRROR=1` |

---

## 5. Security Checks

Run security scans to ensure no vulnerable dependencies:

```bash
# Individual checks
make vuln-audit    # pip-audit vulnerability scan
make sbom          # Generate SBOM (CycloneDX)
make safety        # Safety vulnerability scan

# Or you can run all security checks
make security-all
```

---

## 6. Reporting Issues

When reporting a bug, please include:

* The error message and stack trace
* Your environment details (OS, GPU, PyTorch version)
* Steps to reproduce the issue
* A minimal reproducible example if possible

---

## 7. Contribution Checklist

Before submitting a PR, ensure:

- [ ] Code passes `make check-code`
- [ ] Code is formatted with `make fmt`
- [ ] Type checking passes `make typecheck`
- [ ] Smoke test passes
- [ ] Commit messages follow Conventional Commits
- [ ] Documentation has been updated if needed
- [ ] Change description and validation steps are clear

Thank you for contributing! 🎉
