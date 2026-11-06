# Contributing to deep-vqa-framework

Thank you for your interest! Please follow these guidelines.

---

## 1. Environment Setup

Run `make` for the project command index. `make help`, `make h`, and `make ?`
are equivalent. `make --help` shows GNU Make's own options, not the project's
targets.

We use `uv` for dependency management. Run `make install` to install dependencies (uv will be installed automatically).

```bash
# Install core + dev + security tools (recommended for contributors)
make install DEV=1

# Verify environment
make info
```

> [!NOTE]
> If you're in China, add `--mirror` to use TUNA mirror for faster downloads:

```bash
make install MIRROR=1 DEV=1
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

# Remove only the dev/training or inference/web containers and images
make docker-purge-train
make docker-purge-deploy

# Check container environment status
make docker-manage
```

> [!NOTE]
> The Makefile auto-detects your container runtime (Docker or Podman).
> No manual configuration needed.

### Start the web service

The web interface reaches the IQA/VQA models through Nginx. From the
repository root:

```bash
make docker-infer
make docker-deploy-check
```

Internal mode uses email + password accounts; open the browser entry point and
register the first account. `make env-secrets` is only needed for the legacy
`auth.mode: api_key` deployment.

Use `make docker-infer DEPLOYMENT_MODE=proxy` only when a process in front of
Deep-VQA already owns the reachable IP, terminates HTTPS, and authenticates
users. Mode configuration lives under `deploy-config/profiles/`; see
[docs/deployment.md](../docs/deployment.md) for the IP/DNS/NAT path and mode
details.

---

## 2. Coding Standards

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before committing your code, please run:

```bash
make format-all     # Run all formatters together (black + isort + ruff)
make typecheck      # Run type checking
```

---

## 3. Development Workflow

- **Branching**: Please use descriptive branch names:

- `feature/<your-feature-name>` for new features
- `fix/<issue-description>` for bug fixes
- `docs/<update>` for documentation changes

- **Commits**: Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

```text
<type>(<scope>): <subject>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

- **Testing**: Before submitting a PR, please ensure the smoke test passes:

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

---

## 5. Security and CI

Repository security is handled primarily by GitHub automation and CI:

- CodeQL analysis for Python;
- Dependabot updates for Python and GitHub Actions;
- GitHub dependency review and secret-scanning features when enabled in the repository settings;
- shell, build, lint, type-check, deployment-contract, and pytest checks in CI.

Use `SECURITY.md` for private vulnerability reports. Do not add secrets,
private media, exploit payloads, or scanner reports to commits.

---

## 6. Reporting Issues

When reporting a bug, please include:

- The error message and stack trace
- Your environment details (OS, GPU, PyTorch version)
- Steps to reproduce the issue
- A minimal reproducible example if possible

---

## 7. Contribution Checklist

Before submitting a PR, ensure:

- [ ] Repository gate passes `make preflight`
- [ ] Code is formatted with `make fmt`
- [ ] Type checking passes `make typecheck`
- [ ] Smoke test passes
- [ ] Commit messages follow Conventional Commits
- [ ] Documentation has been updated if needed
- [ ] Change description and validation steps are clear

Thank you for contributing! 🎉
