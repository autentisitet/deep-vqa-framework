# Contributing to deep-vqa-framework

Thank you for your interest! Please follow these guidelines.

---

## 1. Environment Setup

We use `uv` for dependency management. Run `make setup` to install dependencies (uv will be installed automatically).

```bash
# Install core + dev + security tools (recommended for contributors)
make setup SETUP_ARGS="--dev --security"
```

> [!NOTE]
> If you're in China, add `--mirror` to use TUNA mirror for faster downloads:

```bash
make setup SETUP_ARGS="--mirror --dev --security"
```

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
| **Dev** | `[project.optional-dependencies]` | Development tools (ruff, mypy, black, isort) | `make setup SETUP_ARGS="--mirror --dev"` |
| **Security** | `[project.optional-dependencies]` | Security scanners (pip-audit, cyclonedx-bom, safety) | `make setup SETUP_ARGS="--mirror --security"` |

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

## 7. Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code passes `make check-code`
- [ ] Code is formatted with `make fmt`
- [ ] Type checking passes `make typecheck`
- [ ] Smoke test passes
- [ ] Commit messages follow Conventional Commits
- [ ] Documentation has been updated if needed
- [ ] PR has a clear description of changes

Thank you for contributing! 🎉
