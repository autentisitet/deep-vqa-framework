# Change content

---

## Screenshots/Visualizations

---

## Test situation

- [ ] Smoke test passed locally (`uv run python -m src.main --smoke_test`)
- [ ] Code style check passed (`make check-code` and `make typecheck`)

---

## Change type

- [ ] Bug fix
- [ ] New feature
- [ ] Performance optimization
- [ ] Documentation update
- [ ] Refactor

---

## Commit message

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <subject>
```

**Common types:**

| Type | Description |
| :--- | :--- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `refactor` | Code refactoring (no behavior change) |
| `perf` | Performance optimization |
| `test` | Add or modify tests |
| `chore` | Maintenance tasks |
| `style` | Code style, formatting, or output format changes (no logic change) |

**Example:**

- `feat(model): add ResNet50 backbone`
- `fix(engine): resolve OOM in gradient accumulation`
- `docs(readme): update installation guide`

---

## Checklist

- [ ] Commit message follows Conventional Commits
- [ ] Documentation has been updated (if needed)
- [ ] Code passes `make check-code`
- [ ] Code is formatted with `make fmt`
- [ ] Type checking passes `make typecheck` (if applicable)
