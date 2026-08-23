# Deep-VQA Repository Instructions

Use `.agents/skills/deep-vqa-engineering/SKILL.md` for implementation, review,
debugging, deployment, configuration, testing, documentation, and release work.

The project focuses on the ML engineering path from media preprocessing and
dataset validation through IQA/VQA training, evaluation, interpretability,
checkpoint handoff, and lightweight inference delivery. Do not implement
general SaaS infrastructure, multi-tenant identity, SSO, WAF, or server
database platforms as project features.

Treat `pyproject.toml`, `train-config/`, `deploy-config/`, tests, CI, and
`knowledge/` as the authoritative mechanical and architectural contracts.
Training YAML belongs under `train-config/`; deployment profiles, Compose,
Nginx, and Ollama assets belong under `deploy-config/`. Preserve unrelated
user changes and validate every edited surface proportionally.

The current release line is v0.9.0, following v0.7.5. Local release changes
must remain uncommitted until explicitly reviewed; do not push or create a
commit as part of ordinary implementation work.
