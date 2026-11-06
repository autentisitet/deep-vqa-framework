# Deep-VQA Repository Instructions

This file is the standing instruction set for automated coding agents and a
short orientation for contributors. Follow it for implementation, review,
debugging, deployment, configuration, testing, documentation, and release work.

## Scope

Deep-VQA is a single-instance IQA/VQA service: media validation -> training ->
evaluation -> interpretability artifacts -> HTTP delivery. It is an ML
engineering deliverable, not a SaaS platform.

Do not implement: roles and organisations, SSO or identity-provider
integration, WAF, message queues, external caches, microservice splits, or any
bundled language model. Per-instance email + password accounts are in scope
(they bind artifacts to a person).

---

## Authoritative contracts

Treat these as the mechanical and architectural source of truth:

- `pyproject.toml` — dependencies and the release version.
- `train-config/` — training YAML.
- `deploy-config/` — deployment profiles, Compose, and Nginx.
- `tests/` — the executable contracts.
- `docs/contracts.md` — which files change together for a given behaviour.
- `docs/pitfalls.md` — observed landmines (symptom -> cause -> fix).

When prose and implementation disagree, the tests, lockfiles, and YAML win.
Change the smallest surface; extend an existing target, model, or document
instead of adding a parallel layer. Preserve unrelated user changes. Do not
commit or push; local release changes stay uncommitted until reviewed.

---

## Development loop

1. Read `docs/README.md`, then open only the doc page the task needs.
2. Reproduce before changing: a failing test, a `curl`, or a `make` target.
   Record the command and its output.
3. Change the smallest surface.
4. Run the gate: `make preflight` (or `bash scripts/preflight.sh`).
5. When behaviour is user-visible, exercise it through Nginx on
   `http://127.0.0.1:8000`.
6. Report evidence: the command plus its measured value.

---

## Updating documentation

Update the file(s) that describe the behaviour you changed, and keep the
cross-references in sync:

| Change | Update |
| --- | --- |
| Deployment modes, ports, TLS, exposure | `docs/deployment.md`, the mode tables in `README.md` and `README_zh.md`, `SECURITY.md`, `DISCLAIMER.md` |
| Frontend behaviour | `docs/frontend.md`, `frontend/*.js`, `deploy/web/api.py` (`frontend-settings`) |
| Public Make targets | the `help` block in `Makefile`, `docs/README.md` "Supported commands", `README.md`/`README_zh.md`, `.github/CONTRIBUTING.md` |
| Runtime configuration | `.env.example`, `deploy/config.py`, `deploy-config/profiles/*.yaml` |
| API routes and schemas | `deploy/web/api.py`, `docs/openapi.json` (regenerated, not hand-edited), `frontend/app.js`, `frontend/login.js` |
| Cross-file dependencies | `docs/contracts.md` |
| Observed landmines | `docs/pitfalls.md` |
| Release history | `CHANGELOG.md` (append; never rewrite past entries) |

### Style rules

- Explain deployment modes in networking terms: loopback (`127.0.0.1`),
  private IP, public IP, DNS, NAT, and reverse tunnel (ngrok/frp/cloudflared).
  Avoid vendor or cloud jargon such as "edge" or "CDN" where a concrete
  explanation works. The canonical explanation lives in `docs/deployment.md`;
  keep every other file consistent with it.
- Every relative Markdown link must resolve. The link test covers `README.md`,
  `README_zh.md`, `AGENTS.md`, `SECURITY.md`, `DISCLAIMER.md`, `docs/*.md`, and
  `.github/*.md`.
- Keep `README.md` and `README_zh.md` in sync.
- `docs/openapi.json` is generated at startup; regenerate it by running the
  service rather than hand-editing.
- `.env.example` is a template with no real secrets; real values live in the
  gitignored `.env`.

### Tests that pin documentation

- `tests/deployment/test_documentation_links.py` — links resolve, no stale
  links to the removed knowledge directory, and the frontend links only what
  Nginx serves.
- `tests/deployment/test_deployment_contract.py` — Makefile, Compose,
  Dockerfile, and profile contracts.
- `tests/api/` — the HTTP behaviour that the frontend and deployment docs
  describe.

After any documentation edit, run at least:

```text
.venv/bin/python -m pytest tests/deployment/test_documentation_links.py
```

---

## Done means

- The reproduction command now shows the intended result.
- `make preflight` passes, or every skipped check is named with its reason.
- `docs/` and this file still describe the behaviour that exists.
- No secret, endpoint, or machine path entered a tracked file.
- Nothing was committed or pushed.
