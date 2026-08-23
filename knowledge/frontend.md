# Frontend Guide

The project uses one frontend for internal, public, and local development deployments. The backend controls authorization and storage capabilities. In production the browser normally reaches an edge Nginx/TLS or SSO gateway; it does not connect directly to a database or model host.

| Backend policy | Frontend behavior |
| --- | --- |
| `auth.mode: api_key` | Shows the API-key login panel before the workspace. |
| `auth.mode: none` | Opens the workspace directly; use only for trusted local development. |
| `evaluation_store.backend: sqlite` | Shows the current browser session's history and export controls. |
| `evaluation_store.backend: none` | History is empty and durable visualization artifacts are disabled. |

## Internal login

The login is intentionally a deployment-level shared key rather than user
registration. This keeps the internal single-instance mode simple for demos.
If an organization already has multi-user identity infrastructure, it may place
an external OIDC/SAML/SSO gateway in front of this service. That gateway is
outside this project's implementation scope.

```yaml
# deploy-config/profiles/infer_deploy.internal.yaml
auth:
  mode: api_key
evaluation_store:
  backend: sqlite
```

```env
# .env
DEEP_VQA_API_KEY=replace-with-a-long-random-secret
DEEP_VQA_AUTH_SECRET=replace-with-a-different-long-random-secret
```

Run `make docker-infer` or `make docker-infer-internal-ollama`, then open
`http://127.0.0.1:8000/`. The same page shows the login panel and unlocks the workspace after authentication.
Paste the value of `DEEP_VQA_API_KEY` from `.env` into the **API key** field
and click **Unlock**. If the field is missing, check that `auth.mode` is
`api_key`, both secrets are present in `.env`, and the stack was rebuilt.

The login panel and the workspace intentionally use the same URL. The frontend
is a single static page: authentication initially hides the workspace, then a
successful login enables it in place and keeps the HttpOnly cookie for later
API requests. It does not navigate to a separate `/login` route. If an older
browser cache appears to keep the page at an unexpected scroll position, use a
hard refresh after recreating the Nginx container.

## Public stateless mode

Set `evaluation_store.backend: none` to disable SQLite history persistence. Keep API-key or reverse-proxy authentication enabled for public deployments.

See the [deployment guide](deployment.md) and [Ollama guide](ollama.md).

## Frontend implementation notes

Nginx serves this directory at `/` and proxies `/api/` to FastAPI. When FastAPI
runs directly, serve the directory separately with
`python -m http.server 5500 -d frontend` and configure `CORS_ALLOW_ORIGINS`.
The page also supports keyboard file selection, drag and drop, reduced-motion
preferences, media preview, bilingual labels, model interpretation, and JSONL
export of the current session's history. Feature visualization keeps durable
artifacts under `reports/iqa-test/`, shows a reduced-size source thumbnail next
to the generated maps and Grad-CAM, and opens any comparison image on double
click. Formal accessibility conformance
still requires Lighthouse/axe and assistive-technology testing.
