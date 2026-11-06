# Frontend Guide

The project uses one frontend for internal, proxy, and local development
deployments. The backend reports which features are enabled. In a network
deployment, the browser connects to the bundled Nginx, which forwards `/api/`
to the model process; the browser never talks to the model process or database
directly.

| Backend setting | Frontend behavior |
| --- | --- |
| `auth.mode: accounts` | Redirects unauthenticated visitors to `/login.html`, which shows email + password fields and a create-account toggle. |
| `auth.mode: api_key` | Redirects unauthenticated visitors to `/login.html`. |
| `auth.mode: none` | Opens the workspace directly; use only on localhost or when a process in front of it already authenticates users. |
| `evaluation_store.backend: sqlite` | Evaluation history and visualization PNG files are saved on the server. |
| `evaluation_store.backend: none` | No evaluation history or visualization PNG files are saved on the server. |

API schemas are generated from `deploy/web/schemas.py` into
[OpenAPI](openapi.json). Authentication modes (`none`, `api_key`, `accounts`),
task types (`iqa`, `vqa`), and media types (`image`, `video`) expose their allowed
values in the schema. History items have explicit fields, including a two-number
`mos_interval` array; session identifiers are not exposed. Field names and JSON
string values are unchanged.

## Internal login

Internal mode issues per-person accounts: visitors register with an email and
password, and a session cookie keeps them signed in. Passwords are hashed with
scrypt; session tokens are random and stored only as digests. There are no roles
or organisation management. If an organisation already has a company login
system, a process in front of Deep-VQA can authenticate users before forwarding
requests instead; that company login system is outside this project's scope.

```yaml
# deploy-config/profiles/infer_deploy.internal.yaml
auth:
  mode: accounts
evaluation_store:
  backend: sqlite
```

Run `make docker-infer`, then open
`http://127.0.0.1:8000/`. In internal mode the browser navigates to
`/login.html` when unauthenticated; successful login returns to the workspace.
Internal mode uses `auth.mode: accounts`, so the page shows email and password
fields plus a "create account" toggle. No shell command is involved: the first
visitor registers through that form. `DEEP_VQA_API_KEY` and `DEEP_VQA_AUTH_SECRET`
are only used by the legacy `auth.mode: api_key` deployment, which shows a single
API-key field instead.

In the default internal mode, the login page first requests `/api/v1/auth/status`
and reveals the **create account** button when the server reports `accounts`.
The registration request does not require an existing session; it creates the
account, sets a new HttpOnly session cookie, and returns HTTP 201. A new user
can therefore register from the browser without an administrator provisioning
an account. Open registration is rate-limited, so use only on trusted networks
or behind the authenticated front process described in the deployment guide.

The login page and workspace are separate static documents. The login endpoint
has both Nginx request limiting and an in-process per-client failed-attempt
window. State-changing cookie-authenticated requests validate the Origin when it
is present. The HttpOnly cookie
is used for later API requests. Proxy mode skips the application login page and
opens the workspace directly.

---

## Proxy mode

Internal mode uses SQLite history and saves visualization PNG files. Proxy mode
uses `evaluation_store.backend: none`; the browser does not show server-side
history or export controls, and no SQL database is required.
Use `DEPLOYMENT_MODE=proxy` only when a separate process in front of Deep-VQA
owns the reachable IP, terminates TLS, and authenticates users. Network clients
must not connect directly to the FastAPI port. See the [deployment
guide](deployment.md) for the full IP/DNS/NAT path.

See the [deployment guide](deployment.md).

---

## Saved visualization files

Feature maps and Grad-CAM are generated as PNG files under
`reports/iqa-test/<visualization-id>/`. The API response returns file paths,
and the browser fetches those files in later authenticated requests. In the
internal Compose configuration, the reports directory is mounted from the host, so
those files survive the original request and container recreation. SQLite does
not store the images; it only stores evaluation-history records.

Artifacts are also scoped to their owner. In accounts mode the owner is the
account, so signing in from another browser still exposes your own artifacts. In
the legacy `api_key` mode the server instead issues an anonymous HttpOnly session
cookie and records its digest next to the artifact; a different session then gets
404, and clearing cookies loses access to that earlier history. The artifact id
is a random token rather than a timestamp, so a path cannot be derived from the
time it was created.

Artifact directories expire after `artifacts.retention_seconds` (seven days by
default). Expired `viz-*` directories are removed when the next visualization is
created, never at startup, so restarting a container never deletes data. Other
files under the same root, including CLI report output, are not touched.

Proxy mode disables both history and saved visualization files. These are
separate functions even though the current configuration enables or disables
them together. A future configuration can control them independently, but
enabling the current visualization endpoint in proxy mode would make that mode
write files on the server.

---

## Authentication boundary

In `auth.mode: accounts`, the login page is the browser entry for the
single-instance account store. It supports registration and login, but it does
not provide roles, organisation management, email verification, password
recovery, or single sign-on. Accounts are used to scope evaluation history and
generated artifacts to a person; this is not a multi-tenant identity system.

The legacy `auth.mode: api_key` profile uses a deployment-wide API key and a
browser session cookie. It remains available for compatibility, but it is not
the default internal configuration and must not be described as per-user
authentication. In that mode, artifact access is additionally scoped to the
anonymous browser session because all visitors share the same deployment key.

The page is unnecessary when the service is restricted to one developer's
loopback interface or when a process in front already authenticates users. In
those environments, application authentication can be disabled in the selected
YAML file. Do not remove the login check while also exposing Nginx to a LAN or
the Internet without another authentication method.

---

## Frontend implementation notes

Bundled Nginx serves `frontend/` at `/` and forwards `/api/v1/*` to the
private Uvicorn/FastAPI service at `/v1/*`. The workspace and login page use a
fixed same-origin API base, `/api/v1`; neither a query parameter, browser storage,
nor an environment variable can redirect browser API traffic to another host.
CORS is not enabled. The page uses browser navigation to `/login.html` and
back to `/`, not HTTP 3xx redirects. API URLs are exact: a trailing slash
on a registered `/v1` path returns 404 rather than FastAPI's implicit 307.
Origin checks still reject cross-origin
cookie-authenticated state changes. When a separate TLS/authentication process
stands in front of the deployment, it forwards the browser-facing origin through
the bundled Nginx on loopback.

The frontend is split into `index.html`, `login.html`, `styles.css`,
`visualization-overrides.css`, `app.js`, and `login.js`. At startup it reads
`/api/v1/frontend-settings`; the response reports whether login, history, and
visualization are enabled. Proxy mode hides the application logout action,
server-side history, export controls, and saved visualization controls. It
supports keyboard file selection, drag and drop, reduced-motion preferences,
bilingual labels, response visualisation, and JSONL export of the current
session's history. Feature visualization keeps PNG files under
`reports/iqa-test/`, shows an original-versus-feature-map row and an
original-versus-Grad-CAM row, and opens comparison images on double click.
Formal accessibility conformance still requires Lighthouse/axe and
assistive-technology testing.

The frontend intentionally uses browser-native HTML, CSS, and JavaScript. There
is no frontend package manager, dependency tree, transpilation, or minification
step: Nginx serves the reviewed source files directly. Keep those files
human-readable and treat them as deployable source rather than generated build
files.


---

## API errors

Application errors use one JSON response shape:

```json
{
  "code": 1502,
  "error": "Uploaded file exceeds the configured byte limit",
  "detail": null
}
```

`code` is a stable numeric business identifier. All business codes start at
1000 and therefore never occupy the HTTP status-code range 100-599. `error` is
the human-readable message; `detail` contains optional validation context. The
workspace and login page retain the business `code` separately from the HTTP
`status` on their JavaScript Error objects. A non-JSON response has no business
code; its HTTP status remains available through `error.status`. HTTP response
statuses continue to follow the HTTP protocol. Request-validation details contain
field locations and messages, without echoing submitted values such as passwords.

| Code | Symbol | HTTP status | Meaning |
| ---: | --- | ---: | --- |
| 1000 | `INTERNAL_ERROR` | 500 | Unexpected server failure; response hides internal details |
| 1001 | `CONFIGURATION_ERROR` | 503 | Runtime configuration is invalid |
| 1100 | `MODEL_NOT_FOUND` | 404 | Requested model is not configured |
| 1200 | `VALIDATION_ERROR` | 422 | Invalid request structure or missing business input |
| 1201 | `ACCOUNT_VALIDATION_ERROR` | 422 | Invalid email address or password policy violation |
| 1202 | `ACCOUNT_ALREADY_EXISTS` | 422 | Registration conflicts with an existing canonical email |
| 1203 | `ACCOUNT_NOT_FOUND` | 422 | Account-management target does not exist |
| 1300 | `AUTH_REQUIRED` | 401 | No valid authenticated session |

| 1301 | `INVALID_CREDENTIALS` | 401 | Invalid credentials without account enumeration |
| 1302 | `ORIGIN_REJECTED` | 403 | Cookie-authenticated state change from an untrusted origin |
| 1303 | `RATE_LIMITED` | 429 | Login or registration window exhausted |
| 1400 | `NOT_FOUND` | 404 | Resource or route is unavailable |
| 1401 | `ARTIFACT_NOT_FOUND` | 404 | Artifact missing or owned by someone else |
| 1402 | `METHOD_NOT_ALLOWED` | 405 | HTTP method is unsupported for the route |
| 1403 | `HTTP_ERROR` | varies | Other framework-generated HTTP errors |
| 1500 | `FEATURE_UNAVAILABLE` | 422 | Selected deployment mode or model cannot perform the operation |
| 1501 | `MEDIA_VALIDATION_ERROR` | 422 | Media input cannot be processed |
| 1502 | `UPLOAD_TOO_LARGE` | 413 | Upload byte limit exceeded |
| 1503 | `UPLOAD_TIMEOUT` | 413 | Upload read timed out; historical status retained |
| 1504 | `UPLOAD_TYPE_UNSUPPORTED` | 422 | Unsupported file magic or model/media mismatch |
| 1600 | `INFERENCE_FAILED` | 500 | Inference failed; underlying exception remains in server logs |
| 1601 | `VISUALIZATION_FAILED` | 500 | Visualization failed; underlying exception remains in server logs |

The API sends `WWW-Authenticate: Session realm="Deep-VQA"` with its 401
responses. The browser handles the JSON error and navigates to the login page;
this header does not enable HTTP Basic authentication or display a Basic popup.

Nginx-generated errors, such as 414 for a long URI or 429 for an API request
burst, may use Nginx's own response format; the browser handles these through the HTTP status
without inventing a business error code.

Business exception types and codes live in
[`deploy/web/exceptions.py`](../deploy/web/exceptions.py); HTTP conversion lives
in [`deploy/web/api.py`](../deploy/web/api.py). Account-validation exceptions
remain subclasses of `ValueError` so account CLI parsing and existing callers
keep working. Low-level filesystem, numerical, and third-party exceptions keep
their native types; API operation boundaries translate them when appropriate.
`ConfigurationError` (`CONFIGURATION_ERROR`, 503) and `AccountNotFoundError`
(`ACCOUNT_NOT_FOUND`, compatible with `ValueError`) are available to domain
callers; startup configuration failures remain startup errors, not HTTP responses.
