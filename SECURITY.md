# Security Policy

## Supported versions

Security fixes are targeted at the latest version on the default branch. Older
releases may not receive fixes.

---

## Reporting a vulnerability

Please do not open a public issue for an exploitable vulnerability. Report it
privately through the repository's security advisories or contact the
maintainer listed in the repository profile. Include:

- affected version or commit;
- deployment mode (internal or proxy) and container runtime (Docker or Podman);
- reproducible steps or a minimal proof of concept;
- logs with secrets, tokens, personal data, and private media removed;
- the potential impact and any suggested mitigation.

You should receive an acknowledgement within seven days. After validation,
the maintainer will coordinate a fix, disclosure timeline, and credit if the
reporter wishes to be acknowledged.

---

## Scope and safe testing

Only test systems and media for which you have permission. Do not upload
private or sensitive media to a public issue, demo service, or third-party
checkpoint host.

Authentication and rate limiting depend on the selected deployment mode:

- **Internal mode (the default)** requires an account (email + password,
  scrypt-hashed), applies Nginx request limits, and limits repeated failed
  logins per client. It does not terminate TLS, and there is no email
  verification or self-service password recovery; an operator with shell access
  can reset a password from inside the container.
- **Proxy mode** disables application login by design and expects a reverse
  proxy in front of it to provide TLS, authentication, and rate limiting. That
  proxy must remove untrusted forwarding headers and set `X-Forwarded-Proto`
  after TLS termination. Forward to bundled Nginx on loopback port 8000; do not
  expose or forward to the Uvicorn diagnostic port 8001.

The frontend code and requests are observable and editable by each browser
user. A modified page, forged `X-Deep-VQA-Client` header, or handcrafted HTTP
request must not grant access: authorization, media validation, path containment,
and artifact ownership checks execute in FastAPI. The frontend does not hold a
secret. Same-origin policy and absent CORS do not prevent direct requests from
scripts; Nginx limits and backend authentication are the enforcement boundary.
Never rely on hidden buttons or omitted OpenAPI paths as authorization.

Neither mode scans uploaded media for malware. Internal accounts mode restricts
saved evaluation history and artifacts to their owner; the legacy shared API
key is a deployment token, not a per-user identity. Proxy mode disables app
authentication and requires the front process to enforce access control. Do
not expose either mode to an untrusted network without appropriate protection.

---

## DDoS and network exposure

This repository does **not** provide DDoS protection. Nginx request limits,
connection limits, upload-size limits, and application login throttling are
local resource controls; they can reduce ordinary abuse but cannot absorb a
volumetric attack, upstream bandwidth exhaustion, connection-table exhaustion,
or a distributed layer-7 attack.

For any network-facing deployment, place a reverse proxy, load balancer, or
DDoS-filtering service you operate in front of the host. That front process
should terminate TLS, enforce client and request limits, and reject unwanted
traffic before it reaches the host; restrict the origin so clients cannot
bypass it. Keep the FastAPI port private and bind the bundled Nginx port to
localhost unless that front process is already in place.

Nginx rejects overly long request lines before FastAPI, caps upload size,
and returns 429 for local request-rate and connection-limit breaches; these
proxy responses need not use the API JSON business error format. The loopback
Uvicorn diagnostic port bypasses those limits and must not be exposed.

The bundled Compose stack is suitable for local use, trusted internal networks,
and controlled demonstrations. It is not an Internet-facing DDoS mitigation
layer.

---

## Dependency and model security

Keep Python dependencies and base images updated. Repository automation uses
GitHub security features and CI checks; treat downloaded checkpoints and datasets
as untrusted input and verify their source and integrity before loading or serving
them.
