# Security Policy

## Supported versions

Security fixes are targeted at the latest version on the default branch. Older
releases may not receive fixes.

## Reporting a vulnerability

Please do not open a public issue for an exploitable vulnerability. Report it
privately through the repository's security advisories or contact the
maintainer listed in the repository profile. Include:

- affected version or commit;
- deployment mode (host, Docker, or Podman);
- reproducible steps or a minimal proof of concept;
- logs with secrets, tokens, personal data, and private media removed;
- the potential impact and any suggested mitigation.

You should receive an acknowledgement within seven days. After validation,
the maintainer will coordinate a fix, disclosure timeline, and credit if the
reporter wishes to be acknowledged.

## Scope and safe testing

Only test systems and media for which you have permission. Do not upload
private or sensitive media to a public issue, demo service, or third-party
checkpoint host. The API is intended for trusted deployments and does not
provide authentication, authorization, rate limiting, or malware scanning by
default; add those controls before exposing it to an untrusted network.

## Dependency and model security

Keep Python dependencies and base images updated, run the repository's
`pip-audit`/SBOM checks, and treat downloaded checkpoints and datasets as
untrusted input. Verify their source and integrity before loading or serving
them.
