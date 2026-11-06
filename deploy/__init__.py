"""Deep-VQA deployment service.

Layout by concern:

- ``deploy.config`` — instance configuration (``.env`` + deployment profile
  YAML + resolved paths + runtime directories).
- ``deploy.web`` — HTTP service: routes, wire schemas, accounts, evaluation
  store, uploads, and domain exceptions.
- ``deploy.inference`` — model serving: loading, preprocessing, prediction,
  technical-quality heuristics, and the batch CLI.

``deploy/`` may import ``src/``; ``src/`` must not import ``deploy/``.
"""
