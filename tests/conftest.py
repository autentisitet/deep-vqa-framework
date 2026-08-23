"""Test-process defaults for deployment configuration loading."""

import os


os.environ.setdefault(
    "DEEP_VQA_DEPLOYMENT_CONFIG",
    "deploy-config/profiles/infer_deploy.internal.yaml",
)
