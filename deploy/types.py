"""Closed deployment choices shared by configuration and HTTP schemas."""

from enum import StrEnum
from typing import Literal


class AuthMode(StrEnum):
    NONE = "none"
    API_KEY = "api_key"
    ACCOUNTS = "accounts"


EvaluationStoreBackend = Literal["sqlite", "none"]
