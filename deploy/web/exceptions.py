"""Deployment business errors and their stable public codes.

Domain code has no FastAPI dependency. Business error codes use values from
1000 upward so they cannot be confused with HTTP status codes (100-599).
Messages may change, while codes and existing HTTP statuses remain stable.
Low-level Python and third-party exceptions retain their native types.
"""

from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    """Stable public business error codes, intentionally outside HTTP 100-599."""

    INTERNAL_ERROR = 1000
    CONFIGURATION_ERROR = 1001
    MODEL_NOT_FOUND = 1100
    VALIDATION_ERROR = 1200
    ACCOUNT_VALIDATION_ERROR = 1201
    ACCOUNT_ALREADY_EXISTS = 1202
    ACCOUNT_NOT_FOUND = 1203
    AUTH_REQUIRED = 1300
    INVALID_CREDENTIALS = 1301
    ORIGIN_REJECTED = 1302
    RATE_LIMITED = 1303
    NOT_FOUND = 1400
    ARTIFACT_NOT_FOUND = 1401
    METHOD_NOT_ALLOWED = 1402
    HTTP_ERROR = 1403
    FEATURE_UNAVAILABLE = 1500
    MEDIA_VALIDATION_ERROR = 1501
    UPLOAD_TOO_LARGE = 1502
    UPLOAD_TIMEOUT = 1503
    UPLOAD_TYPE_UNSUPPORTED = 1504
    INFERENCE_FAILED = 1600
    VISUALIZATION_FAILED = 1601


class DeepVQAError(Exception):
    """Base for expected deployment errors with a safe public message."""

    code = ErrorCode.INTERNAL_ERROR
    http_status_code = 500

    def __init__(self, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.headers = headers


class ConfigurationError(DeepVQAError):
    """Deployment or checkpoint configuration is invalid."""

    code = ErrorCode.CONFIGURATION_ERROR
    http_status_code = 503


class ModelNotFoundError(DeepVQAError):
    """A requested model or checkpoint is not available."""

    code = ErrorCode.MODEL_NOT_FOUND
    http_status_code = 404


class ValidationError(DeepVQAError, ValueError):
    """Invalid business input; also accepted by existing CLI ValueError handlers."""

    code = ErrorCode.VALIDATION_ERROR
    http_status_code = 422


class AccountValidationError(ValidationError):
    """An email address or password does not satisfy account policy."""

    code = ErrorCode.ACCOUNT_VALIDATION_ERROR


class AccountAlreadyExistsError(ValidationError):
    """The canonical email is already registered (historically HTTP 422)."""

    code = ErrorCode.ACCOUNT_ALREADY_EXISTS


class AccountNotFoundError(ValidationError):
    """An account-management operation references an unknown account."""

    code = ErrorCode.ACCOUNT_NOT_FOUND


class AuthenticationRequiredError(DeepVQAError):
    """The request has no valid authenticated session."""

    code = ErrorCode.AUTH_REQUIRED
    http_status_code = 401


class InvalidCredentialsError(AuthenticationRequiredError):
    """Credentials are invalid, without distinguishing unknown accounts."""

    code = ErrorCode.INVALID_CREDENTIALS


class OriginRejectedError(DeepVQAError):
    """A cookie-authenticated state change came from an untrusted origin."""

    code = ErrorCode.ORIGIN_REJECTED
    http_status_code = 403


class RateLimitError(DeepVQAError):
    """A login or registration attempt window is exhausted."""

    code = ErrorCode.RATE_LIMITED
    http_status_code = 429


class ResourceNotFoundError(DeepVQAError):
    """A resource or endpoint is unavailable in this deployment mode."""

    code = ErrorCode.NOT_FOUND
    http_status_code = 404


class ArtifactNotFoundError(ResourceNotFoundError):
    """An artifact is absent or inaccessible; both cases have one response."""

    code = ErrorCode.ARTIFACT_NOT_FOUND


class FeatureUnavailableError(ValidationError):
    """The deployment mode or selected model does not support the operation."""

    code = ErrorCode.FEATURE_UNAVAILABLE


class MediaValidationError(ValidationError):
    """Uploaded media is present but not acceptable."""

    code = ErrorCode.MEDIA_VALIDATION_ERROR


class UploadLimitError(DeepVQAError):
    """An upload exceeds the configured byte limit."""

    code = ErrorCode.UPLOAD_TOO_LARGE
    http_status_code = 413


class UploadTimeoutError(DeepVQAError):
    """Reading an upload took too long (historically HTTP 413)."""

    code = ErrorCode.UPLOAD_TIMEOUT
    http_status_code = 413


class UploadTypeError(MediaValidationError):
    """Upload file magic or media type is unsupported."""

    code = ErrorCode.UPLOAD_TYPE_UNSUPPORTED


class InferenceError(DeepVQAError):
    """Inference could not complete; the underlying cause stays in server logs."""

    code = ErrorCode.INFERENCE_FAILED


class VisualizationError(DeepVQAError):
    """Visualization could not complete; the cause stays in server logs."""

    code = ErrorCode.VISUALIZATION_FAILED


__all__ = [
    "AccountAlreadyExistsError",
    "AccountNotFoundError",
    "AccountValidationError",
    "ArtifactNotFoundError",
    "AuthenticationRequiredError",
    "ConfigurationError",
    "DeepVQAError",
    "ErrorCode",
    "FeatureUnavailableError",
    "InferenceError",
    "InvalidCredentialsError",
    "MediaValidationError",
    "ModelNotFoundError",
    "OriginRejectedError",
    "RateLimitError",
    "ResourceNotFoundError",
    "UploadLimitError",
    "UploadTimeoutError",
    "UploadTypeError",
    "ValidationError",
    "VisualizationError",
]
