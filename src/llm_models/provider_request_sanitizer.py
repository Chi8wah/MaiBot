"""Utilities for sanitizing provider request snapshots before writing them to disk."""

from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import base64

PROVIDER_REQUEST_SECRET_MARKERS = ("authorization", "api_key", "apikey", "token", "secret", "cookie", "password")
PROVIDER_REQUEST_OMITTED_KEYS = {"extra_headers", "extra_query"}


def _json_friendly_provider_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return sanitize_provider_request_snapshot(model_dump(mode="json", exclude_none=True))
        except TypeError:
            return sanitize_provider_request_snapshot(model_dump(exclude_none=True))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return sanitize_provider_request_snapshot(to_dict())

    return str(value)


def sanitize_provider_request_snapshot(provider_request: Any) -> Any:
    """Omit transport metadata and redact secret-like fields in a provider request snapshot."""
    if isinstance(provider_request, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in provider_request.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in PROVIDER_REQUEST_OMITTED_KEYS:
                continue
            if any(marker in normalized_key for marker in PROVIDER_REQUEST_SECRET_MARKERS):
                sanitized[str(key)] = "<redacted>"
            else:
                sanitized[str(key)] = sanitize_provider_request_snapshot(item)
        return sanitized

    if isinstance(provider_request, Sequence) and not isinstance(provider_request, (bytes, bytearray, str)):
        return [sanitize_provider_request_snapshot(item) for item in provider_request]

    return _json_friendly_provider_value(provider_request)
