"""Standard-library logging with credential redaction.

The SDK emits to the ``typesafe_sdk`` logger and never configures handlers or levels beyond an
optional convenience: set ``TYPESAFE_LOG_LEVEL`` (``debug``/``info``/...) and the level is applied
once at import. Otherwise configure the ``typesafe_sdk`` logger through standard logging as usual.
"""

import json
import logging
import os
import re
from collections.abc import Mapping

from typing_extensions import override

from typesafe_sdk._core.constants import LOGGER_NAME, SECRET_HEADERS
from typesafe_sdk.constants import LOG_LEVEL_ENV

logger = logging.getLogger(LOGGER_NAME)


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "off": logging.CRITICAL + 1,
}


def _is_secret(name: str) -> bool:
    lowered = name.lower()
    return lowered in SECRET_HEADERS or "token" in lowered or "secret" in lowered


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {name: "***" if _is_secret(name) else value for name, value in headers.items()}


def redact_exception(error: BaseException, headers: Mapping[str, str]) -> BaseException:
    """Copy exception messages and chains with credentials masked, without requests or traceback frames."""
    credentials = {value for name, value in headers.items() if _is_secret(name) and value}
    for name, value in headers.items():
        if name.lower() in {"authorization", "proxy-authorization"}:
            match value.split(maxsplit=1):
                case [_, credential]:
                    credentials.add(credential)
    # Errors may contain raw tokens, repr-style header bytes, or JSON-escaped values.
    variants = {variant for value in credentials for variant in (value, repr(value)[1:-1], repr(value.encode())[2:-1], json.dumps(value)[1:-1])}
    pattern = re.compile("|".join(re.escape(value) for value in sorted(variants, key=len, reverse=True))) if variants else None

    def redact(message: str) -> str:
        return pattern.sub("***", message) if pattern is not None else message

    copies: dict[int, BaseException] = {}

    def copy_exception(original: BaseException) -> BaseException:
        if id(original) in copies:
            return copies[id(original)]
        message = redact(str(original))
        try:
            result = type(original)(message)
        except Exception:  # noqa: BLE001 - Custom exception constructors may require more than a message.
            result = Exception(f"{type(original).__name__}: {message}")
        copies[id(original)] = result
        if original.__cause__ is not None:
            result.__cause__ = copy_exception(original.__cause__)
        if original.__context__ is not None:
            result.__context__ = copy_exception(original.__context__)
        result.__suppress_context__ = original.__suppress_context__
        notes = getattr(original, "__notes__", None)
        if notes is not None:
            result.__dict__["__notes__"] = [redact(str(note)) for note in notes]
        return result

    return copy_exception(error)


class SensitiveHeadersFilter(logging.Filter):
    """Redact credential-bearing headers from a record before it reaches any handler.

    Log calls pass headers as the ``headers`` key of a mapping-style ``args``; this filter is the
    single redaction point, so no call site can leak a secret header.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, dict):
            headers = args.get("headers")
            if isinstance(headers, dict):
                record.args = {**args, "headers": _redact(headers)}
        return True


def setup_logging() -> None:
    """Apply ``TYPESAFE_LOG_LEVEL`` to the ``typesafe_sdk`` logger if it names a known level."""
    level = (os.environ.get(LOG_LEVEL_ENV) or "").strip().lower()
    if level in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[level])


logger.addHandler(logging.NullHandler())
logger.addFilter(SensitiveHeadersFilter())
setup_logging()
