"""
Shared test-only value builders for secret-like fields in fixtures.
"""

from __future__ import annotations


def build_test_value(*parts: str) -> str:
    """
    Build a deterministic test value from string parts.

    :param parts: Ordered value segments.
    :type parts: str
    :return: Hyphen-joined value.
    :rtype: str
    """
    return "-".join(part for part in parts if part)


def build_test_openai_api_key() -> str:
    """
    Build the standard OpenAI test API key value.

    :return: Generated test API key value.
    :rtype: str
    """
    return build_test_value("test", "openai", "token")
