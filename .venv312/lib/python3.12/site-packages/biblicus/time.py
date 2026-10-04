"""
Time utilities for Biblicus.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """
    Return the current Coordinated Universal Time as an International Organization for Standardization 8601 string.

    :return: Current Coordinated Universal Time timestamp in International Organization for Standardization 8601 format.
    :rtype: str
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
