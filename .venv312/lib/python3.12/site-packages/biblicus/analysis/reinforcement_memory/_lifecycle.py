"""
Lifecycle tier derivation for topic clusters.

Classifies clusters as new, trending, or established based on the
distribution of member timestamps across short-, medium-, and long-term
time windows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

_SHORT_TERM_DAYS = 14
_MEDIUM_TERM_DAYS = 30


def derive_lifecycle(
    timestamps: List[str],
    now: Optional[datetime] = None,
    short_term_days: int = _SHORT_TERM_DAYS,
    medium_term_days: int = _MEDIUM_TERM_DAYS,
) -> Tuple[str, bool, bool, Optional[int]]:
    """
    Classify a cluster's lifecycle tier from its member timestamps.

    Looks at which time windows are represented among the timestamps:

    - **Short-term**: within ``short_term_days`` days.
    - **Medium-term**: between ``short_term_days`` and ``medium_term_days`` days.
    - **Long-term**: older than ``medium_term_days`` days.

    Returns a four-tuple ``(lifecycle_tier, is_new, is_trending, days_inactive)``:

    - ``lifecycle_tier``: ``"new"``, ``"trending"``, or ``"established"``.
    - ``is_new``: True when there are only short-term members.
    - ``is_trending``: True when there are recent members but no long-term ones.
    - ``days_inactive``: Days since the most recent timestamp, or None if no
      timestamps are provided.

    :param timestamps: ISO 8601 timestamp strings for each cluster member.
    :param now: Reference datetime (UTC).  Defaults to ``datetime.now(UTC)``.
    :param short_term_days: Age threshold (days) for short-term window.
    :param medium_term_days: Age threshold (days) for medium-term window.
    :return: ``(lifecycle_tier, is_new, is_trending, days_inactive)``
    """
    if now is None:
        now = datetime.now(timezone.utc)

    has_short = has_medium = has_long = False
    most_recent: Optional[datetime] = None

    for ts in timestamps:
        dt = _parse_timestamp(ts)
        if dt is None:
            continue

        if most_recent is None or dt > most_recent:
            most_recent = dt

        age_days = max(0, (now - dt).days)
        if age_days <= short_term_days:
            has_short = True
        elif age_days <= medium_term_days:
            has_medium = True
        else:
            has_long = True

    is_new = has_short and not has_medium and not has_long
    is_trending = (has_short or has_medium) and not has_long

    if is_new:
        lifecycle_tier = "new"
    elif is_trending:
        lifecycle_tier = "trending"
    else:
        lifecycle_tier = "established"

    days_inactive: Optional[int] = None
    if most_recent is not None:
        days_inactive = max(0, (now - most_recent).days)

    return lifecycle_tier, is_new, is_trending, days_inactive


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse an ISO 8601 timestamp string into an aware datetime.

    Supports strings with or without a UTC suffix (``Z`` or ``+00:00``).
    Returns None on parse failure.
    """
    if not ts or not isinstance(ts, str):
        return None
    # Normalize Z -> +00:00
    normalized = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
