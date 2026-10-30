"""Metadata filtering with deliberately simple semantics.

A filter is `{key: value | [values]}`. A chunk matches when, for EVERY key, at least one filter
value appears in the chunk's metadata for that key (which may itself be a scalar or a list).
Comparison is case-insensitive. Unknown keys never match, so a typo returns nothing instead of
silently disabling the filter.
"""

from __future__ import annotations

from typing import Any

Filters = dict[str, "str | list[str]"]


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value}
    return {str(value).strip().lower()}


def matches_filters(metadata: dict[str, Any], filters: Filters | None) -> bool:
    if not filters:
        return True
    return all(_as_set(metadata.get(key)) & _as_set(wanted) for key, wanted in filters.items())
