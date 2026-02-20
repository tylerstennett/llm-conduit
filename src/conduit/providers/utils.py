from __future__ import annotations

from typing import Any


def drop_nones(value: Any) -> Any:
    """Recursively strip ``None`` values from dicts and lists.

    Args:
        value: A dict, list, or scalar value.

    Returns:
        A copy of *value* with all ``None`` entries removed from dicts.
    """
    if isinstance(value, dict):
        return {
            key: drop_nones(inner)
            for key, inner in value.items()
            if inner is not None
        }
    if isinstance(value, list):
        return [drop_nones(item) for item in value]
    return value
