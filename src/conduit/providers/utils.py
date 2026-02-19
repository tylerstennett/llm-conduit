from __future__ import annotations

from typing import Any


def drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: drop_nones(inner)
            for key, inner in value.items()
            if inner is not None
        }
    if isinstance(value, list):
        return [drop_nones(item) for item in value]
    return value
