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


def extract_openai_message_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None
