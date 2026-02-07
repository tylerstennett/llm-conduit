from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from conduit.config.base import BaseLLMConfig


class OpenRouterProviderPrefs(BaseModel):
    """OpenRouter provider routing preferences."""

    model_config = ConfigDict(extra="forbid")

    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    data_collection: Literal["allow", "deny"] | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    sort: str | None = None


class OpenRouterConfig(BaseLLMConfig):
    """Runtime request configuration for OpenRouter /chat/completions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = Field(..., min_length=1)

    models: list[str] | None = None
    n: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)

    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logit_bias: dict[str, float] | None = None

    response_format: dict[str, Any] | None = None
    provider_prefs: OpenRouterProviderPrefs | None = Field(
        default=None,
        alias="provider",
        serialization_alias="provider",
    )
    route: Literal["fallback", "sort"] | None = None

    metadata: dict[str, str] | None = None
    plugins: list[dict[str, Any]] | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None

    app_name: str | None = None
    app_url: str | None = None

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("api_key is required for OpenRouter")
        return stripped
