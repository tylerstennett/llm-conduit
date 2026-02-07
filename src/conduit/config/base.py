from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseLLMConfig(BaseModel):
    """Base configuration shared by all providers."""

    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    api_key: str | None = None

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | str | None = None
    seed: int | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model must not be empty")
        return stripped

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("base_url must not be empty")
        return stripped.rstrip("/")
