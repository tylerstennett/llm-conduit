from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from conduit.config.base import BaseLLMConfig


class OllamaConfig(BaseLLMConfig):
    """Runtime request configuration for Ollama /api/chat."""

    base_url: str = "http://localhost:11434"

    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(default=None, gt=0.0)
    repeat_last_n: int | None = None
    tfs_z: float | None = None
    typical_p: float | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None

    num_ctx: int | None = Field(default=None, ge=1)
    num_predict: int | None = None
    num_gpu: int | None = None
    num_thread: int | None = None
    num_keep: int | None = None
    num_batch: int | None = None

    keep_alive: str | int | float | None = None
    format: str | dict[str, Any] | None = None
    think: bool | Literal["high", "medium", "low"] | None = None
    raw: bool | None = None
    suffix: str | None = None

    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_ollama_constraints(self) -> "OllamaConfig":
        if self.mirostat is not None and self.mirostat not in {0, 1, 2}:
            raise ValueError("mirostat must be one of {0, 1, 2}")
        return self
