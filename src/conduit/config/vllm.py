from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from conduit.config.base import BaseLLMConfig


class VLLMStructuredOutputs(BaseModel):
    """vLLM structured output controls."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    json_schema: dict[str, Any] | str | None = Field(
        default=None,
        alias="json",
        serialization_alias="json",
    )
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None

    @model_validator(mode="after")
    def validate_exclusive(self) -> "VLLMStructuredOutputs":
        constraint_count = sum(
            [
                self.json_schema is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
            ]
        )
        if constraint_count > 1:
            raise ValueError(
                "only one structured output constraint may be set at a time"
            )
        return self


class VLLMConfig(BaseLLMConfig):
    """Runtime request configuration for vLLM chat completions."""

    base_url: str = "http://localhost:8000/v1"

    n: int | None = Field(default=1, ge=1)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=-1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    response_format: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None

    use_beam_search: bool = False
    top_k: int | None = Field(default=None, ge=-1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    length_penalty: float | None = None

    stop_token_ids: list[int] | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = Field(default=0, ge=0)
    truncate_prompt_tokens: int | None = Field(default=None, ge=-1)
    prompt_logprobs: int | None = Field(default=None, ge=-1)

    echo: bool = False
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = True

    structured_outputs: VLLMStructuredOutputs | None = None

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> "VLLMConfig":
        if self.continue_final_message and self.add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt cannot both be true"
            )
        return self


class VLLMServerConfig(BaseModel):
    """Optional vLLM launch-time settings, not sent in chat requests."""

    model_config = ConfigDict(extra="forbid")

    max_model_len: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    dtype: str | None = None
    quantization: str | None = None
