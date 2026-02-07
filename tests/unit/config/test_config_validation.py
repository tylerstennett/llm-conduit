from __future__ import annotations

import pytest
from pydantic import ValidationError

from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig, VLLMStructuredOutputs


def test_model_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="   ")


def test_temperature_range_validation() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="m", temperature=3.0)


def test_top_p_range_validation() -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(model="m", top_p=1.5)


def test_vllm_top_k_allows_negative_one() -> None:
    config = VLLMConfig(model="m", top_k=-1)
    assert config.top_k == -1


def test_vllm_accepts_response_format_and_stream_options() -> None:
    config = VLLMConfig(
        model="m",
        response_format={"type": "json_object"},
        stream_options={"include_usage": True},
    )
    assert config.response_format == {"type": "json_object"}
    assert config.stream_options == {"include_usage": True}


def test_vllm_rejects_unsupported_best_of_field() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="m", best_of=2)  # type: ignore[call-arg]


def test_ollama_top_k_requires_positive() -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(model="m", top_k=0)


def test_ollama_accepts_raw_and_suffix() -> None:
    config = OllamaConfig(model="m", raw=True, suffix=" tail")
    assert config.raw is True
    assert config.suffix == " tail"


def test_vllm_guided_aliases_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="m", guided_json={"type": "object"}, guided_regex="foo")


def test_vllm_structured_outputs_and_guided_alias_conflict() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(
            model="m",
            structured_outputs=VLLMStructuredOutputs(regex="foo"),
            guided_json={"type": "object"},
        )


def test_vllm_structured_outputs_constraints_are_exclusive() -> None:
    with pytest.raises(ValidationError):
        VLLMStructuredOutputs(regex="a", grammar="b")


def test_vllm_continue_final_message_conflict() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="m", continue_final_message=True, add_generation_prompt=True)


def test_ollama_mirostat_allowed_values() -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(model="m", mirostat=3)


def test_openrouter_api_key_required() -> None:
    with pytest.raises(ValidationError):
        OpenRouterConfig(model="model", api_key="")


def test_openrouter_accepts_reasoning_transforms_and_include() -> None:
    config = OpenRouterConfig(
        model="model",
        api_key="k",
        reasoning={"effort": "high"},
        transforms=["middle-out"],
        include=["reasoning"],
    )
    assert config.reasoning == {"effort": "high"}
    assert config.transforms == ["middle-out"]
    assert config.include == ["reasoning"]


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        VLLMConfig(model="m", unknown_field=123)  # type: ignore[call-arg]
