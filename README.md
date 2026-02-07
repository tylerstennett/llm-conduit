# llm-conduit

Conduit is a provider-aware Python LLM client for vLLM, Ollama, and OpenRouter.

It exposes strict typed configs, provider-specific request mapping, normalized tool calling,
and a unified async streaming interface.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from conduit import Conduit, OllamaConfig, Message, Role

config = OllamaConfig(model="llama3.1:8b")

async with Conduit(config) as client:
    response = await client.chat(
        messages=[
            Message(role=Role.SYSTEM, content="You are concise."),
            Message(role=Role.USER, content="Explain gradient descent briefly."),
        ]
    )

print(response.content)
```

## Provider notes

### vLLM

- Uses `/chat/completions` under your configured base URL.
- Supports `structured_outputs` directly.
- Legacy `guided_json`, `guided_regex`, `guided_choice`, and `guided_grammar`
  are accepted and mapped to `structured_outputs` with deprecation warnings.
- Tool choice values `auto`, `none`, `required`, and named function objects are supported.

### Ollama

- Uses `/api/chat`.
- Sampling/runtime fields are sent in `options`.
- Tool choice supports `auto` and `none` semantics.
- Canonical `tool_call_id` values are synthesized as `ollama_call_{index}`.
- Tool result messages are converted to Ollama `tool_name` by matching prior assistant tool calls.

### OpenRouter

- Uses `https://openrouter.ai/api/v1/chat/completions` by default.
- `api_key` is required.
- Optional headers:
  - `HTTP-Referer` from `app_url`
  - `X-Title` from `app_name`
- Supports provider routing via `provider` and `route`.

## Strict config overrides

Per-request `config_overrides` must be nested dictionaries and must validate against
the active provider config model. Unknown keys raise `ConfigValidationError`.

```python
await client.chat(
    messages=[Message(role=Role.USER, content="Say hi")],
    config_overrides={"temperature": 0.1},
)
```

## from_env

```python
from conduit import Conduit

client = Conduit.from_env("openrouter")
```

Environment variables:

- vLLM: `CONDUIT_VLLM_MODEL`, `CONDUIT_VLLM_URL`, `CONDUIT_VLLM_API_KEY`
- Ollama: `CONDUIT_OLLAMA_MODEL`, `CONDUIT_OLLAMA_URL`, `CONDUIT_OLLAMA_API_KEY`
- OpenRouter: `CONDUIT_OPENROUTER_MODEL`, `CONDUIT_OPENROUTER_KEY`, `CONDUIT_OPENROUTER_URL`, `CONDUIT_OPENROUTER_APP_NAME`, `CONDUIT_OPENROUTER_APP_URL`

## Development

```bash
pip install -e .[dev]
pytest
```

