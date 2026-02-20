# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Conduit is a provider-aware Python LLM client library that provides a unified interface over three backends: **vLLM**, **Ollama**, and **OpenRouter**. Minimal dependencies: `httpx` and `pydantic>=2.0`. Python 3.11+.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all unit tests
pytest tests/unit

# Run a single test file
pytest tests/unit/client/test_retry.py

# Run a single test by name
pytest tests/unit/client/test_retry.py -k "test_retry_on_rate_limit"

# Run integration tests (requires .env with API keys at project root)
pytest -m integration -v

# Run all tests
pytest
```

Build backend is `hatchling`. Tests use `pytest` + `pytest-asyncio` with `asyncio_mode = "auto"` (async test functions are auto-detected). No linter or formatter is configured.

## Architecture

### Four-Layer Design

```
Client Layer          → client.py (Conduit, SyncConduit)
Provider Abstraction  → providers/base.py (BaseProvider[ConfigT] ABC)
Shared Serialization  → providers/openai_format.py, providers/streaming.py
Config & Models       → config/, models/messages.py, tools/schema.py
```

**Client layer** (`client.py`): `Conduit` (async) and `SyncConduit` (sync wrapper) are the public entry points. Three methods each: `chat()`, `chat_stream()`, `chat_events()`. Provider selection is driven by `ProviderSpec` registry (`_PROVIDER_SPECS` tuple). `from_env()` auto-detects provider from environment variables. `chat(stream=True)` aggregates streaming chunks via `StreamEventAccumulator`.

**Provider layer** (`providers/`): `BaseProvider[ConfigT]` is generic over its config type. Concrete providers implement `build_request_body()`, `chat()`, `chat_stream()`. Base provides HTTP helpers (`post_json`, `make_url`) and error mapping (`map_http_error` maps HTTP status codes to exception subtypes).

**Shared layer**: `openai_format.py` has shared OpenAI-format serialization used by both vLLM and OpenRouter. `streaming.py` has SSE (`iter_sse_data`) and NDJSON (`iter_ndjson`) parsers plus `ToolCallChunkAccumulator`.

**Config/Models layer**: All Pydantic v2 models with `extra="forbid"`. `BaseLLMConfig` holds common sampling params; provider configs extend it. `models/messages.py` defines the canonical message types, `ChatRequest`, `ChatResponse`, `StreamEvent` (discriminated union), etc.

### Provider-Specific Patterns

- **Ollama**: Dual endpoint dispatch (`/api/chat` vs `/api/generate`). Synthesizes tool call IDs (`ollama_call_{index}`). Sampling params go in `options` sub-object.
- **OpenRouter**: Supports `native_finish_reason` in stream chunks. Maps `RequestContext` fields into request metadata. May have an extra `"data"` wrapper key in SSE payloads.
- **vLLM**: OpenAI-compatible API with vLLM-specific extensions (structured outputs, beam search, chat templates).

### Key Patterns

- **Strict Pydantic validation**: `extra="forbid"` on all models catches typos/unknown fields at construction time. Config overrides are re-validated through Pydantic after `deep_merge()`.
- **Tool-call completion state machine**: In `utils/streaming.py`, `should_complete_tool_calls()` uses a four-tier state machine (tracked by `saw_tool_call_delta`, `emitted_completed_tool_calls`, `allow_terminal_tool_call_flush`).
- **Retry**: `RetryPolicy` with exponential backoff + jitter. Respects `Retry-After` header. Streaming retry is blocked once output has been yielded.
- **SyncConduit**: Bridges async to sync via a dedicated event loop in `_iter_async()`.

### Exception Hierarchy

`ConduitError` → `ConfigValidationError`, `ProviderError` (with subtypes: `AuthenticationError`, `RateLimitError`, `ModelNotFoundError`, `ContextLengthError`, `ProviderUnavailableError`, `ResponseParseError`), `ToolError` (with `ToolSchemaError`, `ToolCallParseError`), `StreamError`.

## Testing Patterns

- Unit tests mock HTTP via `httpx.MockTransport` injected into providers
- Provider tests call provider methods directly; client tests monkey-patch `client._provider.chat` / `client._provider.chat_stream`
- Shared fixtures in `tests/unit/conftest.py`: `sample_messages`, `sample_tools`

### Assertion Strictness

Unit tests use deterministic mock data, so assertions must be maximally precise to catch regressions:

- **Exact equality over inequalities**: use `len(x) == 1` not `len(x) >= 1`, `== 2` not `> 0`. Every field with a known expected value should be checked with `==`.
- **Check all fields, not just one**: when a mock produces `UsageStats(prompt_tokens=10, completion_tokens=4, total_tokens=14)`, assert all three fields — not just `total_tokens`. Use full model equality (e.g. `== UsageStats(...)`, `== ToolCall(...)`) where practical.
- **Exact counts on collections**: use `len(response.tool_calls) == 1` not just `is not None`. The mock data is deterministic so the count is always known.
- **Exact error messages**: use `pytest.raises(ExcType, match="full expected message")` instead of partial `"keyword" in str(exc)` checks.
- **No redundant guards**: don't write `assert x is not None` followed by `assert isinstance(x, Foo)` — the isinstance check alone suffices, or better yet assert the exact value.

Integration tests are an exception — LLM responses are non-deterministic, so structural checks (`is not None`, `len() > 0`, `in`) are appropriate there.

### Integration Tests

- Run with `pytest -m integration -v`
- Credentials come from `.env` at project root, loaded by `tests/integration/conftest.py` via `python-dotenv`
- Tests auto-skip when env vars are missing; if all 40 tests skip, the `.env` is not loading

## Commit Convention

Commit messages use the format `<prefix>: <msg>` — a single concise sentence, no body. Common prefixes: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Public API Surface

All public exports are defined in `src/conduit/__init__.py`. When adding new public types, update this file.
