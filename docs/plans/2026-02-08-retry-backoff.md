# Retry with Exponential Backoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add tenacity-based exponential backoff retries for all LLM and embedding API calls to handle rate limits (429) and transient server errors (5xx).

**Architecture:** Wrap `agent.run()` calls with tenacity retry decorator that catches `ModelHTTPError` (429/5xx). For direct `litellm.embedding()` calls, catch `litellm.RateLimitError` and `litellm.InternalServerError`. A single utility module provides the retry logic; each agent call site gets a one-line change.

**Tech Stack:** tenacity (retry library), pydantic-ai `ModelHTTPError`, litellm exceptions

**Key findings from research:**
- litellm's direct `acompletion()`/`embedding()` do NOT auto-retry on 429 - only the Router class does
- `pydantic-ai-litellm` wraps litellm errors into `ModelHTTPError(status_code=429)` via `_completion_create` (line 228-233)
- `litellm.RateLimitError` has `.status_code = 429` and preserves response headers
- VectorSimilarityMatcher calls `litellm.embedding()` directly (not via pydantic-ai agents)

---

### Task 1: Add tenacity dependency

**Files:**
- Modify: `pyproject.toml:17-33`

**Step 1: Add tenacity to dependencies**

Add `"tenacity>=8.0"` after `"pydantic-ai-litellm"` in the dependencies list:

```toml
dependencies = [
    "streamlit>=1.30",
    "pydantic>=2.0",
    "pydantic-ai",
    "pydantic-ai-litellm",
    "tenacity>=8.0",
    "httpx",
    ...
]
```

**Step 2: Install**

Run: `uv sync`
Expected: tenacity installs successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add tenacity dependency for retry/backoff"
```

---

### Task 2: Add retry settings to config

**Files:**
- Modify: `src/hr_breaker/config.py:36-78` (Settings class)
- Modify: `src/hr_breaker/config.py:82-127` (get_settings)
- Modify: `.env.example`

**Step 1: Write failing test**

Create `tests/test_retry.py`:

```python
from hr_breaker.config import Settings


def test_settings_has_retry_fields():
    s = Settings()
    assert s.retry_max_attempts == 5
    assert s.retry_max_wait == 60.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retry.py::test_settings_has_retry_fields -v`
Expected: FAIL with `AttributeError` or validation error

**Step 3: Add retry settings to Settings class**

In `src/hr_breaker/config.py`, add to the `Settings` class after agent limits:

```python
    # Retry settings
    retry_max_attempts: int = 5
    retry_max_wait: float = 60.0
```

Add to `get_settings()` return:

```python
        # Retry settings
        retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "5")),
        retry_max_wait=float(os.getenv("RETRY_MAX_WAIT", "60")),
```

Add to `.env.example` after agent limits section:

```
# Retry settings (for rate limits / transient errors)
# RETRY_MAX_ATTEMPTS=5
# RETRY_MAX_WAIT=60
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retry.py::test_settings_has_retry_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hr_breaker/config.py .env.example tests/test_retry.py
git commit -m "feat: add retry config settings"
```

---

### Task 3: Create retry utility module

**Files:**
- Create: `src/hr_breaker/utils/retry.py`
- Modify: `tests/test_retry.py`

**Step 1: Write failing tests**

Append to `tests/test_retry.py`:

```python
import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ModelHTTPError

from hr_breaker.utils.retry import is_retryable, run_with_retry


def test_is_retryable_429():
    exc = ModelHTTPError(status_code=429, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_500():
    exc = ModelHTTPError(status_code=500, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_502():
    exc = ModelHTTPError(status_code=502, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_400_not_retryable():
    exc = ModelHTTPError(status_code=400, model_name="test")
    assert is_retryable(exc) is False


def test_is_retryable_unrelated_exception():
    assert is_retryable(ValueError("nope")) is False


def test_is_retryable_litellm_rate_limit():
    from litellm.exceptions import RateLimitError

    exc = RateLimitError(
        message="rate limited",
        llm_provider="gemini",
        model="gemini/gemini-3-flash",
    )
    assert is_retryable(exc) is True


async def test_run_with_retry_succeeds_first_try():
    func = AsyncMock(return_value="ok")
    result = await run_with_retry(func, "arg1", key="val")
    assert result == "ok"
    func.assert_called_once_with("arg1", key="val")


async def test_run_with_retry_retries_on_429_then_succeeds():
    func = AsyncMock(
        side_effect=[
            ModelHTTPError(status_code=429, model_name="test"),
            "ok",
        ]
    )
    result = await run_with_retry(func, "arg1")
    assert result == "ok"
    assert func.call_count == 2


async def test_run_with_retry_exhausts_attempts():
    func = AsyncMock(
        side_effect=ModelHTTPError(status_code=429, model_name="test")
    )
    with pytest.raises(ModelHTTPError):
        await run_with_retry(func, "arg1", _max_attempts=2, _max_wait=0.01)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retry.py -v -k "not test_settings"`
Expected: FAIL with `ImportError: cannot import name 'is_retryable'`

**Step 3: Implement retry module**

Create `src/hr_breaker/utils/retry.py`:

```python
"""Retry utilities for LLM API calls with exponential backoff."""

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pydantic_ai.exceptions import ModelHTTPError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    """Check if exception is retryable (rate limit or transient server error)."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS_CODES
    return False


async def run_with_retry(
    func,
    *args,
    _max_attempts: int | None = None,
    _max_wait: float | None = None,
    **kwargs,
):
    """Run an async callable with retry on rate limits and transient errors.

    Args:
        func: Async callable to run.
        *args: Positional args passed to func.
        _max_attempts: Override max retry attempts (default: from settings).
        _max_wait: Override max wait seconds (default: from settings).
        **kwargs: Keyword args passed to func.
    """
    from hr_breaker.config import get_settings

    settings = get_settings()
    max_attempts = _max_attempts or settings.retry_max_attempts
    max_wait = _max_wait or settings.retry_max_wait

    @retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _inner():
        return await func(*args, **kwargs)

    return await _inner()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retry.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/hr_breaker/utils/retry.py tests/test_retry.py
git commit -m "feat: add retry utility with exponential backoff"
```

---

### Task 4: Wire retry into all agent call sites

**Files:**
- Modify: `src/hr_breaker/agents/job_parser.py:35`
- Modify: `src/hr_breaker/agents/optimizer.py:281`
- Modify: `src/hr_breaker/agents/combined_reviewer.py:267`
- Modify: `src/hr_breaker/agents/hallucination_detector.py:134`
- Modify: `src/hr_breaker/agents/ai_generated_detector.py:108`
- Modify: `src/hr_breaker/agents/name_extractor.py:35`

**Step 1: Apply changes to all 6 agents**

Each file gets the same pattern. Add import at top:

```python
from hr_breaker.utils.retry import run_with_retry
```

Change `agent.run()` → `run_with_retry(agent.run, ...)`:

**job_parser.py** line 35:
```python
# Before:
result = await agent.run(f"Parse this job posting:\n\n{text}")
# After:
result = await run_with_retry(agent.run, f"Parse this job posting:\n\n{text}")
```

**optimizer.py** line 281:
```python
# Before:
result = await agent.run(prompt)
# After:
result = await run_with_retry(agent.run, prompt)
```

**combined_reviewer.py** line ~267 (the agent.run call with BinaryContent):
```python
# Before:
result = await agent.run([prompt, BinaryContent(...)])
# After:
result = await run_with_retry(agent.run, [prompt, BinaryContent(...)])
```

**hallucination_detector.py** line 134:
```python
# Before:
result = await agent.run(prompt)
# After:
result = await run_with_retry(agent.run, prompt)
```

**ai_generated_detector.py** line 108:
```python
# Before:
result = await agent.run(prompt)
# After:
result = await run_with_retry(agent.run, prompt)
```

**name_extractor.py** line 35:
```python
# Before:
result = await agent.run(f"Extract the name from this resume:\n\n{snippet}")
# After:
result = await run_with_retry(agent.run, f"Extract the name from this resume:\n\n{snippet}")
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: All existing tests still pass (agent.run is mocked in tests)

**Step 3: Commit**

```bash
git add src/hr_breaker/agents/
git commit -m "feat: add retry with backoff to all agent calls"
```

---

### Task 5: Wire retry into VectorSimilarityMatcher

**Files:**
- Modify: `src/hr_breaker/filters/vector_similarity_matcher.py:42-47`

**Step 1: Write failing test**

Append to `tests/test_retry.py`:

```python
async def test_run_with_retry_retries_litellm_rate_limit():
    from litellm.exceptions import RateLimitError

    exc = RateLimitError(
        message="rate limited",
        llm_provider="gemini",
        model="gemini/gemini-3-flash",
    )
    func = AsyncMock(side_effect=[exc, "ok"])
    result = await run_with_retry(func, "arg1")
    assert result == "ok"
    assert func.call_count == 2
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_retry.py::test_run_with_retry_retries_litellm_rate_limit -v`
Expected: PASS (is_retryable already handles status_code=429 via getattr)

**Step 3: Wrap embedding call with retry**

In `src/hr_breaker/filters/vector_similarity_matcher.py`, add import:

```python
from hr_breaker.utils.retry import run_with_retry
```

Replace the `litellm.embedding(...)` call (lines 42-47). Since `litellm.embedding` is sync, wrap it in an async lambda or use `asyncio.to_thread`. Actually `litellm.aembedding` exists - use that:

```python
# Before:
from litellm import embedding as litellm_embedding
# ...
result = litellm.embedding(
    model=settings.embedding_model,
    input=[resume_text, job_text],
    dimensions=settings.embedding_output_dimensionality,
)

# After:
from litellm import aembedding as litellm_aembedding
# ...
result = await run_with_retry(
    litellm_aembedding,
    model=settings.embedding_model,
    input=[resume_text, job_text],
    dimensions=settings.embedding_output_dimensionality,
)
```

The existing `except Exception` block (lines 48-56) still catches errors after retries are exhausted.

**Step 4: Run existing filter tests**

Run: `uv run pytest tests/test_filters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hr_breaker/filters/vector_similarity_matcher.py tests/test_retry.py
git commit -m "feat: add retry to embedding calls in VectorSimilarityMatcher"
```

---

### Task 6: Update docs

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add retry section to CLAUDE.md**

Add to the Environment Variables section:

```markdown
- `RETRY_MAX_ATTEMPTS` - Max retry attempts for rate limits (default: `5`)
- `RETRY_MAX_WAIT` - Max backoff wait in seconds (default: `60`)
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add retry config to CLAUDE.md"
```

---

### Task 7: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 2: Verify retry logging works**

Run: `LOG_LEVEL=DEBUG uv run python -c "
import asyncio
from pydantic_ai import ModelHTTPError
from hr_breaker.utils.retry import run_with_retry
from unittest.mock import AsyncMock

async def main():
    func = AsyncMock(side_effect=[
        ModelHTTPError(status_code=429, model_name='test'),
        'ok',
    ])
    result = await run_with_retry(func, 'hello', _max_wait=0.1)
    print(f'Result: {result}')

asyncio.run(main())
"`
Expected: WARNING log line showing retry, then "Result: ok"
