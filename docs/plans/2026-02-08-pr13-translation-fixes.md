# PR #13 Translation Feature Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix bugs and quality issues identified in code review of PR #13 (resume language switcher).

**Architecture:** All changes are surgical fixes to the PR's existing code. No new modules. Fix `list_all()` filename parsing for lang suffix, gate translation on `validation.passed`, fix f-string logging, and remove committed `PULL_REQUEST.md`.

**Tech Stack:** Python 3.10+, Pydantic, pytest, Streamlit, Click

---

### Task 1: Fix `list_all()` filename parsing for language suffix

The PR adds `_en`/`_ru` to every filename but `list_all()` still parses the old format. The last `_` segment is now a lang code, not part of the role. This breaks the history sidebar and `hr-breaker list`.

**Files:**
- Modify: `src/hr_breaker/services/pdf_storage.py:50-80`
- Test: `tests/test_pdf_storage.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_pdf_storage.py
"""Tests for PDF storage filename parsing."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from hr_breaker.services.pdf_storage import PDFStorage, sanitize_filename


class TestListAllWithLanguageSuffix:
    def _make_pdf(self, tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"%PDF-fake")
        return p

    def test_parse_filename_with_lang_suffix(self, tmp_path):
        """john_doe_acme_engineer_en.pdf should parse correctly."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer_en.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
        assert "Acme" in r.company
        assert "Engineer" in r.job_title
        # Lang code should NOT appear in job_title or company
        assert "En" not in r.job_title
        assert "En" not in r.company

    def test_parse_filename_with_ru_suffix(self, tmp_path):
        """john_doe_acme_engineer_ru.pdf should parse the same metadata."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer_ru.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
        assert "Engineer" in r.job_title

    def test_parse_old_filename_without_lang_suffix(self, tmp_path):
        """Backward compat: john_doe_acme_engineer.pdf (no lang) still works."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pdf_storage.py -v`
Expected: FAIL - `En` appears in job_title or company since lang code is treated as a filename part.

**Step 3: Implement the fix**

In `pdf_storage.py`, update `list_all()` to strip a trailing 2-letter lang code before parsing:

```python
# Inside list_all(), after splitting parts:
# Strip known language suffix (2-letter code at end)
lang_code = None
if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
    lang_code = parts[-1]
    parts = parts[:-1]
```

Insert this right after `parts = pdf_path.stem.split("_")` (line 55), before the heuristic parsing block.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pdf_storage.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_pdf_storage.py src/hr_breaker/services/pdf_storage.py
git commit -m "fix: list_all() parsing for language suffix in filenames"
```

---

### Task 2: Skip translation when optimization fails

Currently `orchestration.py` runs translation even if `validation.passed` is False. This wastes LLM API calls.

**Files:**
- Modify: `src/hr_breaker/orchestration.py:186-191` (the PR's version)
- Test: `tests/test_orchestration.py`

**Step 1: Write the failing test**

Add to `tests/test_orchestration.py`:

```python
class TestOptimizeForJobTranslationGating:
    @pytest.mark.asyncio
    async def test_skip_translation_when_validation_fails(self):
        """Should NOT translate if optimization didn't pass filters."""
        from hr_breaker.models.language import get_language
        russian = get_language("ru")
        source = ResumeSource(content="Test content")
        mock_optimized = OptimizedResume(
            html="<div>Test</div>",
            source_checksum=source.checksum,
            pdf_text="Test",
            pdf_bytes=b"pdf",
        )
        job = JobPosting(
            title="Dev", company="Co",
            requirements=["Python"], keywords=["python"],
        )

        with patch("hr_breaker.orchestration.optimize_resume", new_callable=AsyncMock) as mock_opt, \
             patch("hr_breaker.orchestration._render_and_extract") as mock_render, \
             patch("hr_breaker.orchestration.run_filters", new_callable=AsyncMock) as mock_filters, \
             patch("hr_breaker.orchestration.translate_and_rerender", new_callable=AsyncMock) as mock_translate:

            mock_opt.return_value = mock_optimized
            mock_render.return_value = mock_optimized
            mock_filters.return_value = ValidationResult(results=[
                FilterResult(filter_name="test", passed=False, score=0.3, threshold=0.7),
            ])

            from hr_breaker.orchestration import optimize_for_job
            await optimize_for_job(source, job=job, language=russian, max_iterations=1)

            mock_translate.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::TestOptimizeForJobTranslationGating -v`
Expected: FAIL - `translate_and_rerender` gets called even though validation failed.

**Step 3: Implement the fix**

In `orchestration.py`, change the translation gate from:

```python
if language is not None and language.code != "en" and optimized is not None and optimized.html:
```

to:

```python
if language is not None and language.code != "en" and validation.passed and optimized is not None and optimized.html:
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hr_breaker/orchestration.py tests/test_orchestration.py
git commit -m "fix: skip translation when optimization fails validation"
```

---

### Task 3: Fix f-string logging anti-pattern

f-strings in `logger.debug()` format even when debug is disabled. Replace with lazy `%s` style.

**Files:**
- Modify: `src/hr_breaker/orchestration.py` (the PR's `translate_and_rerender` function)

**Step 1: Replace all f-string logger calls in `translate_and_rerender`**

Replace these patterns:

```python
# Before
logger.debug(f"{iter_label}: translating to {language.english_name}")
logger.debug(f"{iter_label}: reviewing translation")
logger.debug(f"{iter_label}: review score={review.score:.2f}, passed={review.passed}")
logger.debug(f"Translation approved (score={review.score:.2f})")
logger.debug(f"Translation feedback: {feedback}")
logger.warning(f"Translation review did not pass after {max_translation_iterations} iterations (score={review.score:.2f}), using last translation")

# After
logger.debug("%s: translating to %s", iter_label, language.english_name)
logger.debug("%s: reviewing translation", iter_label)
logger.debug("%s: review score=%.2f, passed=%s", iter_label, review.score, review.passed)
logger.debug("Translation approved (score=%.2f)", review.score)
logger.debug("Translation feedback: %s", feedback)
logger.warning("Translation review did not pass after %d iterations (score=%.2f), using last translation", max_translation_iterations, review.score)
```

**Step 2: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: PASS (no behavior change)

**Step 3: Commit**

```bash
git add src/hr_breaker/orchestration.py
git commit -m "style: use lazy logging instead of f-strings in translation"
```

---

### Task 4: Remove `PULL_REQUEST.md` from the branch

This is PR metadata, not project documentation. Should live in the PR description.

**Step 1: Delete the file**

```bash
git rm PULL_REQUEST.md
```

**Step 2: Commit**

```bash
git commit -m "chore: remove PULL_REQUEST.md (belongs in PR description, not repo)"
```

---

### Task 5: Preserve English result when translating in Streamlit

Currently clicking "Translate" overwrites `st.session_state["last_result"]`. User loses the English version.

**Files:**
- Modify: `src/hr_breaker/main.py` (the PR's translate button section)

**Step 1: Store English HTML separately before overwriting**

In the translate button handler, before updating session state, save the English original:

```python
# Before the st.session_state["last_result"] update:
if "english_html" not in st.session_state["last_result"]:
    st.session_state["last_result"]["english_html"] = optimized.html
```

This way the English HTML is always preserved on first translation, and subsequent translations still use the original English as the source (not a re-translation of a translation).

**Step 2: Verify manually in Streamlit**

Run: `uv run streamlit run src/hr_breaker/main.py`
- Optimize a resume, verify result shown
- Click translate, verify translated result shown
- Check `st.session_state["last_result"]["english_html"]` still holds original

**Step 3: Commit**

```bash
git add src/hr_breaker/main.py
git commit -m "fix: preserve English HTML when translating in UI"
```

---

## Summary

- Task 1: Fix `list_all()` parsing (bug - history/list broken)
- Task 2: Gate translation on `validation.passed` (bug - wastes API calls)
- Task 3: Fix f-string logging (style)
- Task 4: Remove `PULL_REQUEST.md` (cleanup)
- Task 5: Preserve English result in UI (UX bug)

## Unresolved questions

- Should these fixes be pushed as review comments for the PR author to implement, or committed directly to the PR branch?
- Should we also add a programmatic HTML tag-count validation after translation (item 6 from review), or leave that for a follow-up?
