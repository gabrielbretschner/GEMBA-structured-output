# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GEMBA is a GPT-based machine-translation quality metric (GEMBA-MQM, GEMBA-DA, and variants). It prompts an LLM to score translations, parses the response into a numeric score, and (optionally) evaluates those scores against WMT gold data via `mt-metrics-eval`. This fork ("GEMBA-structured-output") adds OpenAI structured-output (JSON schema) support, Azure/Ollama/custom endpoints, and a test suite on top of the upstream Microsoft GEMBA.

## Commands

```bash
pip install -e .                 # editable install (required before running tests — package must be importable)
pip install -e '.[eval]'         # also install mt-metrics-eval (Linux only; not on PyPI) for evaluate.py

python -m pytest                 # run all tests
python -m pytest tests/test_gpt_api.py::TestRequestApiContentFilter::test_content_filter_returns_empty   # single test

# Score two parallel files (same line count). Prints one score per line.
gemba --source=src.txt --hypothesis=hyp.txt --source_lang=English --target_lang=Czech --method=GEMBA-MQM --model=gpt-4
python -m gemba ...              # equivalent; main.py is a compat shim for `python main.py`
```

Credentials are read from the environment at `GptApi.__init__` time, in priority order: `--base_url` flag → `OLLAMA_HOST` → `OPENAI_AZURE_ENDPOINT` (+`OPENAI_AZURE_KEY`) → `OPENAI_API_KEY`. Only the OpenAI path sets `is_openai=True`, which gates `n`/penalty params and `response_format`.

Relevant CLI flags beyond the README: `--no_structured_output` (disable JSON-schema response_format), `--base_url` (custom OpenAI-compatible endpoint), `--api_version` (Azure), `--list_mqm_errors`.

## Architecture

The scoring pipeline lives in `gemba/` and flows: **method → prompt template → LLM request → response parser → score**.

- **`utils.py`** — `get_gemba_scores()` is the single public entry point (re-exported from `gemba/__init__.py`). It builds a pandas DataFrame of segments, picks the prompt template and parser by method name, and calls `GptApi.bulk_request`. It also owns `RESPONSE_FORMATS` and `_get_response_format()`, which map a method to a JSON-schema `response_format` (`score` schema for DA/SQM, `mqm` schema for MQM; `None` otherwise).
- **`gpt_api.py`** — `GptApi` wraps the `openai` client. `request()` handles per-prompt caching, retry-by-raising-temperature when parsing fails (temperature is an int 0–10, divided by 10 before the API call; >10 gives up), and re-requesting with more tokens when `finish_reason != "stop"`. `request_api` swallows `content_filter` / `invalid_model_output` as empty results and strips `<think>...</think>` blocks from reasoning-model output.
- **`prompt.py`** — the `prompts` dict defines every non-MQM method (DA, SQM, stars, classes, and `_ref` reference-based variants) as `{prompt, validate_answer, use_ref}`. Also holds the numeric/stars/class parsers. `_ref` methods require a `reference_seg` column.
- **`gemba_mqm_utils.py`** — MQM few-shot prompt construction (`TEMPLATE_GEMBA_MQM`) and `parse_mqm_answer`, which converts errors into a negative penalty score (critical=25, major=5, minor=1, capped, first 5 errors only). Parsers accept **both** structured JSON and the legacy free-text format, so changes must keep both paths working.
- **`gemba_esa.py`** — GEMBA-ESA, a two-stage method: first request error spans, then a ranking request that consumes them.

### Re-annotation protocol
`get_gemba_scores(..., reannotation_rounds=N, details=...)` layers variable-round re-annotation on **GEMBA-MQM and GEMBA-ESA only** (other methods raise if `rounds>0`). `_run_annotation_rounds` in `utils.py` runs the initial round plus N rounds that continue the *same chat conversation* (`append_reannotation_turn` appends the prior assistant answer + a `REANNOTATION_INSTRUCTION_*` user turn). Semantics are **replace** (the model returns the complete revised set each round; the final round is scored) and **fixed N** (no early stop). Intermediate rounds parse with identity to feed raw text forward; MQM scores the final text via `parse_mqm_answer`, ESA runs its ranking stage once on the final spans. When `reannotation_rounds>0` or `details=True`, `get_gemba_scores` returns per-segment dicts `{score, annotation, trajectory[]}` (ESA adds `error_spans`) instead of the flat score list; `rounds=0 & details=False` is byte-identical to the legacy path. The CLI dumps these as JSONL via `--annotations_out`.

### Parsing is intentionally defensive
The response parsers (`parse_numerical_answer`, `validate_stars`, `parse_mqm_answer`) try structured JSON first, then fall back through many heuristics for verbose/markdown/fraction-formatted model output. When editing, preserve the structured-first-then-fallback ordering and the existing fallbacks — they exist to handle real failure modes from different models. Returning `None` from a parser is meaningful: it triggers the temperature-bump retry in `GptApi.request`.

### Caching
Every run opens a `diskcache` at `cache/{model}_{method}` keyed on `{model, temperature, prompt}`. The cache never expires/evicts. Delete `cache/` (gitignored) to force fresh API calls.

## Evaluation harness (separate from scoring)

`gemba_da.py`, `evaluate.py`, and `gemba/{testset,scores,mtme_tools}.py` form the experiment/benchmark path against WMT data. They depend on `mt-metrics-eval` and a downloaded `mt-metrics-eval-v2/` resource tree (see README for the `mtme --download` setup). This path is gated to Linux in `pyproject.toml` (`[tool.uv] environments`) because the dependency isn't on PyPI. `evaluate.py` needs `PYTHONPATH=mt-metrics-eval`.

## Tests

`tests/` uses pytest with `unittest.mock` to patch `openai.OpenAI` — no network or real keys needed (env keys are set to dummies at import). Coverage focuses on `gpt_api` error handling (content filter, retries, endpoint selection) and the `prompt`/`mqm` parsers, including structured-output and `<think>`-stripping behavior.
