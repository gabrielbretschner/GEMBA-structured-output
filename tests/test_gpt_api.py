"""Tests for gemba.gpt_api error handling."""

from unittest.mock import MagicMock, patch

import pytest

# We need to set env vars before importing GptApi since it checks them in __init__
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gemba.gpt_api import GptApi


@pytest.fixture
def gpt_api():
    """Create a GptApi instance with a mocked OpenAI client (is_openai=True)."""
    with patch("openai.OpenAI"):
        return GptApi()


@pytest.fixture
def ollama_api():
    """Create a GptApi instance configured for a non-OpenAI endpoint."""
    with patch("openai.OpenAI"):
        return GptApi(base_url="http://localhost:11434")


@pytest.fixture
def azure_api():
    """Create a GptApi instance configured for Azure OpenAI (is_azure=True)."""
    env = {
        "OPENAI_AZURE_ENDPOINT": "https://example.openai.azure.com/",
        "OPENAI_AZURE_KEY": "azure-key",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("openai.AzureOpenAI"):
            return GptApi()


class TestRequestApiContentFilter:
    """Tests for content_filter and 4xx error handling in request_api."""

    def test_content_filter_returns_empty(self, gpt_api):
        """BadRequestError with code='content_filter' should return []."""
        from openai import BadRequestError

        err = BadRequestError(
            message="content filter triggered",
            response=MagicMock(status_code=400),
            body={"code": "content_filter"},
        )
        err.code = "content_filter"
        gpt_api.call_api = MagicMock(side_effect=err)
        assert gpt_api.request_api("prompt", "gpt-4") == []

    def test_bad_request_without_content_filter_raises(self, gpt_api):
        """BadRequestError without content_filter code should raise."""
        from openai import BadRequestError

        err = BadRequestError(
            message="invalid request",
            response=MagicMock(status_code=400),
            body={"code": "invalid_request"},
        )
        err.code = "invalid_request"
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(BadRequestError):
            gpt_api.request_api("prompt", "gpt-4")

    def test_not_found_error_raises(self, gpt_api):
        """NotFoundError should raise."""
        from openai import NotFoundError

        err = NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404),
            body={},
        )
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(NotFoundError):
            gpt_api.request_api("prompt", "gpt-4")

    def test_permission_denied_raises(self, gpt_api):
        """PermissionDeniedError should raise."""
        from openai import PermissionDeniedError

        err = PermissionDeniedError(
            message="permission denied",
            response=MagicMock(status_code=403),
            body={},
        )
        gpt_api.call_api = MagicMock(side_effect=err)
        with pytest.raises(PermissionDeniedError):
            gpt_api.request_api("prompt", "gpt-4")


class TestCallApiParameters:
    """Tests for parameter construction in call_api."""

    def test_max_completion_tokens_for_new_models(self, gpt_api):
        """Newer models should use max_completion_tokens instead of max_tokens."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=500)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs

    def test_max_tokens_for_old_models(self, gpt_api):
        """Older models should use max_tokens."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4", temperature=0, max_tokens=500)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs

    def test_response_format_passed_through(self, gpt_api):
        """response_format should be included in API parameters when set."""
        gpt_api.client = MagicMock()
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=None, response_format=rf)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == rf

    def test_response_format_omitted_when_none(self, gpt_api):
        """response_format should not be in parameters when None."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=None)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs


class TestClientInitialization:
    """Tests for GptApi client initialization paths."""

    def test_openai_sets_is_openai_true(self, gpt_api):
        """OpenAI API path should set is_openai=True."""
        assert gpt_api.is_openai is True

    def test_base_url_sets_is_openai_false(self, ollama_api):
        """Custom base_url should set is_openai=False."""
        assert ollama_api.is_openai is False

    def test_base_url_appends_v1(self):
        """base_url should get /v1 appended."""
        with patch("openai.OpenAI") as mock_openai:
            GptApi(base_url="http://localhost:11434")
            mock_openai.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="none"
            )

    def test_base_url_strips_trailing_slash(self):
        """Trailing slash on base_url should be stripped before appending /v1."""
        with patch("openai.OpenAI") as mock_openai:
            GptApi(base_url="http://localhost:11434/")
            mock_openai.assert_called_once_with(
                base_url="http://localhost:11434/v1", api_key="none"
            )

    def test_ollama_host_env_var(self):
        """OLLAMA_HOST env var should configure the client."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://myhost:11434"}, clear=False):
            # Remove OPENAI keys so OLLAMA_HOST takes priority
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.pop("OPENAI_AZURE_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("openai.OpenAI") as mock_openai:
                    api = GptApi()
                    mock_openai.assert_called_once_with(
                        base_url="http://myhost:11434/v1", api_key="ollama"
                    )
                    assert api.is_openai is False

    def test_missing_all_env_vars_raises(self):
        """Should raise when no API credentials are configured."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception, match="Set OPENAI_API_KEY"):
                GptApi()


class TestNonOpenAIParameters:
    """Tests for parameter gating on non-OpenAI providers."""

    def test_non_openai_skips_openai_specific_params(self, ollama_api):
        """Non-OpenAI providers should not send n, frequency_penalty, presence_penalty."""
        ollama_api.client = MagicMock()
        ollama_api.call_api("test prompt", "llama3.2", temperature=0, max_tokens=500)
        call_kwargs = ollama_api.client.chat.completions.create.call_args[1]
        assert "n" not in call_kwargs
        assert "frequency_penalty" not in call_kwargs
        assert "presence_penalty" not in call_kwargs

    def test_non_openai_skips_response_format(self, ollama_api):
        """Non-OpenAI providers should not send response_format."""
        ollama_api.client = MagicMock()
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        ollama_api.call_api("test prompt", "llama3.2", temperature=0, max_tokens=None, response_format=rf)
        call_kwargs = ollama_api.client.chat.completions.create.call_args[1]
        assert "response_format" not in call_kwargs

    def test_non_openai_always_uses_max_tokens(self, ollama_api):
        """Non-OpenAI providers should always use max_tokens, never max_completion_tokens."""
        ollama_api.client = MagicMock()
        ollama_api.call_api("test prompt", "gpt-4o", temperature=0, max_tokens=500)
        call_kwargs = ollama_api.client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs

    def test_openai_includes_openai_specific_params(self, gpt_api):
        """OpenAI provider should include n, frequency_penalty, presence_penalty."""
        gpt_api.client = MagicMock()
        gpt_api.call_api("test prompt", "gpt-4", temperature=0, max_tokens=None)
        call_kwargs = gpt_api.client.chat.completions.create.call_args[1]
        assert call_kwargs["n"] == 1
        assert call_kwargs["frequency_penalty"] == 0
        assert call_kwargs["presence_penalty"] == 0


class TestAzureParameters:
    """Azure OpenAI should be treated as OpenAI-family for request parameters."""

    def test_azure_sets_is_azure_true(self, azure_api):
        """The Azure endpoint path should flag is_azure and clear is_openai."""
        assert azure_api.is_azure is True
        assert azure_api.is_openai is False
        assert azure_api.supports_openai_params is True

    def test_azure_passes_response_format(self, azure_api):
        """Azure should forward response_format (structured output), unlike custom endpoints."""
        azure_api.client = MagicMock()
        rf = {"type": "json_schema", "json_schema": {"name": "test"}}
        azure_api.call_api("test prompt", "gpt-4.1", temperature=0, max_tokens=None, response_format=rf)
        call_kwargs = azure_api.client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == rf

    def test_azure_includes_openai_specific_params(self, azure_api):
        """Azure should include n, frequency_penalty, presence_penalty."""
        azure_api.client = MagicMock()
        azure_api.call_api("test prompt", "gpt-4.1", temperature=0, max_tokens=None)
        call_kwargs = azure_api.client.chat.completions.create.call_args[1]
        assert call_kwargs["n"] == 1
        assert call_kwargs["frequency_penalty"] == 0
        assert call_kwargs["presence_penalty"] == 0

    def test_azure_uses_max_completion_tokens_for_new_models(self, azure_api):
        """Azure gpt-4.1 should use max_completion_tokens like the OpenAI path."""
        azure_api.client = MagicMock()
        azure_api.call_api("test prompt", "gpt-4.1", temperature=0, max_tokens=500)
        call_kwargs = azure_api.client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs


class TestTruncationHandling:
    """Tests for the finish_reason != 'stop' (truncation) retry and cap."""

    def _make_response(self, content, finish_reason):
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish_reason
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_truncated_without_budget_returns_empty(self, gpt_api):
        """A truncated response with no token budget returns [] (unscored)."""
        gpt_api.call_api = MagicMock(return_value=self._make_response("partial", "length"))
        assert gpt_api.request_api("prompt", "model", max_tokens=None) == []

    def test_truncated_grows_budget_then_succeeds(self, gpt_api):
        """A truncated response should retry with a larger budget and succeed."""
        responses = [
            self._make_response("partial", "length"),
            self._make_response("42", "stop"),
        ]
        gpt_api.call_api = MagicMock(side_effect=responses)
        answers = gpt_api.request_api("prompt", "model", max_tokens=500)
        assert answers[0]["answer"] == "42"
        # Second call must request more tokens than the first.
        first_max = gpt_api.call_api.call_args_list[0][0][3]
        second_max = gpt_api.call_api.call_args_list[1][0][3]
        assert second_max > first_max

    def test_persistent_truncation_hits_cap_and_returns_empty(self, gpt_api):
        """If the response never finishes, growth stops at the cap and returns []."""
        from gemba.gpt_api import MAX_OUTPUT_TOKENS

        gpt_api.call_api = MagicMock(return_value=self._make_response("partial", "length"))
        assert gpt_api.request_api("prompt", "model", max_tokens=MAX_OUTPUT_TOKENS) == []


class TestThinkBlockStripping:
    """Tests for stripping <think> blocks from reasoning model responses."""

    def _make_response(self, content, finish_reason="stop"):
        """Create a mock API response with the given content."""
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish_reason
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_strips_think_block(self, gpt_api):
        """Should strip <think>...</think> and return only the answer."""
        gpt_api.call_api = MagicMock(
            return_value=self._make_response("<think>Let me reason about this...</think>\n42")
        )
        answers = gpt_api.request_api("prompt", "model")
        assert answers[0]["answer"] == "42"

    def test_strips_multiline_think_block(self, gpt_api):
        """Should strip multi-line think blocks."""
        gpt_api.call_api = MagicMock(
            return_value=self._make_response(
                "<think>\nStep 1: analyze\nStep 2: evaluate\n</think>\n{\"score\": 75}"
            )
        )
        answers = gpt_api.request_api("prompt", "model")
        assert answers[0]["answer"] == '{"score": 75}'

    def test_no_think_block_unchanged(self, gpt_api):
        """Responses without think blocks should be unchanged."""
        gpt_api.call_api = MagicMock(
            return_value=self._make_response("42")
        )
        answers = gpt_api.request_api("prompt", "model")
        assert answers[0]["answer"] == "42"

    def test_strips_multiple_think_blocks(self, gpt_api):
        """Should strip all think blocks if multiple are present."""
        gpt_api.call_api = MagicMock(
            return_value=self._make_response(
                "<think>first</think>hello <think>second</think>world"
            )
        )
        answers = gpt_api.request_api("prompt", "model")
        assert answers[0]["answer"] == "hello world"
