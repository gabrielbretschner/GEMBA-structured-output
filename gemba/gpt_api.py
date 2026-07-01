import logging
import os
import re
import sys
import time

import openai
import tqdm
from openai import BadRequestError, NotFoundError, PermissionDeniedError

logger = logging.getLogger(__name__)

# Upper bound on how far the token budget is grown when a response is truncated
# (finish_reason != "stop"). Without a cap, a persistently-unfinished response
# (e.g. a content filter surfaced as a finish_reason rather than an exception)
# would recurse forever. When the cap is hit we give up and log it.
MAX_OUTPUT_TOKENS = 4000


def _prompt_preview(prompt, limit=160):
    """Return a short single-line preview of a prompt for log messages.

    Args:
        prompt: Either a raw string prompt or a list of chat-message dicts.
        limit: Maximum number of characters before the preview is truncated.

    Returns:
        The last user turn (for chat prompts) or the prompt itself, collapsed to
        a single line and truncated to ``limit`` characters.
    """
    text = None
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                text = str(message.get("content", ""))
                break
    if text is None:
        text = str(prompt)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False, api_version=None, base_url=None):
        self.verbose = verbose
        # is_openai: the public OpenAI API. is_azure: Azure OpenAI. Both accept
        # OpenAI-native request params (n, penalties, response_format,
        # max_completion_tokens); custom/Ollama endpoints may not.
        self.is_openai = False
        self.is_azure = False

        if base_url is not None:
            # Custom endpoint (e.g. Ollama, vLLM, etc.)
            self.client = openai.OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="none")
            logger.info("GptApi backend: custom OpenAI-compatible endpoint at %s/v1", base_url.rstrip("/"))
        elif "OLLAMA_HOST" in os.environ:
            # Ollama API access
            ollama_host = os.environ["OLLAMA_HOST"].rstrip("/")
            self.client = openai.OpenAI(base_url=ollama_host + "/v1", api_key="ollama")
            logger.info("GptApi backend: Ollama at %s/v1", ollama_host)
        elif "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert "OPENAI_AZURE_KEY" in os.environ, "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            resolved_api_version = api_version or "2023-07-01-preview"
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version=resolved_api_version,
            )
            self.is_azure = True
            logger.info(
                "GptApi backend: Azure OpenAI at %s (api_version=%s)",
                os.environ["OPENAI_AZURE_ENDPOINT"],
                resolved_api_version,
            )
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
            self.is_openai = True
            logger.info("GptApi backend: OpenAI API")
        else:
            raise Exception("Set OPENAI_API_KEY, OPENAI_AZURE_KEY, or OLLAMA_HOST")

        # Suppress noisy HTTP loggers (don't touch the root logger)
        for _name in ("httpx", "openai", "urllib3"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    @property
    def supports_openai_params(self):
        """Whether the backend accepts OpenAI-native request parameters.

        True for the OpenAI API and Azure OpenAI (both honor ``n``, the
        frequency/presence penalties, ``response_format`` JSON schemas, and
        ``max_completion_tokens``). False for Ollama and other custom
        OpenAI-compatible endpoints, which may reject these parameters.
        """
        return self.is_openai or self.is_azure

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None, response_format=None):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens, response_format=response_format)
            cache[request] = answers

        # there is no valid answer
        if len(answers) == 0:
            logger.warning(
                "No answer from the model; scoring this segment as None. Prompt: %s",
                _prompt_preview(prompt),
            )
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose:
                logger.debug("Answer (t=%d): %s (%s)", temperature, answer, full_answer)
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            logger.info(
                "Response could not be parsed at temperature=%.1f; retrying at %.1f. Prompt: %s",
                temperature / 10,
                (temperature + 1) / 10,
                _prompt_preview(prompt),
            )
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache, response_format=response_format)

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=None, response_format=None):
        if temperature > 10:
            logger.warning(
                "Exhausted temperature escalation without a parseable answer; returning empty. Prompt: %s",
                _prompt_preview(prompt),
            )
            return []

        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens, response_format=response_format)
                break
            except (BadRequestError, NotFoundError, PermissionDeniedError) as e:
                if getattr(e, "code", None) == "content_filter":
                    logger.warning("Content filter blocked the request; returning empty. Prompt: %s", _prompt_preview(prompt))
                    return []
                raise
            except Exception as e:
                error_body = getattr(e, "error", None)
                if isinstance(error_body, dict) and error_body.get("code") == "invalid_model_output":
                    logger.warning("API reported invalid_model_output; returning empty. Prompt: %s", _prompt_preview(prompt))
                    return []
                logger.warning("API error, retrying: %s", e)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if choice.message.content is None:
                logger.warning(
                    "API returned empty content (finish_reason=%s); returning empty. Prompt: %s",
                    getattr(choice, "finish_reason", None),
                    _prompt_preview(prompt),
                )
                return []
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()

            # Strip <think>...</think> blocks from reasoning models
            answer = re.sub(r"<think>[\s\S]*?</think>\s*", "", answer).strip()

            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                if max_tokens is None:
                    logger.warning(
                        "Response truncated (finish_reason=%s) and no token budget to grow; returning empty. Prompt: %s",
                        choice.finish_reason,
                        _prompt_preview(prompt),
                    )
                    return []
                if max_tokens >= MAX_OUTPUT_TOKENS:
                    logger.warning(
                        "Response still truncated (finish_reason=%s) at the token cap (%d); giving up and returning empty. Prompt: %s",
                        choice.finish_reason,
                        MAX_OUTPUT_TOKENS,
                        _prompt_preview(prompt),
                    )
                    return []
                new_max_tokens = min(max_tokens + 200, MAX_OUTPUT_TOKENS)
                logger.info(
                    "Response truncated (finish_reason=%s); growing token budget %d -> %d and retrying.",
                    choice.finish_reason,
                    max_tokens,
                    new_max_tokens,
                )
                return self.request_api(prompt, model, temperature=temperature, max_tokens=new_max_tokens, response_format=response_format)

            answers.append({
                "answer": answer,
                "finish_reason": choice.finish_reason,
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens, response_format=None):
        parameters = {
            "temperature": temperature/10,
            "top_p": 1,
            "model": model
        }

        if self.supports_openai_params:
            parameters["n"] = 1
            parameters["frequency_penalty"] = 0
            parameters["presence_penalty"] = 0

        if response_format is not None and self.supports_openai_params:
            parameters["response_format"] = response_format

        if max_tokens is not None:
            if self.supports_openai_params and any(model.startswith(p) for p in ("gpt-4.1", "gpt-4o", "gpt-5")):
                parameters["max_completion_tokens"] = max_tokens
            else:
                parameters["max_tokens"] = max_tokens

        if isinstance(prompt, list):
            # check that prompt contain list of dictionaries with role and content
            assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
            assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

            parameters["messages"] = prompt
        else:
            parameters["messages"] = [{
                "role": "user",
                "content": prompt,
            }]

        return self.client.chat.completions.create(**parameters)

    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None, response_format=None):
        answers = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
            prompt = row["prompt"]
            parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens, response_format=response_format)
            answers += parsed_answers
        return answers
