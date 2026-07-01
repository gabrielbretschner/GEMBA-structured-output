"""Tests for the variable-round re-annotation protocol (GEMBA-MQM / GEMBA-ESA)."""

import os
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gemba import utils
from gemba.gemba_mqm_utils import REANNOTATION_INSTRUCTION_MQM, parse_mqm_answer
from gemba.gemba_esa import REANNOTATION_INSTRUCTION_ESA


class NoCache:
    """Cache stub that always misses (avoids touching the diskcache on disk)."""

    def __contains__(self, key):
        return False

    def __setitem__(self, key, value):
        pass


_INSTRUCTIONS = {REANNOTATION_INSTRUCTION_MQM, REANNOTATION_INSTRUCTION_ESA}


class FakeGptApi:
    """Stand-in for GptApi that scripts one raw answer per re-annotation round.

    The round is detected by counting re-annotation instruction turns already in the
    conversation, so it is decoupled from the few-shot template size. String prompts
    (the ESA ranking stage) get the fixed ``ranking_answer``.
    """

    def __init__(self, answers_by_round, ranking_answer="80", **kwargs):
        self.answers_by_round = answers_by_round
        self.ranking_answer = ranking_answer
        self.seen_prompts = []

    def bulk_request(self, df, model, parse, cache=None, max_tokens=None, response_format=None):
        out = []
        for _, row in df.iterrows():
            prompt = row["prompt"]
            self.seen_prompts.append(prompt)
            if isinstance(prompt, str):
                raw = self.ranking_answer
            else:
                rnd = sum(1 for m in prompt if m.get("content") in _INSTRUCTIONS)
                raw = self.answers_by_round[rnd]
            out.append({
                "answer": parse(raw),
                "temperature": 0,
                "answer_id": 0,
                "prompt": prompt,
                "finish_reason": "stop",
                "model": model,
            })
        return out


def _run(method, answers_by_round, rounds, details=False, ranking_answer="80"):
    fake = FakeGptApi(answers_by_round, ranking_answer=ranking_answer)
    with patch.object(utils, "GptApi", return_value=fake), \
         patch.object(utils.dc, "Cache", return_value=NoCache()):
        result = utils.get_gemba_scores(
            ["src1"], ["hyp1"], "English", "German", method, "gpt-4",
            reannotation_rounds=rounds, details=details,
        )
    return result, fake


MQM_ROUNDS = [
    "Critical:\nno-error\nMajor:\naccuracy/mistranslation - \"a\"\nMinor:\nno-error",
    "Critical:\nno-error\nMajor:\naccuracy/mistranslation - \"a\"\naccuracy/omission - \"b\"\nMinor:\nno-error",
    "Critical:\nnon-translation - \"c\"\nMajor:\nno-error\nMinor:\nno-error",
]


class TestMqmReannotation:
    def test_zero_rounds_no_details_matches_legacy_shape(self):
        """rounds=0 + details=False returns the flat score list (unchanged behavior)."""
        result, _ = _run("GEMBA-MQM", MQM_ROUNDS, rounds=0, details=False)
        assert result == [parse_mqm_answer(MQM_ROUNDS[0], full_desc=True)]

    def test_final_score_comes_from_last_round(self):
        result, _ = _run("GEMBA-MQM", MQM_ROUNDS, rounds=2)
        assert len(result) == 1
        record = result[0]
        assert record["score"] == parse_mqm_answer(MQM_ROUNDS[2], full_desc=True)
        assert record["annotation"] == MQM_ROUNDS[2]

    def test_trajectory_captures_every_round(self):
        result, _ = _run("GEMBA-MQM", MQM_ROUNDS, rounds=2)
        assert result[0]["trajectory"] == MQM_ROUNDS[:3]

    def test_conversation_grows_each_round(self):
        """Each round must add exactly one assistant+user turn to the prior conversation."""
        _, fake = _run("GEMBA-MQM", MQM_ROUNDS, rounds=2)
        prompts = fake.seen_prompts  # one per round: 0, 1, 2
        assert len(prompts) == 3
        assert len(prompts[1]) == len(prompts[0]) + 2
        assert len(prompts[2]) == len(prompts[1]) + 2
        # the appended assistant turn carries the previous round's raw annotation
        assert prompts[1][-2] == {"role": "assistant", "content": MQM_ROUNDS[0]}
        assert prompts[1][-1] == {"role": "user", "content": REANNOTATION_INSTRUCTION_MQM}

    def test_details_without_reannotation(self):
        """details=True with rounds=0 still returns the dict shape with a 1-entry trajectory."""
        result, _ = _run("GEMBA-MQM", MQM_ROUNDS, rounds=0, details=True)
        assert result[0]["annotation"] == MQM_ROUNDS[0]
        assert result[0]["trajectory"] == [MQM_ROUNDS[0]]


ESA_ROUNDS = [
    "Major:\naccuracy/mistranslation - \"a\"\nMinor:\nno-error",
    "Major:\naccuracy/mistranslation - \"a\"\naccuracy/omission - \"b\"\nMinor:\nno-error",
]


class TestEsaReannotation:
    def test_zero_rounds_no_details_unchanged(self):
        result, _ = _run("GEMBA-ESA", ESA_ROUNDS, rounds=0, details=False, ranking_answer="80")
        assert result == [80]

    def test_ranking_runs_on_final_spans(self):
        result, fake = _run("GEMBA-ESA", ESA_ROUNDS, rounds=1, ranking_answer="42")
        record = result[0]
        assert record["score"] == 42
        assert record["error_spans"] == ESA_ROUNDS[1]
        assert record["annotation"] == ESA_ROUNDS[1]
        assert record["trajectory"] == ESA_ROUNDS[:2]
        # the ranking prompt is a string and must embed the final (revised) spans
        ranking_prompt = next(p for p in fake.seen_prompts if isinstance(p, str))
        assert ESA_ROUNDS[1] in ranking_prompt


class RecordingGptApi:
    """Fake GptApi that records the kwargs of every bulk_request call."""

    def __init__(self, ranking_answer="80", **kwargs):
        self.ranking_answer = ranking_answer
        self.calls = []

    def bulk_request(self, df, model, parse, cache=None, max_tokens=None, response_format=None):
        prompt = df.iloc[0]["prompt"]
        stage = "ranking" if isinstance(prompt, str) else "error_spans"
        self.calls.append({"stage": stage, "max_tokens": max_tokens, "response_format": response_format})
        raw = self.ranking_answer if stage == "ranking" else "Major:\nno-error\nMinor:\nno-error"
        return [{
            "answer": parse(raw),
            "temperature": 0,
            "answer_id": 0,
            "prompt": prompt,
            "finish_reason": "stop",
            "model": model,
        } for _ in range(len(df))]


class TestEsaRequestParameters:
    """The ESA stages must carry a token budget, and the ranking stage a score schema."""

    def _run_esa(self, use_structured_output=True):
        fake = RecordingGptApi()
        with patch.object(utils, "GptApi", return_value=fake), \
             patch.object(utils.dc, "Cache", return_value=NoCache()):
            utils.get_gemba_scores(
                ["src1"], ["hyp1"], "English", "German", "GEMBA-ESA", "gpt-4",
                use_structured_output=use_structured_output,
            )
        return fake

    def test_stages_have_token_budget(self):
        fake = self._run_esa()
        spans_call = next(c for c in fake.calls if c["stage"] == "error_spans")
        ranking_call = next(c for c in fake.calls if c["stage"] == "ranking")
        assert spans_call["max_tokens"] == utils.ESA_ERROR_SPANS_MAX_TOKENS
        assert ranking_call["max_tokens"] == utils.ESA_RANKING_MAX_TOKENS

    def test_ranking_uses_score_schema_when_structured(self):
        fake = self._run_esa(use_structured_output=True)
        spans_call = next(c for c in fake.calls if c["stage"] == "error_spans")
        ranking_call = next(c for c in fake.calls if c["stage"] == "ranking")
        # Ranking gets the score schema; the free-form span stage stays schema-less.
        assert ranking_call["response_format"] == utils.RESPONSE_FORMATS["score"]
        assert spans_call["response_format"] is None

    def test_ranking_no_schema_when_structured_disabled(self):
        fake = self._run_esa(use_structured_output=False)
        ranking_call = next(c for c in fake.calls if c["stage"] == "ranking")
        assert ranking_call["response_format"] is None


class TestUnsupportedMethod:
    def test_reannotation_rejected_for_da(self):
        import pytest
        with patch.object(utils.dc, "Cache", return_value=NoCache()):
            with pytest.raises(Exception, match="only supported for GEMBA-MQM and GEMBA-ESA"):
                utils.get_gemba_scores(
                    ["s"], ["h"], "English", "German", "GEMBA-DA", "gpt-4",
                    reannotation_rounds=1,
                )
