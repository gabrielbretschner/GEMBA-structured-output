import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import (
    TEMPLATE_GEMBA_MQM,
    REANNOTATION_INSTRUCTION_MQM,
    apply_template,
    append_reannotation_turn,
    parse_mqm_answer,
)
from gemba.gemba_esa import (
    TEMPLATE_GEMBA_ESA_ERROR_SPANS,
    TEMPLATE_GEMBA_ESA_RANKING,
    REANNOTATION_INSTRUCTION_ESA,
)
from gemba.prompt import prompts, validate_number

# Structured output schemas for OpenAI's response_format parameter.
# These force models to return valid JSON matching the schema, avoiding
# verbose free-text responses that break parsing with newer models.
_ERROR_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}, "description": {"type": "string"}},
    "required": ["category", "description"],
    "additionalProperties": False,
}

RESPONSE_FORMATS = {
    "score": {
        "type": "json_schema",
        "json_schema": {
            "name": "score_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    },
    "mqm": {
        "type": "json_schema",
        "json_schema": {
            "name": "mqm_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "errors": {
                        "type": "object",
                        "properties": {
                            "critical": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                            "major": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                            "minor": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                        },
                        "required": ["critical", "major", "minor"],
                        "additionalProperties": False,
                    },
                },
                "required": ["errors"],
                "additionalProperties": False,
            },
        },
    },
}


def _get_response_format(method, use_structured_output):
    """Determine the response_format for a given GEMBA method."""
    if not use_structured_output:
        return None
    if method.startswith(("GEMBA-DA", "GEMBA-SQM")):
        return RESPONSE_FORMATS["score"]
    elif method == "GEMBA-MQM":
        return RESPONSE_FORMATS["mqm"]
    return None


def _run_annotation_rounds(gptapi, df, model, instruction, cache, max_tokens, response_format, rounds):
    """Run the initial annotation round plus ``rounds`` re-annotation rounds.

    ``df`` must already carry a ``prompt`` column with the initial (formatted) chat
    messages. Each re-annotation round continues the same conversation, appending the
    model's previous answer and ``instruction`` and re-requesting. Intermediate rounds
    use identity parsing so the raw annotation text is fed forward; nothing is scored here.

    Returns ``(final_answers, trajectories)``, both aligned to ``df`` rows. ``trajectories``
    holds the raw annotation from every round (length ``rounds + 1``).
    """
    identity = lambda x: x
    n = len(df)
    messages_list = list(df['prompt'])
    work_df = df.copy()
    trajectories = [[] for _ in range(n)]

    results = gptapi.bulk_request(work_df, model, identity, cache=cache, max_tokens=max_tokens, response_format=response_format)
    answers = [r['answer'] for r in results]
    for i in range(n):
        trajectories[i].append(answers[i])

    for _ in range(rounds):
        messages_list = [append_reannotation_turn(messages_list[i], answers[i], instruction) for i in range(n)]
        work_df = work_df.copy()
        work_df['prompt'] = pd.Series(messages_list, index=work_df.index)
        results = gptapi.bulk_request(work_df, model, identity, cache=cache, max_tokens=max_tokens, response_format=response_format)
        answers = [r['answer'] for r in results]
        for i in range(n):
            trajectories[i].append(answers[i])

    return answers, trajectories


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method, model,
                     list_mqm_errors=False, api_version=None, use_structured_output=True,
                     reference=None, base_url=None, reannotation_rounds=0, details=False):
    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = source_lang
    df['target_lang'] = target_lang
    if reference is not None:
        df['reference_seg'] = reference

    if reannotation_rounds > 0 and method not in ("GEMBA-MQM", "GEMBA-ESA"):
        raise Exception(f"Re-annotation is only supported for GEMBA-MQM and GEMBA-ESA, not {method}.")

    # Detailed output (final annotation + per-round trajectory) is produced whenever
    # the caller asks for it or whenever re-annotation rounds are run.
    use_details = details or reannotation_rounds > 0

    cache = dc.Cache(f'cache/{model}_{method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi(api_version=api_version, base_url=base_url)

    response_format = _get_response_format(method, use_structured_output)

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        if not use_details:
            parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=list_mqm_errors, full_desc=True)
            answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500, response_format=response_format)
            return list(pd.DataFrame(answers)['answer'])

        final_answers, trajectories = _run_annotation_rounds(
            gptapi, df, model, REANNOTATION_INSTRUCTION_MQM, cache,
            max_tokens=500, response_format=response_format, rounds=reannotation_rounds)
        return [
            {
                "score": parse_mqm_answer(ann, list_mqm_errors=list_mqm_errors, full_desc=True),
                "annotation": ann,
                "trajectory": traj,
            }
            for ann, traj in zip(final_answers, trajectories)
        ]
    elif method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[method]['prompt'], x), axis=1)
        validate = prompts[method]["validate_answer"]
        if not use_details:
            answers = gptapi.bulk_request(df, model, validate, cache=cache, max_tokens=500, response_format=response_format)
            return list(pd.DataFrame(answers)['answer'])

        raw = [r['answer'] for r in gptapi.bulk_request(df, model, lambda x: x, cache=cache, max_tokens=500, response_format=response_format)]
        return [
            {
                "score": validate(ann) if ann is not None else None,
                "annotation": ann,
                "trajectory": [ann],
            }
            for ann in raw
        ]
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        if reannotation_rounds == 0:
            error_spans = gptapi.bulk_request(df, model, lambda x: x, cache=cache)
            final_spans = list(pd.DataFrame(error_spans)['answer'])
            trajectories = [[s] for s in final_spans]
        else:
            final_spans, trajectories = _run_annotation_rounds(
                gptapi, df, model, REANNOTATION_INSTRUCTION_ESA, cache,
                max_tokens=None, response_format=None, rounds=reannotation_rounds)
        df['error_spans'] = pd.Series(final_spans, index=df.index)

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        scores = list(pd.DataFrame(gptapi.bulk_request(df, model, validate_number, cache=cache))['answer'])
        if not use_details:
            return scores
        return [
            {"score": sc, "annotation": sp, "error_spans": sp, "trajectory": traj}
            for sc, sp, traj in zip(scores, final_spans, trajectories)
        ]
    else:
        raise Exception(f"Method {method} not supported.")
