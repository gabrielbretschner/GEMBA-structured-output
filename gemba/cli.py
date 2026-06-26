import json
import os
import sys

from absl import app, flags

from gemba.utils import get_gemba_scores

FLAGS = flags.FLAGS
flags.DEFINE_string('method', "GEMBA-MQM", 'Which method to use?')
flags.DEFINE_string('model', "gpt-4", 'OpenAI model')
flags.DEFINE_string('source', None, 'Filepath to the source file.')
flags.DEFINE_string('hypothesis', None, 'Filepath to the translation file.')
flags.DEFINE_string('source_lang', None, 'Source language name.')
flags.DEFINE_string('target_lang', None, 'Target language name.')
flags.DEFINE_boolean('list_mqm_errors', False, 'List MQM errors.')
flags.DEFINE_string('api_version', None, 'API version for Azure OpenAI (overrides default).')
flags.DEFINE_boolean('no_structured_output', False, 'Disable structured output (JSON schema response_format).')
flags.DEFINE_string('base_url', None, 'Custom API base URL (e.g. http://localhost:11434 for Ollama).')
flags.DEFINE_integer('reannotation_rounds', 0, 'Number of re-annotation rounds after the initial GEMBA round (GEMBA-MQM / GEMBA-ESA only).')
flags.DEFINE_string('annotations_out', None, 'If set, write per-segment annotations and round trajectory as JSONL to this path.')

def main(argv):
    assert FLAGS.source is not None, "Source file must be provided."
    assert FLAGS.hypothesis is not None, "Hypothesis file must be provided."

    if not os.path.isfile(FLAGS.source):
        print(f"Source file {FLAGS.source} does not exist.")
        sys.exit(1)
    if not os.path.isfile(FLAGS.hypothesis):
        print(f"Hypothesis file {FLAGS.hypothesis} does not exist.")
        sys.exit(1)

    assert FLAGS.source_lang is not None, "Source language name must be provided."
    assert FLAGS.target_lang is not None, "Target language name must be provided."

    with open(FLAGS.source, 'r') as f:
        source = f.readlines()
    source = [x.strip() for x in source]
    with open(FLAGS.hypothesis, 'r') as f:
        hypothesis = f.readlines()
    hypothesis = [x.strip() for x in hypothesis]

    assert len(source) == len(hypothesis), "Source and hypothesis files must have the same number of lines."

    need_details = FLAGS.annotations_out is not None or FLAGS.reannotation_rounds > 0

    answers = get_gemba_scores(
        source, hypothesis, FLAGS.source_lang, FLAGS.target_lang,
        FLAGS.method, FLAGS.model, FLAGS.list_mqm_errors,
        api_version=FLAGS.api_version,
        use_structured_output=not FLAGS.no_structured_output,
        base_url=FLAGS.base_url,
        reannotation_rounds=FLAGS.reannotation_rounds,
        details=need_details,
    )

    if need_details:
        if FLAGS.annotations_out is not None:
            with open(FLAGS.annotations_out, 'w') as f:
                for record in answers:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for record in answers:
            print(record["score"])
    else:
        for answer in answers:
            print(answer)


def run():
    """Entry point used by console_scripts and `python -m gemba`."""
    app.run(main)


if __name__ == "__main__":
    run()
