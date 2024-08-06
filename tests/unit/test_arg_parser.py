import os

import pytest

from opendevin.core.config import get_parser


@pytest.mark.skipif(
    os.getenv('TEST_IN_CI') != 'true',
    reason='This test is too dependent on pytest version. Skipping local execution, only run on CI',
)
def test_help_message(capsys):
    parser = get_parser()
    with pytest.raises(SystemExit):  # `--help` causes SystemExit
        parser.parse_args(['--help'])
    captured = capsys.readouterr()
    expected_help_message = """
usage: pytest [-h] [-d DIRECTORY] [-t TASK] [-f FILE] [-c AGENT_CLS]
              [-i MAX_ITERATIONS] [-b MAX_BUDGET_PER_TASK]
              [--eval-output-dir EVAL_OUTPUT_DIR]
              [--eval-n-limit EVAL_N_LIMIT]
              [--eval-num-workers EVAL_NUM_WORKERS] [--eval-note EVAL_NOTE]
              [-l LLM_CONFIG]

Run an agent with a specific task

options:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        The working directory for the agent
  -t TASK, --task TASK  The task for the agent to perform
  -f FILE, --file FILE  Path to a file containing the task. Overrides -t if
                        both are provided.
  -c AGENT_CLS, --agent-cls AGENT_CLS
                        Name of the default agent to use
  -i MAX_ITERATIONS, --max-iterations MAX_ITERATIONS
                        The maximum number of iterations to run the agent
  -b MAX_BUDGET_PER_TASK, --max-budget-per-task MAX_BUDGET_PER_TASK
                        The maximum budget allowed per task, beyond which the
                        agent will stop.
  --eval-output-dir EVAL_OUTPUT_DIR
                        The directory to save evaluation output
  --eval-n-limit EVAL_N_LIMIT
                        The number of instances to evaluate
  --eval-num-workers EVAL_NUM_WORKERS
                        The number of workers to use for evaluation
  --eval-note EVAL_NOTE
                        The note to add to the evaluation directory
  -l LLM_CONFIG, --llm-config LLM_CONFIG
                        Replace default LLM ([llm] section in config.toml)
                        config with the specified LLM config, e.g. "llama3"
                        for [llm.llama3] section in config.toml
"""

    actual_lines = captured.out.strip().split('\n')
    print('\n'.join(actual_lines))
    expected_lines = expected_help_message.strip().split('\n')

    # Ensure both outputs have the same number of lines
    assert len(actual_lines) == len(
        expected_lines
    ), 'The number of lines in the help message does not match.'

    # Compare each line
    for actual, expected in zip(actual_lines, expected_lines):
        assert (
            actual.strip() == expected.strip()
        ), f"Expected '{expected}', got '{actual}'"
