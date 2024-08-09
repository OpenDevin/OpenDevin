import asyncio
import os
from typing import Any

import pandas as pd

from evaluation.toolqa.utils import encode_question, eval_answer, get_data
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    codeact_user_response,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from opendevin.controller.state.state import State
from opendevin.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    get_parser,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import create_runtime, run_controller
from opendevin.events.action import CmdRunAction
from opendevin.events.observation import CmdOutputObservation
from opendevin.runtime.runtime import Runtime

AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have completed the request, please run the following command: <execute_bash> exit </execute_bash>.\n'
}


def get_config(
    metadata: EvalMetadata,
) -> AppConfig:
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_devin=False,
        runtime='eventstream',
        max_iterations=metadata.max_iterations,
        sandbox=SandboxConfig(
            container_image='python:3.11-bookworm',
            enable_auto_lint=True,
            use_host_network=False,
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    return config


async def initialize_runtime(runtime: Runtime):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    await runtime.add_env_vars({'WOLFRAM_ALPHA_APPID': args.wolfram_alpha_appid})

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


async def process_instance(
    instance: Any, metadata: EvalMetadata, reset_logger: bool = True
):
    config = get_config(metadata)

    qid = instance.qid
    question = instance.question
    answer = instance.answer

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, qid, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {qid}.')

    # Prepare instruction
    instruction = encode_question(question)
    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]
    logger.info(f'Instruction:\n{instruction}', extra={'msg_type': 'OBSERVATION'})

    runtime = await create_runtime(config, sid=qid)
    await initialize_runtime(runtime)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
    )
    # ======= Attempt to evaluate the agent's edits =======
    # If you are working on simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.

    if state is None:
        raise ValueError('State should not be None.')

    # retrieve the last message from the agent
    model_answer_raw = state.history.get_last_agent_message()

    # attempt to parse model_answer
    correct = eval_answer(str(model_answer_raw), str(answer))
    logger.info(f'Final message: {model_answer_raw} | Correctness: {correct}')

    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    # Save the output
    output = EvalOutput(
        instance_id=qid,
        test_result={
            'model_answer_raw': model_answer_raw,
            'correct': correct,
        },
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        help='Which dataset to evaluate from ToolQA. ToolQA contains 8 datasets, namely agenda, airbnb, coffee, dblp, flight, gsm8k, scirex, yelp. For example, the default is --dataset flight.',
        default='flight',
    )
    parser.add_argument(
        '--hardness',
        type=str,
        help='Which level of difficulty to evaluate from ToolQA. ToolQA contains 2 levels of hardness, namely easy and hard. For example, the default is --hardness easy.',
        default='easy',
    )
    parser.add_argument(
        '--wolfram_alpha_appid',
        type=str,
        help='wolfram alpha appid to use for wolfram alpha related tests',
        default='YOUR_WOLFRAMALPHA_APPID',
    )
    args, _ = parser.parse_known_args()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    dataset = ''
    hardness = ''
    dataset_choices = [
        'agenda',
        'airbnb',
        'coffee',
        'dblp',
        'flight',
        'gsm8k',
        'scirex',
        'yelp',
        'genda',
    ]
    if args.dataset not in dataset_choices:
        raise ValueError(
            'Please choose from agenda, airbnb, coffee, dblp, flight, gsm8k, scirex, yelp for dataset.'
        )
    if args.hardness not in ['easy', 'hard']:
        raise ValueError('Please choose from easy and hard for hardness.')

    toolqa_test = pd.DataFrame(get_data(dataset, hardness))
    toolqa_test.rename(columns={'qid': 'instance_id'}, inplace=True)

    metadata = make_metadata(
        llm_config,
        f'toolqa-{args.dataset}-{args.hardness}',
        args.agent_cls,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(toolqa_test, output_file, args.eval_n_limit)
    asyncio.run(
        run_evaluation(
            instances, metadata, output_file, args.eval_num_workers, process_instance
        )
    )
