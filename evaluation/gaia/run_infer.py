import asyncio
import functools
import os
import re

import huggingface_hub
import pandas as pd
from datasets import load_dataset

from evaluation.gaia.scorer import question_scorer
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
from opendevin.events.action import AgentFinishAction, CmdRunAction, MessageAction
from opendevin.events.observation import CmdOutputObservation
from opendevin.runtime.runtime import Runtime

DATASET_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': functools.partial(codeact_user_response, encapsulate_solution=True),
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have solved the question, please first send your answer to user through message and then exit.\n'
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


async def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    if instance['file_name'] != '':
        # if this question comes with a file, we need to save it to the workspace
        assert metadata.data_split is not None
        src_file = os.path.join(
            DATASET_CACHE_DIR, '2023', metadata.data_split, instance['file_name']
        )
        assert os.path.exists(src_file)
        dest_file = os.path.join('/workspace', instance['file_name'])
        await runtime.copy_to(src_file, dest_file)

        # rename to file.extension_name
        extension_name = instance['file_name'].split('.')[-1]
        action = CmdRunAction(
            command=f'mv /workspace/{instance["file_name"]} /workspace/file.{extension_name}'
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = await runtime.run_action(action)
        assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


async def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance['instance_id'], log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance["instance_id"]}.')

    if instance['file_name'] != '':
        extension_name = instance['file_name'].split('.')[-1]
        dest_file = os.path.join('/workspace', f'file.{extension_name}')
    else:
        dest_file = None

    # Prepare instruction
    instruction = f"{instance['Question']}\n"
    logger.info(f'Instruction: {instruction}')
    if dest_file:
        instruction += f"\n\nThe mentioned file is provided in the workspace at: {dest_file.split('/')[-1]}"

    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    instruction += 'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
    instruction += (
        'For example: The answer to the question is <solution> 42 </solution>.\n'
    )
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX.get(metadata.agent_class, '')
    logger.info(f'Instruction:\n{instruction}', extra={'msg_type': 'OBSERVATION'})

    runtime = await create_runtime(config, sid=instance['instance_id'])
    await initialize_runtime(runtime, instance)

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

    model_answer_raw = ''
    # get the last message or thought from the agent
    for event in state.history.get_events(reverse=True):
        if event.source == 'agent':
            if isinstance(event, AgentFinishAction):
                model_answer_raw = event.thought
                break
            elif isinstance(event, CmdRunAction):
                model_answer_raw = event.thought
                break
            elif isinstance(event, MessageAction):
                model_answer_raw = event.content
                break

    # attempt to parse model_answer
    model_answer = re.findall(r'<solution>(.*?)</solution>', model_answer_raw)
    if len(model_answer) == 0:
        logger.warning(f'Failed to parse model answer: {model_answer_raw}')
        model_answer = model_answer_raw
    else:
        model_answer = model_answer[0]

    logger.info(
        f'Final message: {model_answer} | Ground truth: {instance["Final answer"]}'
    )
    score = question_scorer(
        model_answer=model_answer, ground_truth=instance['Final answer']
    )
    test_result = {
        'score': score,
        'model_answer_raw': model_answer_raw,
        'model_answer': model_answer,
        'ground_truth': instance['Final answer'],
    }
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    # Save the output
    output = EvalOutput(
        instance_id=instance['instance_id'],
        instance=instance.to_dict(),
        instruction=instance['Question'],
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--level',
        type=str,
        help='gaia level to evaluate, eg. 2023_level1',
    )
    parser.add_argument(
        '--data-split',
        type=str,
        help='data split to evaluate, eg. test',
        default='validation',
    )
    args, _ = parser.parse_known_args()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name='gaia',
        agent_class=args.agent_cls,
        max_iterations=args.max_iterations,
        eval_note=args.eval_note,
        eval_output_dir=args.eval_output_dir,
        data_split=args.data_split,
        details={'gaia-level': args.level},
    )

    dataset = load_dataset('gaia-benchmark/GAIA', args.level)
    huggingface_hub.snapshot_download(
        'gaia-benchmark/GAIA',
        repo_type='dataset',
        local_dir=DATASET_CACHE_DIR,
    )
    gaia_tests = dataset[metadata.data_split].to_pandas()
    gaia_tests.rename(columns={'task_id': 'instance_id'}, inplace=True)

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    prepared_dataset = prepare_dataset(gaia_tests, output_file, args.eval_n_limit)

    asyncio.run(
        run_evaluation(
            dataset=prepared_dataset,
            metadata=metadata,
            output_file=output_file,
            num_workers=args.eval_num_workers,
            process_instance_func=process_instance,
        )
    )
