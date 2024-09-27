import asyncio
import os
import tempfile
from typing import Any

import pandas as pd
from datasets import load_dataset

from evaluation.aider_bench.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    INSTRUCTIONS_ADDENDUM,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    parse_arguments,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation
from openhands.runtime.runtime import Runtime

# Configure visibility of unit tests to the Agent.
USE_UNIT_TESTS = os.environ.get('USE_UNIT_TESTS', 'false').lower() == 'true'
SKIP_NUM = os.environ.get('SKIP_NUM')
SKIP_NUM = (
    int(SKIP_NUM) if SKIP_NUM and SKIP_NUM.isdigit() and int(SKIP_NUM) >= 0 else None
)


def get_config(
    metadata: EvalMetadata,
) -> AppConfig:
    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        runtime='eventstream',
        max_iterations=metadata.max_iterations,
        sandbox=SandboxConfig(
            base_container_image='python:3.11-bookworm',
            enable_auto_lint=True,
            use_host_network=False,
            timeout=100,
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"\n{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}\n")
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, f'{instance.instance_name}.py')
        with open(file_path, 'w') as f:
            f.write(instance.signature)
        runtime.copy_to(
            file_path,
            '/workspace',
        )
        if USE_UNIT_TESTS:
            file_path = os.path.join(tmpdir, f'{instance.instance_name}_test.py')
            with open(file_path, 'w') as f:
                f.write(instance.test)
            runtime.copy_to(
                file_path,
                '/workspace',
            )
    logger.info(f"\n{'-' * 50} END Runtime Initialization Fn {'-' * 50}\n")


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"\n{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}\n")
    obs: CmdOutputObservation

    # Rewriting the test file to ignore any changes Agent may have made.
    script_name = f'{instance.instance_name}_test.py'
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, script_name)
        with open(file_path, 'w') as f:
            f.write(instance.test)
        runtime.copy_to(
            file_path,
            '/workspace',
        )
        logger.info(f'Running test file: {script_name}')

    action = CmdRunAction(
        command=f'python -m unittest {script_name}',
        keep_prompt=False,
    )
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    exit_code = 1
    if isinstance(obs, CmdOutputObservation):
        exit_code = obs.exit_code

    logger.info(f"\n{'-' * 50} END Runtime Completion Fn {'-' * 50}\n")

    runtime.close()

    return {
        'test_output': obs.content,
        'exit_code': exit_code,
    }


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, str(instance.instance_id), log_dir)
    else:
        logger.info(
            f'\nStarting evaluation for instance {str(instance.instance_id)}.\n'
        )

    # =============================================
    # build instruction
    # =============================================

    # Prepare instruction
    logger.info(instance)
    instruction = instance.instruction
    instruction += INSTRUCTIONS_ADDENDUM.format(
        signature_file=f'{instance.instance_name}.py',
    )
    if USE_UNIT_TESTS:
        print(f'\nInstruction to run test_file: {instance.instance_name}_test.py\n')
        instruction += (
            f'Use `python -m unittest {instance.instance_name}_test.py` to run the test_file '
            'and verify the correctness of your solution. DO NOT EDIT the test file.\n\n'
        )

    instruction += (
        'IMPORTANT: You should ONLY interact with the environment provided '
        'to you AND NEVER ASK FOR HUMAN HELP.\n'
    )
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += INST_SUFFIXES[metadata.agent_class]

    # =============================================
    # create sandbox and run the agent
    # =============================================

    runtime: Runtime = create_runtime(config, sid=str(instance.instance_id))

    initialize_runtime(runtime, instance=instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = asyncio.run(
        run_controller(
            config=config,
            task_str=instruction,
            runtime=runtime,
            fake_user_response_fn=FAKE_RESPONSES[metadata.agent_class],
        )
    )
    if state is None:
        raise ValueError('State should not be None.')

    # # =============================================
    # # result evaluation
    # # =============================================

    return_val = complete_runtime(runtime, instance)
    exit_code = return_val['exit_code']
    test_output = return_val['test_output']

    errors = []
    test_cases = None
    if test_output.find('SyntaxError') != -1:
        errors += 'SyntaxError'
    elif test_output.find('IndentationError') != -1:
        errors += 'IndentationError'
    else:
        test_cases = test_output[: test_output.find('\r')]

    test_result = {
        'exit_code': exit_code,
        'test_cases': test_cases,
        'errors': errors,
    }

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()
    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = EvalOutput(
        instance_id=str(instance.instance_id),
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result=test_result,
    )
    return output


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset('RajMaheshwari/Exercism-Python')
    aider_bench_tests = dataset['train'].to_pandas()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'AiderBench',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    # Parse dataset IDs if provided
    eval_ids = None
    if args.eval_ids:
        eval_ids = str(args.eval_ids).split(',')
        logger.info(f'\nUsing specific dataset IDs: {eval_ids}\n')

    instances = prepare_dataset(
        aider_bench_tests,
        output_file,
        args.eval_n_limit,
        eval_ids=eval_ids,
        skip_num=SKIP_NUM,
    )

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
    )
