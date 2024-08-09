import asyncio
import os
import re
import tempfile
from typing import Any

import pandas as pd
from datasets import load_dataset

from evaluation.agent_bench.helper import (
    FAKE_RESPONSES,
    INST_SUFFIXES,
    compare_results,
    create_sh_file,
)
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
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
    parse_arguments,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import create_runtime, run_controller
from opendevin.events.action import AgentFinishAction, CmdRunAction, MessageAction
from opendevin.events.observation import CmdOutputObservation
from opendevin.runtime.runtime import Runtime


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

    # Set instance id
    action = CmdRunAction(command='mkdir -p /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    assert obs.exit_code == 0

    init_cmd = instance.init
    if init_cmd is not None:
        script_name = f'{instance.instance_id}_init.sh'

        with tempfile.TemporaryDirectory() as tmpdir:
            host_script_path = os.path.join(tmpdir, script_name)
            create_sh_file(host_script_path, init_cmd)
            await runtime.copy_to(
                host_script_path,
                '/workspace',
            )

        logger.info(f'Running init script: {script_name}')
        action = CmdRunAction(command=f'chmod +x ./{script_name} && ./{script_name}')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = await runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")


async def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}")
    obs: CmdOutputObservation

    agent_answer = None
    get_agent_result_cmd = instance.get_agent_result
    if get_agent_result_cmd is not None:
        script_name = 'get_agent_result.sh'

        with tempfile.TemporaryDirectory() as tmpdir:
            host_script_path = os.path.join(tmpdir, script_name)
            create_sh_file(host_script_path, get_agent_result_cmd)
            await runtime.copy_to(
                host_script_path,
                '/workspace',
            )
            logger.info(f'Running get agent result cmd: {script_name}')

        action = CmdRunAction(
            command=f'chmod +x ./{script_name} && ./{script_name}',
            keep_prompt=False,
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = await runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
        agent_answer = obs.content
    # IF the agent answer is not found, retrieve it from the history
    # We wait until the controller finishes

    final_ans = None
    if instance.ground_truth is not None:
        final_ans = instance.ground_truth
    else:
        get_ground_truth_cmd = instance.get_ground_truth
        if get_ground_truth_cmd is not None:
            script_name = 'get_ground_truth.sh'
            with tempfile.TemporaryDirectory() as tmpdir:
                host_script_path = os.path.join(tmpdir, script_name)
                create_sh_file(host_script_path, get_ground_truth_cmd)
                await runtime.copy_to(
                    host_script_path,
                    '/workspace',
                )
            logger.info(f'Running get ground truth cmd: {script_name}')

            action = CmdRunAction(
                command=f'chmod +x ./{script_name} && ./{script_name}',
                keep_prompt=False,
            )
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = await runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            final_ans = obs.content

    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return {
        'final_ans': final_ans,
        'agent_answer': agent_answer,
    }


async def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # =============================================
    # build instruction
    # =============================================

    # Prepare instruction
    instruction = (
        f'Please fix the following issue.\n'
        'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
        'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
        'For example: The answer to the question is <solution> 42 </solution>.\n'
        '# Problem \n'
        f'{instance.description}\n\n'
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

    runtime: Runtime = await create_runtime(config, sid=instance.instance_id)

    await initialize_runtime(runtime, instance=instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=FAKE_RESPONSES[metadata.agent_class],
    )
    if state is None:
        raise ValueError('State should not be None.')

    # =============================================
    # result evaluation
    # =============================================

    return_val = await complete_runtime(runtime, instance)
    agent_answer = return_val['agent_answer']
    final_ans = return_val['final_ans']

    # If the agent answer is not found, retrieve it from the history
    if agent_answer is None:
        agent_answer = ''
        logger.info('Retrieving agent answer from history.')
        raw_ans = ''

        # retrieve the last agent message or thought
        for event in state.history.get_events(reverse=True):
            if event.source == 'agent':
                if isinstance(event, AgentFinishAction):
                    raw_ans = event.thought
                    break
                elif isinstance(event, MessageAction):
                    raw_ans = event.content
                    break
                elif isinstance(event, CmdRunAction):
                    raw_ans = event.thought
                    break

        # parse the answer for a solution tag
        agent_answer = re.findall(r'<solution>(.*?)</solution>', raw_ans, re.DOTALL)
        if len(agent_answer) == 0:
            logger.warning(f'Failed to parse model answer: {raw_ans}')
            agent_answer = raw_ans
        else:
            agent_answer = agent_answer[0]

    comparison_method = instance.comparison_method
    logger.info(
        f'Final message: {agent_answer} | Ground truth: {final_ans} | Comparison method: {comparison_method}'
    )
    test_result = compare_results(comparison_method, agent_answer, final_ans)

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    metrics = state.metrics.get() if state.metrics else None

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'agent_answer': agent_answer,
            'final_answer': final_ans,
            'check_method': comparison_method,
            'result': test_result,
        },
    )
    return output


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset('iFurySt/AgentBench')
    agent_bench_tests = dataset['osbench'].to_pandas()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'AgentBench-OS',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(agent_bench_tests, output_file, args.eval_n_limit)

    asyncio.run(
        run_evaluation(
            instances, metadata, output_file, args.eval_num_workers, process_instance
        )
    )
