import asyncio
import json
import os
from typing import Any

import browsergym.webarena  # noqa F401 register webarena tasks as gym environments
import gymnasium as gym
import pandas as pd

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
from opendevin.events.action import (
    BrowseInteractiveAction,
    CmdRunAction,
    MessageAction,
)
from opendevin.events.observation import CmdOutputObservation
from opendevin.runtime.browser.browser_env import (
    BROWSER_EVAL_GET_GOAL_ACTION,
    BROWSER_EVAL_GET_REWARDS_ACTION,
)
from opendevin.runtime.runtime import Runtime

SUPPORTED_AGENT_CLS = {'BrowsingAgent'}


def get_config(
    metadata: EvalMetadata,
    env_id: str,
) -> AppConfig:
    base_url = os.environ.get('WEBARENA_BASE_URL', None)
    openai_api_key = os.environ.get('OPENAI_API_KEY', None)
    assert base_url is not None, 'WEBARENA_BASE_URL must be set'
    assert openai_api_key is not None, 'OPENAI_API_KEY must be set'

    config = AppConfig(
        default_agent=metadata.agent_class,
        run_as_devin=False,
        runtime='eventstream',
        max_iterations=metadata.max_iterations,
        sandbox=SandboxConfig(
            container_image='python:3.11-bookworm',
            enable_auto_lint=True,
            use_host_network=False,
            browsergym_eval_env=env_id,
            od_runtime_startup_env_vars={
                'BASE_URL': base_url,
                'OPENAI_API_KEY': openai_api_key,
                'SHOPPING': f'{base_url}:7770/',
                'SHOPPING_ADMIN': f'{base_url}:7780/admin',
                'REDDIT': f'{base_url}:9999',
                'GITLAB': f'{base_url}:8023',
                'WIKIPEDIA': f'{base_url}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing',
                'MAP': f'{base_url}:3000',
                'HOMEPAGE': f'{base_url}:4399',
            },
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(metadata.llm_config)
    return config


async def initialize_runtime(
    runtime: Runtime,
) -> dict:
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

    action = BrowseInteractiveAction(browser_actions=BROWSER_EVAL_GET_GOAL_ACTION)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    goal = obs.content

    logger.info(f"{'-' * 50} END Runtime Initialization Fn {'-' * 50}")
    return goal


async def complete_runtime(
    runtime: Runtime,
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Completion Fn {'-' * 50}")
    obs: CmdOutputObservation

    action = BrowseInteractiveAction(browser_actions=BROWSER_EVAL_GET_REWARDS_ACTION)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return {
        'rewards': json.loads(obs.content),
    }


async def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
):
    env_id = instance.instance_id
    config = get_config(metadata, env_id)

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, env_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {env_id}.')

    runtime = await create_runtime(config, sid=env_id)
    task_str = await initialize_runtime(runtime)

    state: State | None = await run_controller(
        config=config,
        task_str=task_str,
        runtime=runtime,
    )

    # ======= Attempt to evaluate the agent's environment impact =======

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.

    if state is None:
        raise ValueError('State should not be None.')

    metrics = state.metrics.get() if state.metrics else None

    # Instruction is the first message from the USER
    instruction = ''
    for event in state.history.get_events():
        if isinstance(event, MessageAction):
            instruction = event.content
            break

    return_val = await complete_runtime(runtime)
    logger.info(f'Return value from complete_runtime: {return_val}')
    reward = max(return_val['rewards'])

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    # Save the output
    output = EvalOutput(
        instance_id=env_id,
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'reward': reward,
        },
    )
    return output


if __name__ == '__main__':
    args = parse_arguments()

    dataset = pd.DataFrame(
        {
            'instance_id': [
                id
                for id in gym.envs.registry.keys()
                if id.startswith('browsergym/webarena')
            ]
        }
    )

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        args.dataset_name,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    asyncio.run(
        run_evaluation(
            instances,
            metadata,
            output_file,
            args.eval_num_workers,
            process_instance,
        )
    )
