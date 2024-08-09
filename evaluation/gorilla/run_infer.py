import asyncio
import json
import os

import pandas as pd

from evaluation.gorilla.utils import encode_question, get_data_for_hub
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


async def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)
    instance_id = instance['question_id']
    question = instance['question']

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # Prepare instruction
    instruction = encode_question(question, instance['hub'])
    instruction += 'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]
    # logger.info(f'Instruction:\n{instruction}', extra={'msg_type': 'OBSERVATION'})

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    runtime = await create_runtime(config, sid=instance_id)
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        runtime=runtime,
        fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
            metadata.agent_class
        ),
    )
    # ======= Attempt to evaluate the agent's edits =======
    # If you are working on simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.

    if state is None:
        raise ValueError('State should not be None.')

    # retrieve the last message from the agent
    model_answer_raw = state.history.get_last_agent_message()

    # attempt to parse model_answer
    ast_eval_fn = instance['ast_eval']
    correct, hallucination = ast_eval_fn(instance_id, model_answer_raw)
    metrics = state.metrics.get() if state.metrics else None
    logger.info(
        f'Final message: {model_answer_raw} | Correctness: {correct} | Hallucination: {hallucination}'
    )

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    output = EvalOutput(
        instance_id=instance_id,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'text': model_answer_raw,
            'correct': correct,
            'hallucination': hallucination,
        },
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--hubs',
        type=str,
        help='Which hubs to evaluate from APIBench. APIBench contains 3 hubs, namely huggingface, torch, and tensorflow. You could choose one or more from hf, torch, or tf, separated by commas. For example, the default is --hub hf,torch,tf.',
        default='hf,torch,tf',
    )
    args, _ = parser.parse_known_args()

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    hubs = args.hubs.split(',')
    if len(hubs) == 0:
        raise ValueError('Please choose at least one from hf, torch, and tf for hubs.')

    dfs = []
    for hub in hubs:
        logger.info(f'Evaluating APIBench {hub} test')
        df = get_data_for_hub(hub)
        dfs.append(df)
    dataset_df = pd.concat(dfs)
    dataset_df.rename(columns={'question_id': 'instance_id'}, inplace=True)

    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name=f'gorilla-{hub}',
        agent_class=args.agent_cls,
        max_iterations=args.max_iterations,
        eval_note=args.eval_note,
        eval_output_dir=args.eval_output_dir,
        data_split=args.data_split,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    dataset = prepare_dataset(
        dataset_df, output_file=output_file, eval_n_limit=args.eval_n_limit
    )

    asyncio.run(
        run_evaluation(
            dataset=dataset,
            metadata=metadata,
            output_file=output_file,
            num_workers=args.eval_num_workers,
            process_instance_func=process_instance,
        )
    )

    # Read the output file and calculate the accuracy
    total_correct = 0
    total_hallucination = 0
    output = []
    with open(output_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['test_result']['correct']:
                total_correct += 1
            if data['test_result']['hallucination']:
                total_hallucination += 1
            output.append(data)
    logger.info(
        f'Evaluation finished for {hub}. Total: {len(output)}; Correct: {total_correct}; Hallucination: {total_hallucination}. Accuracy: {total_correct / len(output)}'
    )
