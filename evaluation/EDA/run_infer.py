import asyncio
import os

import pandas as pd
from datasets import load_dataset

from evaluation.EDA.game import Q20Game, Q20GameCelebrity
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
    get_parser,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import create_runtime, run_controller

game = None


def codeact_user_response_eda(state: State) -> str:
    global game
    model_guess = ''

    # retrieve the latest model message from history
    if state.history:
        model_guess = state.history.get_last_agent_message()

    assert game is not None, 'Game is not initialized.'
    msg = game.generate_user_response(model_guess)
    game.curr_turn += 1
    logger.info(f'Model guess: {model_guess}')
    logger.info(f'Answer response: {msg}')
    if 'bingo!' in msg.lower():
        return '/exit'
    return msg


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response_eda,
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
            enable_auto_lint=False,
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
    instance_id = instance['text'].strip()

    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # Prepare instruction
    _game_class = {'eda-things': Q20Game, 'eda-celebs': Q20GameCelebrity}

    guesser_kargs = {
        'max_new_tokens': 64,
        'temperature': 0.8,
        'repetition_penalty': 1.0,
        'do_sample': True,
    }  # no penalty

    # Use codeactagent as guesser_model
    global game
    assert metadata.dataset is not None
    assert metadata.details is not None
    game = _game_class[metadata.dataset](
        item=instance['text'].strip(),
        answerer_model=metadata.details['answerer_model'],
        guesser_model=None,
        num_turns=metadata.max_iterations,
        openai_api_key=metadata.details['openai_api_key'],
        guesser_kargs=guesser_kargs,
    )

    instruction = f'{game.first_user_utterance}'
    logger.info(f'Instruction: {instruction}')
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    runtime = await create_runtime(config, sid=instance['text'].strip())

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

    final_message = state.history.get_last_agent_message()

    logger.info(f'Final message: {final_message} | Ground truth: {instance["text"]}')
    test_result = game.reward()
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    # Save the output
    output = EvalOutput(
        instance_id=instance_id,
        instance=instance.to_dict(),
        instruction=instruction,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
        test_result={
            'success': test_result,
            'final_message': final_message,
            'ground_truth': instance['text'],
        },
    )
    return output


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--answerer_model', '-a', default='gpt-3.5-turbo', help='answerer model'
    )
    parser.add_argument(
        '--dataset',
        default='things',
        choices=['things', 'celebs'],
        type=str,
        help='dataset to be used',
    )
    parser.add_argument(
        '--OPENAI_API_KEY', type=str, required=True, help='Your OpenAI API key'
    )
    parser.add_argument(
        '--data-split',
        default='test',
        type=str,
        help='data split, eg, test',
    )
    args, _ = parser.parse_known_args()

    eda_dataset = load_dataset(
        'yizheapple/entity-deduction-arena', name=args.dataset, split=args.data_split
    )
    eda_dataset.rename(columns={'text': 'instance_id'}, inplace=True)

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        f'eda-{args.dataset}',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        data_split=args.data_split,
        details={
            'answerer_model': str(args.answerer_model),
            'openai_api_key': str(args.OPENAI_API_KEY),
        },
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    prepared_dataset = prepare_dataset(
        eda_dataset.to_pandas(), output_file, args.eval_n_limit
    )

    asyncio.run(
        run_evaluation(
            prepared_dataset,
            metadata,
            output_file,
            args.eval_num_workers,
            process_instance,
        )
    )
