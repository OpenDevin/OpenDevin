import json
import logging
import multiprocessing as mp
import os
import pathlib
import subprocess
import time
import traceback
from typing import Any, Awaitable, Callable, TextIO

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from openhands.controller.state.state import State
from openhands.core.config import LLMConfig
from openhands.core.logger import get_console_handler
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.action.message import MessageAction


class EvalMetadata(BaseModel):
    agent_class: str
    llm_config: LLMConfig
    max_iterations: int
    eval_output_dir: str
    start_time: str
    git_commit: str
    dataset: str | None = None
    data_split: str | None = None
    details: dict[str, Any] | None = None

    def model_dump(self, *args, **kwargs):
        dumped_dict = super().model_dump(*args, **kwargs)
        # avoid leaking sensitive information
        dumped_dict['llm_config'] = self.llm_config.to_safe_dict()
        return dumped_dict

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        # avoid leaking sensitive information
        dumped_dict['llm_config'] = self.llm_config.to_safe_dict()
        logger.debug(f'Dumped metadata: {dumped_dict}')
        return json.dumps(dumped_dict)


class EvalOutput(BaseModel):
    # NOTE: User-specified
    instance_id: str
    # output of the evaluation
    # store anything that is needed for the score calculation
    test_result: dict[str, Any]

    instruction: str | None = None

    # Interaction info
    metadata: EvalMetadata | None = None
    # list[tuple[dict[str, Any], dict[str, Any]]] - for compatibility with the old format
    history: (
        list[dict[str, Any]] | list[tuple[dict[str, Any], dict[str, Any]]] | None
    ) = None
    llm_completions: list[dict[str, Any]]
    metrics: dict[str, Any] | None = None
    error: str | None = None

    # Optionally save the input test instance
    instance: dict[str, Any] | None = None

    def model_dump(self, *args, **kwargs):
        dumped_dict = super().model_dump(*args, **kwargs)
        # Remove None values
        dumped_dict = {k: v for k, v in dumped_dict.items() if v is not None}
        # Apply custom serialization for metadata (to avoid leaking sensitive information)
        if self.metadata is not None:
            dumped_dict['metadata'] = self.metadata.model_dump()
        return dumped_dict

    def model_dump_json(self, *args, **kwargs):
        dumped = super().model_dump_json(*args, **kwargs)
        dumped_dict = json.loads(dumped)
        # Apply custom serialization for metadata (to avoid leaking sensitive information)
        if 'metadata' in dumped_dict:
            dumped_dict['metadata'] = json.loads(self.metadata.model_dump_json())
        return json.dumps(dumped_dict)


def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please first send your answer to user through message and then <execute_bash> exit </execute_bash>.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = state.history.get_last_action()
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history.get_events()
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, run: <execute_bash> exit </execute_bash>.\n'
            )
    return msg


def cleanup():
    print('Cleaning up child processes...')
    for process in mp.active_children():
        print(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def make_metadata(
    llm_config: LLMConfig,
    dataset_name: str,
    agent_class: str,
    max_iterations: int,
    eval_note: str | None,
    eval_output_dir: str,
    data_split: str | None = None,
    details: dict[str, Any] | None = None,
) -> EvalMetadata:
    model_name = llm_config.model.split('/')[-1]
    model_path = model_name.replace(':', '_')
    eval_note = f'_N_{eval_note}' if eval_note else ''

    eval_output_path = os.path.join(
        eval_output_dir,
        dataset_name,
        agent_class,
        f'{model_path}_maxiter_{max_iterations}{eval_note}',
    )

    pathlib.Path(eval_output_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(eval_output_path, 'logs')).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f'Using evaluation output directory: {eval_output_path}')

    metadata = EvalMetadata(
        agent_class=agent_class,
        llm_config=llm_config,
        max_iterations=max_iterations,
        eval_output_dir=eval_output_path,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip(),
        dataset=dataset_name,
        data_split=data_split,
        details=details,
    )
    metadata_json = metadata.model_dump_json()
    logger.info(f'Metadata: {metadata_json}')
    with open(os.path.join(eval_output_path, 'metadata.json'), 'w') as f:
        f.write(metadata_json)

    return metadata


def prepare_dataset(
    dataset: pd.DataFrame,
    output_file: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
):
    assert (
        'instance_id' in dataset.columns
    ), "Expected 'instance_id' column in the dataset. You should define your own unique identifier for each instance and use it as the 'instance_id' column."
    id_column = 'instance_id'
    logger.info(f'Writing evaluation output to {output_file}')
    finished_ids: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                finished_ids.add(str(data[id_column]))
        logger.warning(
            f'\nOutput file {output_file} already exists. Loaded {len(finished_ids)} finished instances.'
        )

    if eval_ids:
        eval_ids_converted = [dataset[id_column].dtype.type(id) for id in eval_ids]
        dataset = dataset[dataset[id_column].isin(eval_ids_converted)]
        logger.info(f'Limiting evaluation to {len(eval_ids)} specific instances.')
    elif skip_num and skip_num >= 0:
        skip_num = min(skip_num, len(dataset))
        dataset = dataset.iloc[skip_num:]
        logger.info(
            f'Starting evaluation with skipping first {skip_num} instances ({len(dataset)} instances to run).'
        )
        if eval_n_limit and eval_n_limit > 0:
            dataset = dataset.head(eval_n_limit)
            logger.info(f'Limiting evaluation to {eval_n_limit} instances.')
    elif eval_n_limit and eval_n_limit > 0:
        dataset = dataset.head(eval_n_limit)
        logger.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    new_dataset = [
        instance
        for _, instance in dataset.iterrows()
        if str(instance[id_column]) not in finished_ids
    ]
    logger.info(
        f'Finished instances: {len(finished_ids)}, Remaining instances: {len(new_dataset)}'
    )

    return pd.DataFrame(new_dataset)


def update_progress(
    result: EvalOutput,
    pbar: tqdm,
    output_fp: TextIO,
):
    """Update the progress bar and write the result to the output file."""
    pbar.update(1)
    pbar.set_description(f'Instance {result.instance_id}')
    pbar.set_postfix_str(f'Test Result: {result.test_result}')
    logger.info(
        f'Finished evaluation for instance {result.instance_id}: {str(result.test_result)[:300]}...\n'
    )
    output_fp.write(json.dumps(result.model_dump()) + '\n')
    output_fp.flush()


def _process_instance_wrapper(
    process_instance_func: Callable[[pd.Series, EvalMetadata, bool], EvalOutput],
    instance: pd.Series,
    metadata: EvalMetadata,
    use_mp: bool,
    max_retries: int = 5,
) -> EvalOutput:
    """Wrap the process_instance_func to handle retries and errors.

    Retry an instance up to max_retries times if it fails (e.g., due to transient network/runtime issues).
    """
    for attempt in range(max_retries + 1):
        try:
            result = process_instance_func(instance, metadata, use_mp)
            return result
        except Exception as e:
            error = str(e)
            stacktrace = traceback.format_exc()
            if attempt == max_retries:
                logger.exception(e)
                msg = (
                    '-' * 10
                    + '\n'
                    + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                    + '\n'
                    + f'[Encountered after {max_retries} retries. Please check the logs and report the issue.]'
                    + '-' * 10
                )
                # Raise an error after all retries & stop the evaluation
                logger.exception(e)
                raise RuntimeError(
                    f'Maximum error retries reached for instance {instance.instance_id}'
                ) from e
            msg = (
                '-' * 10
                + '\n'
                + f'Error in instance [{instance.instance_id}]: {error}. Stacktrace:\n{stacktrace}'
                + '\n'
                + '-' * 10
                + f'[The above error occurred. Retrying... (attempt {attempt + 1} of {max_retries})]'
                + '-' * 10
                + '\n'
            )
            logger.error(msg)
            if use_mp:
                print(msg)  # use print to directly print to console
            time.sleep(5)


def _process_instance_wrapper_mp(args):
    """Wrapper for multiprocessing, especially for imap_unordered."""
    return _process_instance_wrapper(*args)


def run_evaluation(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_file: str,
    num_workers: int,
    process_instance_func: Callable[
        [pd.Series, EvalMetadata, bool], Awaitable[EvalOutput]
    ],
    max_retries: int = 5,  # number of retries for each instance
):
    use_multiprocessing = num_workers > 1

    if metadata is not None:
        logger.info(
            f'Evaluation started with Agent {metadata.agent_class}:\n'
            f'model {metadata.llm_config.model}, max iterations {metadata.max_iterations}.\n'
        )
    else:
        logger.info(f'Evaluation started with {num_workers} workers.')

    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')
    output_fp = open(output_file, 'a')

    try:
        if use_multiprocessing:
            with mp.Pool(num_workers) as pool:
                args_iter = (
                    (process_instance_func, instance, metadata, True, max_retries)
                    for _, instance in dataset.iterrows()
                )
                results = pool.imap_unordered(_process_instance_wrapper_mp, args_iter)
                for result in results:
                    update_progress(result, pbar, output_fp)
        else:
            for _, instance in dataset.iterrows():
                result = _process_instance_wrapper(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                )
                update_progress(result, pbar, output_fp)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()

    output_fp.close()
    logger.info('\nEvaluation finished.\n')


def reset_logger_for_multiprocessing(
    logger: logging.Logger, instance_id: str, log_dir: str
):
    """Reset the logger for multiprocessing.

    Save logs to a separate file for each process, instead of trying to write to the
    same file/console from multiple processes.
    """
    # Set up logger
    log_file = os.path.join(
        log_dir,
        f'instance_{instance_id}.log',
    )
    # Remove all existing handlers from logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # add console handler to print ONE line
    console_handler = get_console_handler(log_level=logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            f'Instance {instance_id} - ' + '%(asctime)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(console_handler)
    logger.info(
        f'Starting evaluation for instance {instance_id}.\n'
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )
    # Only log WARNING or higher to console
    console_handler.setLevel(logging.WARNING)

    # Log INFO and above to file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
