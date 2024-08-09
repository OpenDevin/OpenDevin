import asyncio
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import zipfile
from typing import Any

import pandas as pd
from datasets import load_dataset
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

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
from opendevin.events.action import CmdRunAction, MessageAction
from opendevin.events.observation import CmdOutputObservation
from opendevin.runtime.runtime import Runtime


def codeact_user_response(state: State) -> str:
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have completed the SQL, please run the following command: <execute_bash> exit </execute_bash>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP OR USE THE INTERNET TO SOLVE THIS TASK.\n'
    )
    if state.history:
        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history.get_events()
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) > 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, run: <execute_bash> exit </execute_bash>.\n'
            )
    return msg


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': 'When you think you have fixed the issue through code changes, please run the following command: <execute_bash> exit </execute_bash>.\n'
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


def execute_sql(db_path, gen_sql, gold_sql):
    """Execute the generated SQL and the ground truth SQL and compare the results."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(gen_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(gold_sql)
        ground_truth_res = cursor.fetchall()
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
    return res


LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')


def load_bird():
    """Main function to handle the flow of downloading, processing, and loading the bird dataset."""

    def _download_bird():
        """Downloads and extracts the bird dataset from a specified URL into a local directory."""
        devset_path = os.path.join(LOCAL_DATASET_PATH, 'dev')
        if not os.path.exists(devset_path):
            logger.info(
                f'{LOCAL_DATASET_PATH} folder does not exist, starting download and extraction...'
            )
            os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)

            download_url = 'https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip'
            download_path = os.path.join(LOCAL_DATASET_PATH, 'dev.zip')
            if not os.path.exists(download_path):
                logger.info('Start Downloading...')
                subprocess.run(['wget', download_url, '-O', download_path])
                logger.info('Download completed.')

            devset_path = os.path.join(LOCAL_DATASET_PATH, 'dev')
            if not os.path.exists(devset_path):
                logger.info('Start Extracting...')
                os.makedirs(devset_path, exist_ok=True)
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(devset_path)
                # move everything in 'dev_20240627' to the root folder
                for file in os.listdir(os.path.join(devset_path, 'dev_20240627')):
                    os.rename(
                        os.path.join(devset_path, 'dev_20240627', file),
                        os.path.join(devset_path, file),
                    )
                os.rmdir(os.path.join(devset_path, 'dev_20240627'))
                logger.info('Extraction completed.')

            # extract databases
            database_path = os.path.join(devset_path, 'dev_databases.zip')
            assert os.path.exists(database_path)
            logger.info('Start Extracting...')
            with zipfile.ZipFile(database_path, 'r') as zip_ref:
                zip_ref.extractall(devset_path)
            logger.info('Extraction completed.')
        else:
            logger.info(f'{LOCAL_DATASET_PATH} folder already exists.')
        return devset_path

    def _extract_create_table_prompt(db_path, limit_value=0):
        """Generates a SQL prompt with CREATE TABLE statements and sample data from the database."""
        table_query = "SELECT * FROM sqlite_master WHERE type='table';"
        tables = sqlite3.connect(db_path).cursor().execute(table_query).fetchall()
        prompt = ''
        for table in tables:
            table_name = table[1]
            create_table_statement = table[-1]

            table_info_query = f'PRAGMA table_info(`{table_name}`);'
            top_k_row_query = f'SELECT * FROM {table_name} LIMIT {limit_value};'
            try:
                headers = [
                    x[1]
                    for x in sqlite3.connect(db_path)
                    .cursor()
                    .execute(table_info_query)
                    .fetchall()
                ]
            except Exception:
                logger.error(f'Error Connection: {table_info_query}, {top_k_row_query}')
                exit(0)

            prompt += create_table_statement + ';\n'
            if limit_value > 0:
                top_k_rows = (
                    sqlite3.connect(db_path)
                    .cursor()
                    .execute(top_k_row_query)
                    .fetchall()
                )
                prompt += (
                    f"/*\n3 example rows:\n{top_k_row_query}\n{'    '.join(headers)}\n"
                )
                for row in top_k_rows:
                    row = [str(x) for x in row]
                    row = [x if x is not None else '' for x in row]
                    prompt += '    '.join(row) + '\n'
                prompt += '*/\n'
            prompt += '\n'
        return prompt

    def _create_prompt(e, database_path):
        """Create a prompt for the given example"""
        db_id = e['db_id']
        db_path = pathlib.Path(database_path) / db_id / f'{db_id}.sqlite'

        # Extract the CREATE TABLE statements and sample data from the database
        prompt = _extract_create_table_prompt(db_path)
        prompt += f"-- External Knowledge: {e['evidence']}\n\n"
        prompt += '-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above.\n\n'
        prompt += '-- Using valid SQLite, answer the following questions for the tables provided above.\n'
        prompt += f"Question: {e['question']}\n"

        return prompt

    def _process_bird(dataset_path):
        """Processes the raw bird dataset into a structured format and saves it as JSON."""
        processed_path = os.path.join(LOCAL_DATASET_PATH, 'dev', 'processed_dev.json')
        if not os.path.exists(processed_path):
            logger.info(
                f'{processed_path} folder does not exist, starting processing...'
            )
            raw_data_path = os.path.join(LOCAL_DATASET_PATH, 'dev', 'dev.json')
            database_path = os.path.join(LOCAL_DATASET_PATH, 'dev', 'dev_databases')
            processed_data = []
            with pathlib.Path(raw_data_path).open('r') as f:
                data = json.load(f)
                for e in tqdm(data):
                    item = {
                        'instance_id': f'{len(processed_data)}',
                        'db_path': os.path.join(
                            database_path, e['db_id'], f"{e['db_id']}.sqlite"
                        ),
                        'db_id': e['db_id'],
                        'instruction': _create_prompt(e, database_path),
                        'SQL': e['SQL'],
                    }
                    processed_data.append(item)

            with pathlib.Path(processed_path).open('w') as f:
                json.dump(processed_data, f, indent=2)
                logger.info(f'Processed data saved to {processed_path}')
        else:
            logger.info(f'{processed_path} folder already exists.')
        bird_dataset = load_dataset('json', data_files={'test': processed_path})
        return bird_dataset

    raw_dataset_path = _download_bird()
    bird_dataset = _process_bird(raw_dataset_path)
    return bird_dataset


async def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info(f"{'-' * 50} BEGIN Runtime Initialization Fn {'-' * 50}")
    obs: CmdOutputObservation

    # Copy the database to the workspace
    db_file = os.path.join(
        LOCAL_DATASET_PATH,
        'dev',
        'dev_databases',
        instance.db_id,
        f'{instance.db_id}.sqlite',
    )
    await runtime.copy_to(db_file, '/workspace')

    # Check the database is copied
    action = CmdRunAction(
        command='cd /workspace && ls -l',
        keep_prompt=False,
    )
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    assert f'{instance.db_id}.sqlite' in obs.content

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
    timeout = 30

    test_result = {'result': {}, 'metadata': {}}

    # Read the generated python file
    instance_id = instance.instance_id.replace('/', '__')
    path = os.path.join('/workspace', f'{instance_id}.py')

    action = CmdRunAction(
        command=f'cat {path}',
        keep_prompt=False,
    )
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code != 0:
        test_result['result'] = {'passed': 0, 'status': 'error'}
        return test_result

    gen_file = obs.content.strip().replace('\r\n', '\n')

    # Extract the SQL from the python file
    gen_sql = ''
    pattern = r'sql\s*=\s*"([^"]+)"'
    match = re.search(pattern, gen_file)
    if match:
        gen_sql = match.group(1)
    else:
        print('No match found.')

    gold_sql = instance.SQL
    # Execute the SQL
    try:
        res = func_timeout(
            timeout,
            execute_sql,
            args=(
                instance.db_path,
                gen_sql,
                gold_sql,
            ),
        )
        status = 'success'
    except FunctionTimedOut:
        res = 0
        status = 'timeout'
    except Exception as e:
        res = 0
        status = 'error'
        logger.error(f'Error: {e}')

    # Save the test result
    test_result['result'] = {'passed': res, 'status': status}
    test_result['metadata'] = {
        'timeout': timeout,
        'gen_sql': gen_sql,
        'gold_sql': gold_sql,
    }
    logger.info(f"{'-' * 50} END Runtime Completion Fn {'-' * 50}")
    return test_result


async def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
) -> EvalOutput:
    config = get_config(metadata)
    # use session id for concurrent evaluation
    instance_id = instance.instance_id.replace('/', '__')

    # Set up the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance_id}.')

    # Create file with BIRD instance
    database_path = os.path.join('/workspace', f'{instance.db_id}.sqlite')
    statements = f"""
    import sqlite3
    def execute_sql(db_path, sql):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result

    if __name__ == '__main__':
        sql = "" # fill in your SQL here
        db_path = "{database_path}"
        print(db_path)
        result = execute_sql(db_path, sql)
        print(result)
    """

    instruction = (
        f'You are a SQL expert and need to complete the following text-to-SQL tasks.'
        f'\n\n{instance.instruction}\n\n'
        'Please write the SQL in one line without line breaks.'
        f'And write a new python file named {instance_id}.py to call the SQL you wrote.'
        'You need to follow the code template below:'
        f'\n\n{statements}\n\n'
        'Environment has been set up for you to start working.'
        'You may assume all necessary tools are installed.\n\n'
    )
    instruction += (
        'IMPORTANT: You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.\n'
        'You SHOULD INCLUDE PROPER INDENTATION in your edit commands.\n'
    )
    # NOTE: You can actually set slightly different instruction for different agents
    instruction += AGENT_CLS_TO_INST_SUFFIX[metadata.agent_class]

    runtime = await create_runtime(config, sid=instance_id)
    await initialize_runtime(runtime, instance)

    # Here's how you can run the agent (similar to the `main` function) and get the final task state
    state: State | None = await run_controller(
        config=config,
        task_str=instruction,
        fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[metadata.agent_class],
        runtime=runtime,
    )

    # ======= Attempt to evaluate the agent's edits =======
    test_result = await complete_runtime(runtime, instance)

    # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
    # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
    if state is None:
        raise ValueError('State should not be None.')
    metrics = state.metrics.get() if state.metrics else None

    # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
    # for compatibility with the existing output format, we can remake the pairs here
    # remove when it becomes unnecessary
    histories = state.history.compatibility_for_eval_history_pairs()

    # Save the output
    output = EvalOutput(
        instance_id=instance.instance_id,
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
    bird_dataset = load_bird()
    dataset = bird_dataset['test'].to_pandas()
    dataset.rename(columns={'task_id': 'instance_id'}, inplace=True)

    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config)
    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    metadata = make_metadata(
        llm_config,
        'BIRD',
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    asyncio.run(
        run_evaluation(
            instances, metadata, output_file, args.eval_num_workers, process_instance
        )
    )
