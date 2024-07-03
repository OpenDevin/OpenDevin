"""
Overview:
This code implements the evaluation of agents on the GPQA Benchmark with Open Book setting.
- The benchmark consists of 448 high-quality and extremely difficult multiple-choice questions in the domains of biology, physics, and chemistry. The questions are intentionally designed to be "Google-proof," meaning that even highly skilled non-expert validators achieve only 34% accuracy despite unrestricted access to the web.
- Even experts in the corresponding domains achieve only 65% accuracy.
- State-of-the-art AI systems achieve only 39% accuracy on this challenging dataset.

Accurate solving of above graduate level questions would require both tool use (e.g., python for calculations) and web-search for finding related facts as information required for the questions might not be part of the LLM knowledge / training data.

Further references:
- https://arxiv.org/pdf/2311.12022
- https://paperswithcode.com/dataset/gpqa
- https://github.com/idavidrein/gpqa

TODOs:
- Add evaluation on other Agent classes (e.g., MonologueAgent)
- Batch inference and evaluation of agents on the GPQA Benchmark.
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import pathlib
import random
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from opendevin.controller.state.state import State
from opendevin.core.config import config, get_llm_config_arg, get_parser
from opendevin.core.logger import get_console_handler
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.main import main
from opendevin.events.action import MessageAction


def cleanup():
    logger.info('Cleaning up child processes...')
    for process in mp.active_children():
        logger.info(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def codeact_user_response(state: State) -> str:
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'Feel free to use all tools for calculations and solving the problem, and web-search for finding relevant facts during the process if needed\n'
        'If you think you have reliably finished solving the problem, first generate a message reporting the final concise answer to the user. Once that is done, please run the following command: <execute_bash> exit </execute_bash>.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP TO SOLVE THIS TASK.\n'
    )

    # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
    if state.history:
        user_msgs = [
            event
            for event in state.history.get_events()
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) > 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, just generate a final answer message to the user and in the next turn --> run: <execute_bash> exit </execute_bash>.\n'
            )
    return msg


def monologue_user_response(state: State) -> str:
    raise NotImplementedError('MonologueAgent should never ask for user responses.')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
    'MonologueAgent': monologue_user_response,
}

AGENT_CLS_TO_INST_SUFFIX = {
    'CodeActAgent': '\n\n SUPER IMPORTANT: When you think you have solved the question, first report it back to the user in the requested format. Only once that is done, in the next turn, please run the following command: <execute_bash> exit </execute_bash>.\n'
}


def parse_final_answer(final_answer: str) -> str:
    """
    Parse the final answer from the final message generated by the agent
    to extract the final answer. The final answer is usually enclosed in the format:
    <<FINAL_ANSWER||
    <insert correct answer here>
    ||FINAL_ANSWER>>
    """
    pattern = re.compile(r'<<FINAL_ANSWER\|\|(.*?)\|\|FINAL_ANSWER>>', re.DOTALL)
    match = pattern.search(final_answer)

    if match:
        return match.group(1).strip()
    else:
        return 'No final answer found in the provided string.'


def compare_answers(predicted_answer, ground_truth):
    """
    Compare the predicted answer with the ground truth answer
    """
    return predicted_answer == ground_truth


def get_test_result(model_output, ground_truth):
    """
    Implements the evaluation logic for GPQA
    Checks if the output of a given instance is correct (as per the ground truth)
    """
    # parse the final answer from model output
    predicted_answer = parse_final_answer(model_output)

    # check if the model output matches the ground truth
    result = compare_answers(predicted_answer, ground_truth)

    return result


def convert_instance_dict(instance):
    """
    Used for preprocessing the hf dataset into a format that can be used by the agent.
    Reads and extracts relevant information from the dataset instance.
    """
    out_instance_dict = {}
    out_instance_dict['question'] = instance['Question']
    correct_answer = instance['Correct Answer']
    out_instance_dict['choices'] = [
        correct_answer,
        instance['Incorrect Answer 1'],
        instance['Incorrect Answer 2'],
        instance['Incorrect Answer 3'],
    ]

    # Randomize the order of choices
    random.shuffle(out_instance_dict['choices'])

    # Find the index of the correct answer after shuffling and store it as a letter (A/B/C/D)
    correct_index = out_instance_dict['choices'].index(correct_answer)
    correct_letter = chr(
        65 + correct_index
    )  # Convert index (0-3) to corresponding letter (A-D)

    out_instance_dict['correct_solution'] = correct_letter

    return out_instance_dict


def process_instance(
    instance: dict,
    agent_class: str,
    metadata: dict,
    skip_workspace_mount: bool,
    eval_output_dir: str,
    reset_logger: bool = True,
):
    """
    Process a single instance from the dataset
    """
    old_workspace_mount_path = config.workspace_mount_path
    old_workspace_base = config.workspace_base
    try:
        workspace_mount_path = os.path.join(
            config.workspace_mount_path, '_eval_workspace'
        )
        # create process-specific workspace dir
        # if `not skip_workspace_mount` - we will create a workspace directory for EACH process
        # so that different agent don't interfere with each other.
        skip_workspace_mount = False
        if not skip_workspace_mount:
            workspace_mount_path = os.path.join(workspace_mount_path, str(os.getpid()))
            pathlib.Path(workspace_mount_path).mkdir(parents=True, exist_ok=True)

        # reset workspace to config
        config.workspace_base = workspace_mount_path
        config.workspace_mount_path = workspace_mount_path

        # workspace_mount_path = os.path.join(config.workspace_mount_path, '_eval_workspace')
        # workspace_mount_path = os.path.abspath(workspace_mount_path)
        # # create process-specific workspace dir
        # # if `not skip_workspace_mount` - we will create a workspace directory for EACH process
        # # so that different agent don't interfere with each other.
        # if not skip_workspace_mount:
        #     workspace_mount_path = os.path.join(workspace_mount_path, str(os.getpid()))
        #     pathlib.Path(workspace_mount_path).mkdir(parents=True, exist_ok=True)

        # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
        if reset_logger:
            # Set up logger
            log_file = os.path.join(
                eval_output_dir, 'logs', f'instance_{instance.instance_id}.log'
            )
            # Remove all existing handlers from logger
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # add back the console handler to print ONE line
            logger.addHandler(get_console_handler())
            logger.info(
                f'Starting evaluation for instance {instance.instance_id}.\nHint: run "tail -f {log_file}" to see live logs in a separate shell'
            )
            # Remove all existing handlers from logger
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)
        else:
            logger.info(f'Starting evaluation for instance {instance.instance_id}.')

        if not skip_workspace_mount:
            logger.info(f'Process-specific workspace mounted at {workspace_mount_path}')

        # ======= Run the agent on the instance =======
        # Prepare instruction for the agent using suggested format in gpqa codebase
        instruction = f"""
        What is the correct answer to this question:\n
        {instance['question']}\n

        Choices:\n
        (A) {instance['choices'][0]}\n
        (B) {instance['choices'][1]}\n
        (C) {instance['choices'][2]}\n
        (D) {instance['choices'][3]}\n
        \n\n

        MOST IMPORTANT: Format your response as follows:
        <<FINAL_ANSWER||
        <insert correct answer here, must be one of A, B, C, D> (Please dont use any additional characters. Just the letter of the correct answer (A/B/C/D).)
        ||FINAL_ANSWER>>

        Additional Instructions:
        - You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
        """

        # NOTE: You can actually set slightly different instruction for different agents
        instruction += AGENT_CLS_TO_INST_SUFFIX.get(agent_class, '')

        # Here's how you can run the agent (similar to the `main` function) and get the final task state
        state: State = asyncio.run(
            main(
                instruction,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN.get(
                    agent_class
                ),
                sid=instance.instance_id,
            )
        )

        # ======= Attempt to evaluate the agent's edits =======
        # get the final message from the state history (default to None if not found)
        final_message = next(
            (
                act.content
                for act in state.history.get_events(reverse=True)
                if isinstance(act, MessageAction)
            ),
            None,
        )

        logger.info(f'Final message generated by the agent: {final_message}')

        test_result = get_test_result(final_message, instance.correct_solution)

        # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
        # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
        if state is None:
            raise ValueError('State should not be None.')

        metrics = state.metrics.get() if state.metrics else None

        # history is now available as a stream of events, rather than list of pairs of (Action, Observation)
        # for compatibility with the existing output format, we can remake the pairs here
        # remove when it becomes unnecessary
        histories = state.history.compatibility_for_eval_history_tuples()

        # Save the output
        output = {
            'task_id': instance.task_id,
            'instance_id': instance.instance_id,
            'instruction': instruction,
            'metadata': metadata,
            'history': histories,
            'metrics': metrics,
            'error': state.last_error if state and state.last_error else None,
            'test_result': test_result,
        }

    except Exception:
        logger.error('Process instance failed')
        raise
    finally:
        config.workspace_mount_path = old_workspace_mount_path
        config.workspace_base = old_workspace_base
    return output


if __name__ == '__main__':
    parser = get_parser()
    # data split must be one of 'gpqa_main', 'gqpa_diamond', 'gpqa_experts', 'gpqa_extended'
    parser.add_argument(
        '--data-split',
        type=str,
        choices=['gpqa_main', 'gpqa_diamond', 'gpqa_experts', 'gpqa_extended'],
        default='gpqa_diamond',
        help='data split to evaluate, eg. gpqa_diamond',
    )
    args, _ = parser.parse_known_args()

    # NOTE: It is preferable to load datasets from huggingface datasets and perform post-processing
    # so we don't need to manage file uploading to OpenDevin's repo
    dataset = load_dataset('Idavidrein/gpqa', args.data_split)
    gpqa_dataset = dataset['train']
    # preprocess the dataset
    gpqa_dataset = gpqa_dataset.map(convert_instance_dict)
    gpqa_dataset = gpqa_dataset.to_pandas()
    # Add a new column 'instance_id' with the index
    gpqa_dataset['instance_id'] = gpqa_dataset.index
    gpqa_dataset['task_id'] = gpqa_dataset.index
    # gpqa_dataset = dataset['train'].to_pandas().sort_values(by='id').reset_index(drop=True)

    # Check https://github.com/OpenDevin/OpenDevin/blob/main/evaluation/swe_bench/README.md#configure-opendevin-and-your-llm
    # for details of how to set `llm_config`
    if args.llm_config:
        specified_llm_config = get_llm_config_arg(args.llm_config)
        if specified_llm_config:
            config.llm = specified_llm_config
    logger.info(f'Config for evaluation: {config}')

    # TEST METADATA
    agent_class = args.agent_cls
    assert (
        agent_class in AGENT_CLS_TO_FAKE_USER_RESPONSE_FN
    ), f'Unsupported agent class: {agent_class}'
    model_name = config.llm.model.split('/')[-1]
    max_iterations = args.max_iterations
    eval_note = ''
    if args.eval_note is not None:
        eval_note += '_N_' + args.eval_note
    eval_output_dir = os.path.join(
        args.eval_output_dir,
        'gpqa',
        agent_class,
        model_name + '_maxiter_' + str(max_iterations) + eval_note,
    )

    pathlib.Path(eval_output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(eval_output_dir, 'logs')).mkdir(
        parents=True, exist_ok=True
    )
    logger.info(f'Using evaluation output directory: {eval_output_dir}')

    metadata = {
        'agent_class': agent_class,
        'model_name': model_name,
        'max_iterations': max_iterations,
        'eval_output_dir': eval_output_dir,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        # get the commit id of current repo for reproduciblity
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip(),
    }
    logger.info(f'Metadata: {metadata}')
    with open(os.path.join(eval_output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # LIMIT EVALUATION
    eval_n_limit = args.eval_n_limit  # NOTE: This is useful for debugging and testing using a smaller subset of the dataset
    if eval_n_limit:
        # start_index = 20
        # gpqa_dataset = gpqa_dataset.iloc[start_index:]
        gpqa_dataset = gpqa_dataset.head(eval_n_limit)
        logger.info(f'Limiting evaluation to first {eval_n_limit} instances.')

    logger.info('#############################################')
    logger.info(f'{eval_n_limit} instances will be evaluated.')
    logger.info('#############################################')

    # OUTPUT FILE
    output_file = os.path.join(eval_output_dir, 'output.jsonl')
    logger.info(f'Writing evaluation output to {output_file}')
    finished_instance_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                finished_instance_ids.add(data['instance_id'])
        logger.warning(
            f'Output file {output_file} already exists. Loaded {len(finished_instance_ids)} finished instances.'
        )
    output_fp = open(output_file, 'a')

    logger.info(
        f'Evaluation started with Agent {agent_class}, model {model_name}, max iterations {max_iterations}.'
    )

    # =============================================
    # filter out finished instances
    new_gpqa_dataset = []
    for idx, instance in gpqa_dataset.iterrows():
        # instance = convert_instance_dict(instance) # preprocessing
        if instance.instance_id in finished_instance_ids:
            logger.info(
                f'Skipping instance {instance.instance_id} as it is already finished.'
            )
            continue
        new_gpqa_dataset.append(instance)

    gpqa_dataset = pd.DataFrame(new_gpqa_dataset)
    logger.info(
        f'Finished instances: {len(finished_instance_ids)}, Remaining instances: {len(gpqa_dataset)}'
    )
    # =============================================

    pbar = tqdm(total=len(gpqa_dataset))

    # This function tracks the progress AND write the output to a JSONL file
    def update_progress(future):
        pbar.update(1)
        output = future.result()
        pbar.set_description(f'Instance {output["instance_id"]}')
        pbar.set_postfix_str(f'Test Result: {output["test_result"]["result"]}')
        logger.info(
            f'Finished evaluation for instance {output["instance_id"]}: {output["test_result"]["result"]}'
        )
        output_fp.write(json.dumps(output) + '\n')
        output_fp.flush()

    # This sets the multi-processing
    num_workers = args.eval_num_workers
    logger.info(f'Using {num_workers} workers for evaluation.')

    # This is SWE-Bench specific - CodeActAgent doesn't require mounted workspace to work
    skip_workspace_mount = agent_class == 'CodeActAgent'
    logger.info(f'Skipping workspace mount: {skip_workspace_mount}')

    try:
        with ProcessPoolExecutor(num_workers) as executor:
            futures = []
            # This is how we perform multi-processing
            for row_idx, instance in gpqa_dataset.iterrows():
                future = executor.submit(
                    process_instance,
                    instance,
                    agent_class,
                    metadata,
                    skip_workspace_mount,
                    eval_output_dir,
                    reset_logger=bool(num_workers > 1),
                )
                future.add_done_callback(update_progress)
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        print('KeyboardInterrupt received. Cleaning up...')
        cleanup()

    output_fp.close()
    logger.info('Evaluation finished.')
