import json
import os
import pprint

import tqdm

from opendevin.core.config import get_llm_config_arg, get_parser, load_app_config
from opendevin.core.logger import opendevin_logger as logger
from opendevin.llm.llm import LLM

config = load_app_config()


def extract_test_results(res_file_path: str) -> tuple[list[str], list[str]]:
    passed = []
    failed = []
    costs = []
    instance_ids = set()
    instances = []
    with open(res_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            success = data['metrics']['success']
            if data['instance_id'] in instance_ids:
                print(f'WARNING: Duplicate instance_id found: {data["instance_id"]}')
                continue
            instance_ids.add(data['instance_id'])
            instances.append(data)
            if success:
                passed.append(
                    {
                        'instance_id': data['instance_id'],
                        'repo': data['repo'],
                        'instruction': data['instruction'],
                        'eval_script': data['eval_script'],
                        'eval_exit_code': data['eval_exit_code'],
                        'eval_output': data['eval_output'],
                        'accumulated_cost': data['metrics']['accumulated_cost'],
                    }
                )
            else:
                failed.append(
                    {
                        'instance_id': data['instance_id'],
                        'repo': data['repo'],
                        'instruction': data['instruction'],
                        'metadata': data['metadata'],
                        'history': data['history'],
                        'eval_script': data['eval_script'],
                        'eval_exit_code': data['eval_exit_code'],
                        'eval_output': data['eval_output'],
                        'accumulated_cost': data['metrics']['accumulated_cost'],
                    }
                )
            costs.append(data['metrics']['accumulated_cost'])

        # sort by instance_id
        instances.sort(key=lambda x: x['instance_id'])
        with open(res_file_path, 'w') as file:
            for instance in instances:
                file.write(json.dumps(instance) + '\n')
        return passed, failed, costs


async def classify_error(llm: LLM, failed_case: dict) -> str:
    prompt = f"""
    Please classify the error for the following failed case based on the history and eval_output:

    Instruction:
    {failed_case['instruction']}

    Eval Script:
    {failed_case['eval_script']}s

    History:
    {failed_case['history']}

    Eval Output:
    {failed_case['eval_output']}

    The error categories are:
    E1: Hallucination Errors - The model misinterpreted the user's intention, misplaced Python code and bash script, or generated random or irrelevant code.
    E2: Lack of Knowledge or Information - The model lacks sufficient information or domain-specific knowledge to satisfy the user's requirements.
    E3: Knowledge Manipulation - The model failed to integrate or manipulate information properly.
    E4: Syntax Errors - The model generated code with syntax errors.
    E5: Operational Error - The model gave up easily or exited without finishing the tasks.

    Please provide only the error category (E1, E2, E3, E4, or E5) without any explanation.
    """

    try:
        response = await llm.completion(messages=[{'content': prompt, 'role': 'user'}])
        error_category = response.choices[0].message['content']
    except Exception as e:
        logger.error(
            f"Failed to classify the error for the failed case: {failed_case['instance_id']}"
        )
        logger.error(e)
        error_category = input(
            failed_case['instruction']
            + ': '
            + failed_case['eval_script']
            + ' - '
            + failed_case['eval_output']
        )

    if error_category not in ['E1', 'E2', 'E3', 'E4', 'E5']:
        raise ValueError(f'Invalid error category: {error_category}')

    return error_category


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--json_file_path',
        type=str,
        required=True,
        help='Path to the jsonl file containing the evaluation results',
    )
    args, _ = parser.parse_known_args()

    # Check https://github.com/OpenDevin/OpenDevin/blob/main/evaluation/swe_bench/README.md#configure-opendevin-and-your-llm
    # for details of how to set `llm_config`
    if args.llm_config:
        specified_llm_config = get_llm_config_arg(args.llm_config)
        if specified_llm_config:
            config.llm = specified_llm_config
    logger.info(f'Config for evaluation: {config}')
    llm = LLM(llm_config=specified_llm_config)

    passed, new_failed, costs = extract_test_results(args.json_file_path)

    failed = []
    if os.path.exists(args.json_file_path.replace('.jsonl', '_failed.jsonl')):
        with open(args.json_file_path.replace('.jsonl', '_failed.jsonl'), 'r') as file:
            for line in file:
                failed.append(json.loads(line.strip()))
        print(
            f'Loaded {len(failed)} failed cases from {args.json_file_path.replace(".jsonl", "_failed.jsonl")}'
        )

    for failed_case in tqdm.tqdm(new_failed):
        if failed_case['instance_id'] in [case['instance_id'] for case in failed]:
            continue
        error_category = classify_error(llm, failed_case)
        failed_case['error_category'] = error_category
        failed.append(failed_case)
        with open(args.json_file_path.replace('.jsonl', '_failed.jsonl'), 'a') as file:
            file.write(json.dumps(failed_case) + '\n')

    # Print the summary
    print('Summary:')
    print(f'Passed: {len(passed)}')
    print(f'Failed: {len(failed)}')
    print(f'Costs: {costs}')
    print('Failed cases:')
    error_categories = {}
    for case in failed:
        error_category = case['error_category']
        if error_category not in error_categories:
            error_categories[error_category] = 0
        error_categories[error_category] += 1
    pprint.pprint(error_categories)
