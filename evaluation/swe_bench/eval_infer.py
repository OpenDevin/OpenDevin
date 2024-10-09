import os
import tempfile
import time

import pandas as pd
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)
from swebench.harness.test_spec import SWEbenchInstance, TestSpec, make_test_spec
from swebench.harness.utils import load_swebench_dataset

from evaluation.swe_bench.run_infer import get_instance_docker_image
from evaluation.utils.shared import (
    EvalMetadata,
    EvalOutput,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.core.config import (
    AppConfig,
    LLMConfig,
    SandboxConfig,
    get_parser,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation

# TODO: migrate all swe-bench docker to ghcr.io/openhands
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')


def process_git_patch(patch):
    if not isinstance(patch, str):
        return ''

    if not patch.strip():
        # skip empty patches
        return ''

    patch = patch.replace('\r\n', '\n')
    # There might be some weird characters at the beginning of the patch
    # due to some OpenHands inference command outputs

    # FOR EXAMPLE:
    # git diff --no-color --cached 895f28f9cbed817c00ab68770433170d83132d90
    # [A[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[C[K0
    # diff --git a/django/db/models/sql/.backup.query.py b/django/db/models/sql/.backup.query.py
    # new file mode 100644
    # index 0000000000..fc13db5948

    # We "find" the first line that starts with "diff" and then we remove lines before it
    lines = patch.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            patch = '\n'.join(lines[i:])
            break

    patch = patch.rstrip() + '\n'  # Make sure the last line ends with a newline
    return patch


def get_config(instance: pd.Series) -> AppConfig:
    # We use a different instance image for the each instance of swe-bench eval
    base_container_image = get_instance_docker_image(instance['instance_id'])
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )
    default_llm = LLMConfig(
        draft_editor=LLMConfig()  # just to prevent eval errors
    )
    config = AppConfig(
        run_as_openhands=False,
        runtime=os.environ.get('RUNTIME', 'eventstream'),
        sandbox=SandboxConfig(
            base_container_image=base_container_image,
            use_host_network=False,
            # large enough timeout, since some testcases take very long to run
            timeout=1800,
            api_key=os.environ.get('ALLHANDS_API_KEY', None),
            remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        ),
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(default_llm)
    return config


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata | None = None,
    reset_logger: bool = True,
) -> EvalOutput:
    # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
    if reset_logger:
        global output_file
        log_dir = output_file.replace('.jsonl', '.logs')
        os.makedirs(log_dir, exist_ok=True)
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    config = get_config(instance)
    instance_id = instance.instance_id
    model_patch = instance['model_patch']
    test_spec: TestSpec = instance['test_spec']
    logger.info(f'Starting evaluation for instance {instance_id}.')

    if 'test_result' not in instance.keys():
        instance['test_result'] = {}
    instance['test_result']['report'] = {
        'empty_generation': False,
        'resolved': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
    }

    if model_patch == '':
        instance['test_result']['report']['empty_generation'] = True
        return EvalOutput(
            instance_id=instance_id,
            test_result=instance['test_result'],
        )

    runtime = create_runtime(config, sid=instance_id)

    # Get patch and save it to /tmp/patch.diff
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch file
        patch_file_path = os.path.join(temp_dir, 'patch.diff')
        with open(patch_file_path, 'w') as f:
            f.write(model_patch)
        runtime.copy_to(patch_file_path, '/tmp')
        # Eval script
        eval_script_path = os.path.join(temp_dir, 'eval.sh')
        with open(eval_script_path, 'w') as f:
            f.write(test_spec.eval_script)
        runtime.copy_to(eval_script_path, '/tmp')

    # Set +x
    action = CmdRunAction(command='chmod +x /tmp/eval.sh')
    action.timeout = 600
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    # Apply patch
    exec_command = (
        'cd /testbed && '
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    action = CmdRunAction(command=exec_command, keep_prompt=False)
    action.timeout = 600
    obs = runtime.run_action(action)
    assert isinstance(obs, CmdOutputObservation)
    apply_patch_output = obs.content
    assert isinstance(apply_patch_output, str)
    instance['test_result']['apply_patch_output'] = apply_patch_output

    try:
        if 'APPLY_PATCH_FAIL' in apply_patch_output:
            logger.info(f'[{instance_id}] {APPLY_PATCH_FAIL}:\n{apply_patch_output}')
            instance['test_result']['report']['failed_apply_patch'] = True

            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
            )
        elif 'APPLY_PATCH_PASS' in apply_patch_output:
            logger.info(f'[{instance_id}] {APPLY_PATCH_PASS}:\n{apply_patch_output}')

            # Run eval script in background and save output to log file
            log_file = '/tmp/eval_output.log'
            action = CmdRunAction(
                command=f'/tmp/eval.sh > {log_file} 2>&1 & echo $!', keep_prompt=False
            )
            action.timeout = 60  # Short timeout just to get the process ID
            obs = runtime.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                pid = obs.content.split()[-1].strip()
                logger.info(
                    f'[{instance_id}] Evaluation process started with PID: {pid}'
                )

                # Poll for completion
                start_time = time.time()
                timeout = 1800  # 30 minutes
                while True:
                    seconds_elapsed = time.time() - start_time
                    if seconds_elapsed > timeout:
                        logger.info(
                            f'[{instance_id}] Evaluation timed out after {timeout} seconds'
                        )
                        instance['test_result']['report']['test_timeout'] = True
                        break
                    check_action = CmdRunAction(
                        command=f'ps -p {pid} > /dev/null; echo $?', keep_prompt=False
                    )
                    check_action.timeout = 60
                    check_obs = runtime.run_action(check_action)
                    if (
                        isinstance(check_obs, CmdOutputObservation)
                        and check_obs.content.split()[-1].strip() == '1'
                    ):
                        logger.info(
                            f'[{instance_id}] Evaluation process completed after {seconds_elapsed} seconds'
                        )
                        break
                    logger.info(
                        f'[{instance_id}] [{seconds_elapsed:.0f}s] Evaluation still running, waiting...'
                    )
                    time.sleep(30)  # Wait for 30 seconds before checking again

                # Read the log file
                cat_action = CmdRunAction(command=f'cat {log_file}', keep_prompt=False)
                cat_action.timeout = 300
                cat_obs = runtime.run_action(cat_action)

                # Grade answer
                if isinstance(cat_obs, CmdOutputObservation) and cat_obs.exit_code == 0:
                    test_output = cat_obs.content
                    assert isinstance(test_output, str)
                    instance['test_result']['test_output'] = test_output

                    # Get report from test output
                    logger.info(f'[{instance_id}] Grading answer...')
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a directory structure that matches the expected format
                        # NOTE: this is a hack to make the eval report format consistent
                        # with the original SWE-Bench eval script
                        log_dir = os.path.join(temp_dir, 'logs', instance_id)
                        os.makedirs(log_dir, exist_ok=True)
                        test_output_path = os.path.join(log_dir, 'test_output.txt')
                        with open(test_output_path, 'w') as f:
                            f.write(test_output)

                        _report = get_eval_report(
                            test_spec=test_spec,
                            prediction={
                                'model_patch': model_patch,
                                'instance_id': instance_id,
                            },
                            log_path=test_output_path,
                            include_tests_status=True,
                        )
                        report = _report[instance_id]
                        logger.info(
                            f"[{instance_id}] report: {report}\nResult for {instance_id}: resolved: {report['resolved']}"
                        )
                        instance['test_result']['report']['resolved'] = report[
                            'resolved'
                        ]
            else:
                logger.info(f'[{instance_id}] Error when starting eval:\n{obs.content}')
                instance['test_result']['report']['error_eval'] = True

            return EvalOutput(
                instance_id=instance_id,
                test_result=instance['test_result'],
            )
        else:
            logger.info(
                f'[{instance_id}] Unexpected output when applying patch:\n{apply_patch_output}'
            )
            raise RuntimeError(
                instance_id,
                f'Unexpected output when applying patch:\n{apply_patch_output}',
                logger,
            )
    finally:
        runtime.close()


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input predictions file',
        required=True,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    args, _ = parser.parse_known_args()

    # Load SWE-Bench dataset
    full_dataset: list[SWEbenchInstance] = load_swebench_dataset(
        args.dataset, args.split
    )
    instance_id_to_instance = {
        instance['instance_id']: instance for instance in full_dataset
    }
    logger.info(
        f'Loaded dataset {args.dataset} with split {args.split} to run inference on.'
    )

    # Load predictions
    assert args.input_file.endswith('.jsonl'), 'Input file must be a jsonl file.'
    predictions = pd.read_json(args.input_file, lines=True)
    assert (
        'instance_id' in predictions.columns
    ), 'Input file must contain instance_id column.'

    if 'model_patch' not in predictions.columns and (
        'test_result' in predictions.columns
        and 'model_patch' in predictions['test_result'].iloc[0]
    ):
        raise ValueError(
            'Input file must contain model_patch column OR test_result column with model_patch field.'
        )
    assert len(predictions['instance_id'].unique()) == len(
        predictions
    ), 'instance_id column must be unique.'

    if 'model_patch' not in predictions.columns:
        predictions['model_patch'] = predictions['test_result'].apply(
            lambda x: x['git_patch']
        )
    assert {'instance_id', 'model_patch'}.issubset(
        set(predictions.columns)
    ), 'Input file must contain instance_id and model_patch columns.'

    # Process model_patch
    predictions['model_patch'] = predictions['model_patch'].apply(process_git_patch)

    # Merge predictions with dataset
    predictions['instance'] = predictions['instance_id'].apply(
        lambda x: instance_id_to_instance[x]
    )
    predictions['test_spec'] = predictions['instance'].apply(make_test_spec)

    # Prepare dataset
    output_file = args.input_file.replace('.jsonl', '.swebench_eval.jsonl')
    instances = prepare_dataset(predictions, output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata=None,
        output_file=output_file,
        num_workers=args.eval_num_workers,
        process_instance_func=process_instance,
    )

    # Load evaluated predictions & print number of resolved predictions
    evaluated_predictions = pd.read_json(output_file, lines=True)
    fields = ['resolved', 'failed_apply_patch', 'error_eval', 'empty_generation']

    def count_report_field(row, field):
        return row['test_result']['report'][field]

    for field in fields:
        count = evaluated_predictions.apply(
            count_report_field, args=(field,), axis=1
        ).sum()
        logger.info(
            f'# {field}: {count} / {len(evaluated_predictions)}. ({count / len(evaluated_predictions):.2%})'
        )
