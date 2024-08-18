import os
import pickle
import subprocess
from typing import Any, Dict

from ..utils.config import _get_max_token, _get_openai_client, _get_openai_model


def apply_git_patch(patch_str, directory='/workspace'):
    """
    Apply a git patch from a string to a designated folder within the given directory.

    Parameters:
    - patch_str (str): The git patch as a string.
    - directory (str): The directory containing the subdirectory where the patch should be applied.

    Returns:
    - bool: True if the patch was successfully applied, False otherwise.
    """
    print("yay, I'm applying a patch!!!!")
    try:
        # Ensure the target directory exists
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'The directory {directory} does not exist.')

        # Find the subdirectory containing "__" in its name
        subdirectory_path = ''
        for d in os.listdir(directory):
            full_path = os.path.join(directory, d)
            if os.path.isdir(full_path) and '__' in d:
                subdirectory_path = full_path
                break  # Assuming there's only one such subdirectory

        print(subdirectory_path)
        # Change the current working directory to the subdirectory
        os.chdir(subdirectory_path)

        # Apply the patch
        result = subprocess.run(
            ['git', 'apply', '-'],
            input=patch_str.replace('<newline_character>', '\\n'),
            text=True,
            capture_output=True,
        )

        # Check if the patch was applied successfully
        if result.returncode != 0:
            print(f'Error applying patch: {result.stderr}')
            return False

        print('Patch applied successfully.')
        return True

    except Exception as e:
        print(f'An error occurred: {e}')
        return False


MAX_CONTEXT_LENGTH = 128000
TOP_N = 3
CONTEXT_WINDOW = 10
ADD_SPACE = False
STICKY_SCROLL = False
NO_LINE_NUMBER = False
TEMPERATURE = 0.8
LOC_INTERVAL = True
FINE_GRAIN_LOC_ONLY = False
COT = True
DIFF_FORMAT = True
STOP_AT_N_UNIQUE_VALID_SAMPLES = -1
SKIP_GREEDY = False

AGENTLESS_FOUND_FILE_OBSERVATION = '<--- One or more of the following files likely need to be edited to solve the issue:'
AGENTLESS_FOUND_FILE_PICKLE = '/workspace/agentless_file_localization.pkl'

AGENTLESS_RELATED_LOCATIONS_OBSERVATION = '<--- One or more of the following locations likely need to be edited to solve the issue:'
AGENTLESS_RELATED_LOCATIONS_PICKLE = '/workspace/agentless_related_localization.pkl'

AGENTLESS_LINE_LOCATIONS_OBSERVATION = '<--- One or more of the following line locations likely need to be edited to solve the issue:'
AGENTLESS_LINE_LOCATIONS_PICKLE = '/workspace/agentless_line_localization.pkl'

AGENTLESS_REPAIR_OBSERVATION = (
    '<--- The following are proposed git patches to solve the issue:'
)
AGENTLESS_MULTI_LINE_LOCATION_PICKLE = '/workspace/agentless_multi_repair.pkl'
AGENTLESS_REPAIR_PICKLE = '/workspace/agentless_repair.pkl'

AGENTLESS_FINAL_PATCH_OBSERVATION = (
    '<--- The following patch is proposed by AgentlessAgent:'
)
AGENTLESS_FINAL_PATCH_PICKLE = '/workspace/agentless_final_patch.pkl'

OBSERVATION_ENDING = '|--->'


def escape_quotes(input_string):
    # Replace single quotes with escaped single quotes
    escaped_string = input_string.replace("'", "\\'")
    # Replace double quotes with escaped double quotes
    escaped_string = escaped_string.replace('"', '\\"')
    return escaped_string


def unescape_quotes(input_string):
    # Replace escaped single quotes with single quotes
    unescaped_string = input_string.replace("\\'", "'")
    # Replace escaped double quotes with double quotes
    unescaped_string = unescaped_string.replace('\\"', '"')
    return unescaped_string


def convert_list_of_lists_to_str(lists, level=0, separators=None):
    """Helper function to convert lists to strings for Agentless"""
    if separators is None:
        separators = [
            '\n<->\n',
            '\n<--->\n',
            '\n<----->\n',
            '\n<------->\n',
        ]  # Define more if needed
        separators.reverse()
    if level >= len(separators):
        raise ValueError('Exceeded the number of supported levels.')

    if not isinstance(lists, list):
        return str(lists)

    joined_inner_lists = [
        convert_list_of_lists_to_str(inner_list, level + 1, separators)
        for inner_list in lists
    ]
    final_string = separators[level].join(joined_inner_lists)

    return final_string


def convert_str_to_list_of_lists(s, level=0, separators=None):
    """Helper function to convert strings to lists for Agentless"""
    if separators is None:
        separators = [
            '\n<->\n',
            '\n<--->\n',
            '\n<----->\n',
            '\n<------->\n',
        ]  # Define more if needed
        separators.reverse()

    if level >= len(separators):
        raise ValueError('Exceeded the number of supported levels.')

    if separators[level] not in s:
        return s

    lists = s.split(separators[level])
    list_of_lists = [
        convert_str_to_list_of_lists(inner_str, level + 1, separators)
        for inner_str in lists
    ]

    return list_of_lists


def _parse_model_return_lines(content: str) -> list[str]:
    """Parsing utility function"""
    if content:
        return content.strip().split('\n')
    return ['']


def install_agentless_dependencies():
    """Install Agentless dependencies"""
    import subprocess
    import sys

    def install_packages():
        subprocess.check_call(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '-r',
                '/workspace/Agentless/requirements.txt',
            ]
        )

    def clone_and_setup_workspace():
        repo_url = 'https://github.com/OpenAutoCoder/Agentless.git'
        workspace_dir = '/workspace'

        subprocess.run(
            ['git', 'clone', repo_url, workspace_dir + '/Agentless'], check=True
        )

    clone_and_setup_workspace()
    install_packages()
    return 'Agentless dependencies installed'


def install_agentless_dependencies_dummy():
    """Install Agentless dependencies"""
    import subprocess
    import sys

    def install_packages():
        subprocess.check_call(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '-r',
                '/workspace/Agentless/requirements.txt',
            ]
        )

    def clone_and_setup_workspace():
        repo_url = 'https://github.com/OpenAutoCoder/Agentless.git'
        workspace_dir = '/workspace'

        subprocess.run(
            ['git', 'clone', repo_url, workspace_dir + '/Agentless'], check=True
        )

    # clone_and_setup_workspace()
    # install_packages()
    return 'Agentless dependencies installed'


def retrieve_structure(base_path='/workspace'):
    """Create the structure of the repository directory by parsing Python files.

    If a pickled structure exists, load it. Otherwise, create it and pickle the result.

    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """

    import os
    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    from Agentless.agentless.util.preprocess_data import (
        filter_none_python,
        filter_out_test_files,
    )
    from Agentless.get_repo_structure.get_repo_structure import (
        parse_python_file,
    )

    def find_directory_with_double_underscore(base_path):
        for root, dirs, files in os.walk(base_path):
            for directory in dirs:
                if '__' in directory:
                    return os.path.abspath(os.path.join(root, directory))
        return None

    # Call the function and print the result
    directory_path = find_directory_with_double_underscore(base_path)

    pickle_file = os.path.join(base_path, 'repo_structure.pkl')

    # Try to load the pickle file if it exists
    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as pf_in:
            return pickle.load(pf_in)

    # Proceed to create the structure if pickle file doesn't exist
    structure: Dict[str, Any] = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)

        if relative_root == '.':
            relative_root = repo_name

        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]

        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    'classes': class_info,
                    'functions': function_names,
                    'text': file_lines,
                }
            else:
                curr_struct[file_name] = {}

    filter_none_python(structure)
    filter_out_test_files(structure)

    # Save the structure to a pickle file
    with open(pickle_file, 'wb') as pf_out:
        pickle.dump(structure, pf_out)

    return structure


def extract_observation(string, observation_delineator):
    return unescape_quotes(
        string.split(observation_delineator)[1].split(OBSERVATION_ENDING)[0]
    )


def agentless_file_localization(problem_statement):
    """Agentless file localization"""

    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    from Agentless.agentless.fl.FL import LLMFL
    from Agentless.agentless.util.preprocess_data import (
        correct_file_paths,
        get_full_file_paths_and_classes_and_functions,
        show_project_structure,
    )

    structure = retrieve_structure()

    message = LLMFL.obtain_relevant_files_prompt.format(
        problem_statement=problem_statement,
        structure=show_project_structure(structure).strip(),
    ).strip()
    # print(f'prompting with file localization message:\n{message}')
    # print('=' * 80)

    raw_output = (
        _get_openai_client()
        .chat.completions.create(
            model=_get_openai_model(),
            messages=[{'content': message, 'role': 'user'}],
            temperature=0,
            max_tokens=_get_max_token(),
        )
        .choices[0]
        .message.content
    )
    model_found_files = _parse_model_return_lines(raw_output)

    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)

    # Never match partial paths for consistency with main Agentless results
    found_files = correct_file_paths(model_found_files, files, False)[:TOP_N]

    # print("Returning file: ", AGENTLESS_FOUND_FILE_OBSERVATION + "\n---\n".join(found_files))
    with open(AGENTLESS_FOUND_FILE_PICKLE, 'wb') as file:
        pickle.dump(found_files, file)
    return escape_quotes(
        AGENTLESS_FOUND_FILE_OBSERVATION
        + '\n---\n'.join(found_files)
        + OBSERVATION_ENDING
    )


def agentless_related_localization(problem_statement, prev_observation):
    """Agentless related localization"""

    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    from Agentless.agentless.fl.FL import LLMFL
    from Agentless.agentless.util.api_requests import (
        num_tokens_from_messages,
    )
    from Agentless.agentless.util.compress_file import get_skeleton
    from Agentless.agentless.util.postprocess_data import (
        extract_code_blocks,
        extract_locs_for_files,
    )
    from Agentless.agentless.util.preprocess_data import (
        get_repo_files,
    )

    structure = retrieve_structure()

    import os
    import pickle

    file_path = AGENTLESS_FOUND_FILE_PICKLE
    found_files = None

    if os.path.exists(file_path):
        print('found file pickle')
        with open(file_path, 'rb') as file:
            found_files = pickle.load(file)
    else:
        found_files = extract_observation(
            prev_observation, AGENTLESS_FOUND_FILE_OBSERVATION
        ).split('\n---\n')

    print(found_files)

    file_contents = get_repo_files(structure, found_files)
    compressed_file_contents = {
        fn: get_skeleton(code) for fn, code in file_contents.items()
    }
    contents = [
        LLMFL.file_content_in_block_template.format(file_name=fn, file_content=code)
        for fn, code in compressed_file_contents.items()
    ]
    file_contents = ''.join(contents)
    template = (
        LLMFL.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
    )
    message = template.format(
        problem_statement=problem_statement, file_contents=file_contents
    )

    def message_too_long(message):
        return (
            num_tokens_from_messages(message, _get_openai_model()) >= MAX_CONTEXT_LENGTH
        )

    while message_too_long(message) and len(contents) > 1:
        # print(f'reducing to \n{len(contents)} files')
        contents = contents[:-1]
        file_contents = ''.join(contents)
        message = template.format(
            problem_statement=problem_statement, file_contents=file_contents
        )  # Recreate message

    if message_too_long(message):
        # print
        raise ValueError(
            'The remaining file content is too long to fit within the context length'
        )
    # print(f'prompting with related localization message:\n{message}')
    # print('=' * 80)

    raw_output = (
        _get_openai_client()
        .chat.completions.create(
            model=_get_openai_model(),
            messages=[{'content': message, 'role': 'user'}],
            temperature=0,
            max_tokens=_get_max_token(),
        )
        .choices[0]
        .message.content
    )

    model_found_locs = extract_code_blocks(raw_output)
    model_found_locs_separated = extract_locs_for_files(model_found_locs, found_files)

    loc_str = found_files[0] + ' : ' + model_found_locs_separated[0][0]
    for idx, loc in enumerate(model_found_locs_separated[1:]):
        loc_str += '\n---\n' + found_files[idx + 1] + ' : ' + loc[0]

    with open(AGENTLESS_RELATED_LOCATIONS_PICKLE, 'wb') as related_loc_file:
        pickle.dump(
            [found_files, [item[0] for item in model_found_locs_separated]],
            related_loc_file,
        )

    return escape_quotes(
        AGENTLESS_RELATED_LOCATIONS_OBSERVATION + loc_str + OBSERVATION_ENDING
    )


def agentless_line_level_localization(problem_statement, prev_observation, num_samples):
    """Agentless line localization"""

    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    structure = retrieve_structure()

    import os
    import pickle

    from Agentless.agentless.fl.FL import LLMFL
    from Agentless.agentless.repair.repair import (
        construct_topn_file_context,
    )
    from Agentless.agentless.util.api_requests import (
        num_tokens_from_messages,
    )
    from Agentless.agentless.util.postprocess_data import (
        extract_code_blocks,
        extract_locs_for_files,
    )
    from Agentless.agentless.util.preprocess_data import (
        get_repo_files,
    )

    file_path = AGENTLESS_RELATED_LOCATIONS_PICKLE
    locs = None

    if os.path.exists(file_path):
        print('found related pickle')
        with open(file_path, 'rb') as file:
            locs = pickle.load(file)
        file_names = locs[0]
        found_related_locs = locs[1]
    else:
        locs = extract_observation(
            prev_observation, AGENTLESS_RELATED_LOCATIONS_OBSERVATION
        ).split('\n---\n')
        file_names = [loc.split(' : ', 1)[0] for loc in locs]
        found_related_locs = [loc.split(' : ', 1)[1] for loc in locs]

    print(locs)

    coarse_locs = {}
    for i, pred_file in enumerate(file_names):
        if len(found_related_locs) > i:
            coarse_locs[pred_file] = found_related_locs[i]

    file_contents = get_repo_files(structure, file_names)
    topn_content, _ = construct_topn_file_context(
        coarse_locs,
        file_names,
        file_contents,
        structure,
        context_window=CONTEXT_WINDOW,
        loc_interval=True,
        add_space=ADD_SPACE,
        sticky_scroll=STICKY_SCROLL,
        no_line_number=NO_LINE_NUMBER,
    )
    if NO_LINE_NUMBER:
        template = LLMFL.obtain_relevant_code_combine_top_n_no_line_number_prompt
    else:
        template = LLMFL.obtain_relevant_code_combine_top_n_prompt
    message = template.format(
        problem_statement=problem_statement, file_contents=topn_content
    )
    # print(f'prompting with line localization message:\n{message}')
    # print('=' * 80)
    assert num_tokens_from_messages(message, _get_openai_model()) < MAX_CONTEXT_LENGTH

    raw_trajs = (
        _get_openai_client()
        .chat.completions.create(
            model=_get_openai_model(),
            messages=[{'content': message, 'role': 'user'}],
            temperature=TEMPERATURE,
            max_tokens=_get_max_token(),
            n=num_samples,
        )
        .choices
    )

    # Merge trajectories
    raw_outputs = [raw_traj.message.content for raw_traj in raw_trajs]

    model_found_locs_separated_in_samples = []
    for raw_output in raw_outputs:
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )
        model_found_locs_separated_in_samples.append(model_found_locs_separated)

        # print('==== raw output ====')
        # print(raw_output)
    # print("model_found_locs_separated_in_samples: ", model_found_locs_separated_in_samples)

    coarse_info = ''
    for fn, found_locs in coarse_locs.items():
        coarse_info += f'### {fn}\n'
        if isinstance(found_locs, str):
            coarse_info += found_locs + '\n'
        else:
            coarse_info += '\n'.join(found_locs) + '\n'

    gen_locs_with_files = []
    for gen_locs in model_found_locs_separated_in_samples:
        loc_list = []
        for idx, loc in enumerate(gen_locs):
            loc_list.append(file_names[idx] + ' : ' + loc[0])
        gen_locs_with_files.append(loc_list)

    with open(AGENTLESS_MULTI_LINE_LOCATION_PICKLE, 'wb') as file:
        pickle.dump(gen_locs_with_files, file)

    loc_str = convert_list_of_lists_to_str(gen_locs_with_files)

    return escape_quotes(
        AGENTLESS_LINE_LOCATIONS_OBSERVATION + loc_str + OBSERVATION_ENDING
    )


def get_file_contents(pred_files, structure):
    """Get file contents for Agentless"""
    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    from Agentless.agentless.util.preprocess_data import (
        get_full_file_paths_and_classes_and_functions,
    )

    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for pred_file in pred_files:
        content = None

        for file_content in files:
            if file_content[0] == pred_file:
                content = '\n'.join(file_content[1])
                file_contents[pred_file] = content
                break
    return file_contents


def agentless_repair(problem_statement, prev_observation, num_samples):
    """Agentless repair"""

    structure = retrieve_structure()

    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    import logging

    from Agentless.agentless.repair.repair import (
        _post_process_multifile_repair,
        construct_topn_file_context,
        repair_prompt_combine_topn,
        repair_prompt_combine_topn_cot,
        repair_prompt_combine_topn_cot_diff,
        repair_relevant_file_instruction,
    )

    logger = logging.getLogger('simpleLogger')

    import os
    import pickle

    file_path = AGENTLESS_LINE_LOCATIONS_PICKLE
    found_edit_locs_with_file = None

    if os.path.exists(file_path):
        print('found line pickle')
        with open(file_path, 'rb') as file:
            found_edit_locs_with_file = pickle.load(file)
    else:
        found_edit_locs_with_file = convert_str_to_list_of_lists(
            extract_observation(prev_observation, AGENTLESS_LINE_LOCATIONS_OBSERVATION)
        )

    print(found_edit_locs_with_file)

    pred_files = [item.split(' : ', 1)[0] for item in found_edit_locs_with_file]
    found_edit_locs = [item.split(' : ', 1)[1] for item in found_edit_locs_with_file]

    file_to_edit_locs = {}
    for i, pred_file in enumerate(pred_files):
        if len(found_edit_locs) > i:
            file_to_edit_locs[pred_file] = found_edit_locs[i]

    # Construct file contents
    file_contents = get_file_contents(pred_files, structure)

    # print('FILE CONTENTS')
    # print(file_contents)

    topn_content, file_loc_intervals = construct_topn_file_context(
        file_to_edit_locs,
        pred_files,
        file_contents,
        structure,
        context_window=CONTEXT_WINDOW,
        loc_interval=LOC_INTERVAL,
        fine_grain_loc_only=FINE_GRAIN_LOC_ONLY,
        add_space=ADD_SPACE,
        no_line_number=NO_LINE_NUMBER,
        sticky_scroll=STICKY_SCROLL,
    )

    if topn_content.strip() == '':
        return []

    prompt_template = (
        repair_prompt_combine_topn_cot_diff
        if COT and DIFF_FORMAT
        else repair_prompt_combine_topn_cot
        if COT
        else repair_prompt_combine_topn
    )
    file_instruction = repair_relevant_file_instruction
    message = prompt_template.format(
        repair_relevant_file_instruction=file_instruction,
        problem_statement=problem_statement,
        content=topn_content.rstrip(),
    ).strip()
    # print(f'prompting with repair message:\n{message}')

    all_generations, counts, _, prev_contents, file_names = [], [], [], [], []
    sample_responses: list[Any] = []
    # Using early stopping will cost more since the input tokens will be charged multiple times.
    # For now we disable it.
    assert STOP_AT_N_UNIQUE_VALID_SAMPLES == -1

    if not SKIP_GREEDY:
        greedy_traj: Any = (
            _get_openai_client()
            .chat.completions.create(
                model=_get_openai_model(),
                messages=[{'content': message, 'role': 'user'}],
                temperature=0,
                max_tokens=1024,
            )
            .choices[0]
        )
    sample_responses.append(greedy_traj)

    if num_samples - 1:
        sample_trajs: list[Any] = (
            _get_openai_client()
            .chat.completions.create(
                model=_get_openai_model(),
                messages=[{'content': message, 'role': 'user'}],
                temperature=TEMPERATURE,
                max_tokens=1024,
                n=num_samples - 1,
            )
            .choices
        )

    sample_responses.extend([item for item in sample_trajs])

    count = 0
    raw_outputs = []
    while count < num_samples:
        # print(f'trying the {count + 1}-th sample ...')
        if sample_responses:
            raw_output = sample_responses[count].message.content
            count += 1

            all_generations.append(raw_output)

            edited_file, new_content = _post_process_multifile_repair(
                raw_output,
                file_contents,
                logger,
                file_loc_intervals,
                diff_format=DIFF_FORMAT,
            )

            if new_content == '':
                prev_contents.append('')
                file_names.append('')
            else:
                prev_content = file_contents[edited_file]
                prev_contents.append(prev_content)
                file_names.append(edited_file)

            counts.append(count)
            raw_outputs.append(edited_file + ' : ' + raw_output)

    return raw_outputs


def agentless_repair_multi_context(problem_statement, prev_observation, num_samples):
    """Run Agentless repair for multiple contexts"""

    import os
    import pickle

    file_path = AGENTLESS_MULTI_LINE_LOCATION_PICKLE
    line_locations = None

    if os.path.exists(file_path):
        print('found multi line pickle')
        with open(file_path, 'rb') as file:
            line_locations = pickle.load(file)
    else:
        line_locations = convert_str_to_list_of_lists(
            extract_observation(prev_observation, AGENTLESS_LINE_LOCATIONS_OBSERVATION)
        )

    print(line_locations)

    # Create the merged locations
    merged_locs = [
        line_locations[0] + line_locations[1],
        line_locations[2] + line_locations[3],
    ]

    # print("merged_locations: ", merged_locs)
    repairs = []
    for loc in merged_locs:
        with open(AGENTLESS_LINE_LOCATIONS_PICKLE, 'wb') as file:
            pickle.dump(loc, file)

        loc_str = escape_quotes(
            AGENTLESS_LINE_LOCATIONS_OBSERVATION
            + convert_list_of_lists_to_str(loc)
            + OBSERVATION_ENDING
        )
        repair = agentless_repair(problem_statement, loc_str, num_samples)
        repairs.extend(repair)

    new_observation = escape_quotes(
        AGENTLESS_REPAIR_OBSERVATION
        + convert_list_of_lists_to_str([repairs, merged_locs])
        + OBSERVATION_ENDING
    )

    with open(AGENTLESS_REPAIR_PICKLE, 'wb') as file:
        pickle.dump([repairs, merged_locs], file)

    return new_observation


def post_process_raw_output(raw_output_text, file_contents, file_loc_intervals):
    """Helper function for Agentless post-processing"""
    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    """Agentless post processing helper function"""
    # Just to satisfy the requirement of passing a logger to the function
    import logging
    from difflib import unified_diff

    from Agentless.agentless.repair.repair import (
        _post_process_multifile_repair,
    )
    from Agentless.agentless.util.postprocess_data import (
        check_code_differ_by_just_empty_lines,
        check_syntax,
        fake_git_repo,
        lint_code,
    )

    logger = logging.getLogger('simpleLogger')

    git_diffs = ''
    raw_git_diffs = ''
    lint_success = False
    content = ''
    try:
        # This converts the patch to the git diff format
        edited_file, new_content = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=DIFF_FORMAT,
        )
        # print('EDITED FILE')
        # print(edited_file)
        # print('NEW CONTENT')
        # print(new_content)
        # This parts tries applying and linting the patch
        if (edited_file in file_contents) and new_content:
            content = file_contents[edited_file]

            git_diff = fake_git_repo('playground', edited_file, content, new_content)
            # print('1 git diff')
            # print(git_diff)

            raw_git_diffs += '\n' + git_diff.replace(
                '\\ No newline at end of file\n', ''
            )

            syntax_success = check_syntax(new_content)
            lint_success, prev_errors, errors = lint_code(
                'playground', 'test.py', new_content, file_contents[edited_file]
            )

            differ_by_empty_lines = check_code_differ_by_just_empty_lines(
                new_content, file_contents[edited_file]
            )

            # print(lint_success, prev_errors, errors, differ_by_empty_lines)
            # print(
            #     f'checks pass: {lint_success, prev_errors, errors, differ_by_empty_lines}'
            # )
            if syntax_success and not differ_by_empty_lines:
                git_diffs = raw_git_diffs
            else:
                git_diffs = ''  # no need to evaluate
        else:
            diff = list(
                unified_diff(
                    content.split('\n'),
                    new_content.split('\n'),
                    fromfile=edited_file,
                    tofile=edited_file,
                    lineterm='',
                )
            )
            print('Failed parsing diff!')
            print('\n'.join(diff))
    except Exception as e:
        print(raw_output_text)
        print(e)

    return git_diffs, raw_git_diffs, content


def agentless_post_process_repair(prev_observation):
    """
    apply some diff formatting. Also strips comments and trailing spaces to better identify identical patches
    """

    import sys

    sys.path.append('/workspace')
    sys.path.append('/workspace/Agentless')

    import os
    import pickle

    from Agentless.agentless.util.postprocess_data import normalize_patch
    from Agentless.agentless.util.preprocess_data import (
        transfer_arb_locs_to_locs,
    )

    file_path = AGENTLESS_REPAIR_PICKLE
    prev_observation_list = None

    if os.path.exists(file_path):
        print('found repair pickle')
        with open(file_path, 'rb') as file:
            prev_observation_list = pickle.load(file)
    else:
        prev_observation_list = convert_str_to_list_of_lists(
            extract_observation(
                prev_observation.replace('<newline_character>', '\\n'),
                AGENTLESS_REPAIR_OBSERVATION,
            )
        )

    print(prev_observation_list)

    repairs_with_files = prev_observation_list[0]
    all_found_edit_locs_with_files = prev_observation_list[1]
    generation_pred_files = [item.split(' : ', 1)[0] for item in repairs_with_files]
    raw_outputs = [item.split(' : ', 1)[1] for item in repairs_with_files]

    all_pred_files = []
    for loc_list in all_found_edit_locs_with_files:
        for loc in loc_list:
            all_pred_files.append(loc.split(' : ', 1)[0])

    structure = retrieve_structure()

    # print("all_pred_files: ", all_pred_files)
    generation_original_file_contents = get_file_contents(all_pred_files, structure)

    # Number of sections, this should correspond to the length of both_found_edit_locs
    num_sections = len(all_found_edit_locs_with_files)

    # Calculate the number of items in each section
    num_each_edit_loc = int(len(raw_outputs) / num_sections)

    processed_patches = []

    print('all_found_locs_with_files: ', all_found_edit_locs_with_files)

    for generation_idx, raw_output_text in enumerate(raw_outputs):
        # Determine the section index based on generation_idx
        section_index = generation_idx // num_each_edit_loc
        print('section index: ', section_index)

        pred_files = [
            item.split(' : ', 1)[0]
            for item in all_found_edit_locs_with_files[section_index]
        ]
        found_edit_locs = [
            item.split(' : ', 1)[1]
            for item in all_found_edit_locs_with_files[section_index]
        ]

        if raw_output_text != '':
            try:
                pred_file = generation_pred_files[generation_idx]
                original_file_content = generation_original_file_contents[pred_file]

                git_diffs = ''

                file_contents = {pred_file: original_file_content}

                file_loc_intervals = dict()

                for i, tmp_pred_file in enumerate(pred_files):
                    if tmp_pred_file != pred_file:
                        continue
                    if len(found_edit_locs) > i:
                        # try:
                        _, context_intervals = transfer_arb_locs_to_locs(
                            found_edit_locs[i],
                            None,
                            pred_files[i],
                            CONTEXT_WINDOW,
                            LOC_INTERVAL,
                            FINE_GRAIN_LOC_ONLY,
                            file_content=(
                                file_contents[pred_file]
                                if pred_file in file_contents
                                else ''
                            ),
                        )
                        # print('context interval')
                        # print(context_intervals)
                    else:
                        _, context_intervals = [], []  # default values.

            except Exception as e:
                print('file loc interval error')
                print(e)
                print(e)
                raw_output_text = ''

            file_loc_intervals[pred_file] = context_intervals

            # print('RAW OUTPUT TEXT')
            # print(raw_output_text)

            if raw_output_text:
                # print('FILE LOCAL INTERVAL')
                # print(file_loc_intervals)
                # print('FILE CONTENT')
                # print(file_contents)
                git_diffs, raw_git_diffs, content = post_process_raw_output(
                    raw_output_text, file_contents, file_loc_intervals
                )

                # Setting 0 as the instance_id since this function doesn't
                # actually use the instance_id for anything
                # print('GIT DIFF BEFORE NORMALIZING')
                # print(git_diffs)
                patch_lines = git_diffs.split('\n')
                patch_lines = [
                    line for line in patch_lines if not line.startswith('index')
                ]
                git_diffs = '\n'.join(patch_lines)

                normalized_patch = normalize_patch(0, git_diffs, original_file_content)

                normalized_lines = normalized_patch.split('\n')
                normalized_lines = [
                    line for line in normalized_lines if not line.startswith('index')
                ]
                normalized_patch = '\n'.join(patch_lines)

                if normalized_patch.lstrip():
                    processed_patches.append(
                        [
                            normalized_patch,
                            git_diffs.replace(
                                '\\ No newline at end of file\n', ''
                            ).lstrip(),
                        ]
                    )

    processed_patches = [patch for patch in processed_patches if patch[0].strip() != '']

    final_patch = most_frequent_string(processed_patches)
    print('final patch: ', final_patch)
    return AGENTLESS_FINAL_PATCH_OBSERVATION + final_patch + OBSERVATION_ENDING


def most_frequent_string(processed_patches):
    "Get most frequent patch"
    from collections import Counter

    if not processed_patches:
        return None  # Return None if the list is empty

    # Count the frequency of each string in the list
    counter = Counter([patch[0] for patch in processed_patches])

    # Find the string with the highest frequency
    most_common_normalized_patch, freq = counter.most_common(1)[0]
    # print('Number of times most frequent patch occured: ', freq)
    for patch in processed_patches:
        if patch[0] == most_common_normalized_patch:
            return patch[1]
    return ''


__all__ = [
    'agentless_file_localization',
    'agentless_related_localization',
    'agentless_line_level_localization',
    'install_agentless_dependencies',
    'install_agentless_dependencies_dummy',
    'agentless_repair',
    'agentless_repair_multi_context',
    'agentless_post_process_repair',
    'apply_git_patch',
]
