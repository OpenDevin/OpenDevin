"""file_ops.py

This module provides various file manipulation skills for the OpenHands agent.

Functions:
- open_file(path: str, line_number: int | None = 1, context_lines: int = 100): Opens a file and optionally moves to a specific line.
- goto_line(line_number: int): Moves the window to show the specified line number.
- scroll_down(): Moves the window down by the number of lines specified in WINDOW.
- scroll_up(): Moves the window up by the number of lines specified in WINDOW.
- create_file(filename: str): Creates and opens a new file with the given name.
- search_dir(search_term: str, dir_path: str = './'): Searches for a term in all files in the specified directory.
- search_file(search_term: str, file_path: str | None = None): Searches for a term in the specified file or the currently open file.
- find_file(file_name: str, dir_path: str = './'): Finds all files with the given name in the specified directory.
- edit_file_by_replace(file_name: str, to_replace: str, new_content: str): Replaces specific content in a file with new content.
- insert_content_at_line(file_name: str, line_number: int, content: str): Inserts given content at the specified line number in a file.
- append_file(file_name: str, content: str): Appends the given content to the end of the specified file.
"""

import os
import re
import shutil
import tempfile
import uuid

from openhands.linter import DefaultLinter, LintResult

CURRENT_FILE: str | None = None
CURRENT_LINE = 1
WINDOW = 100

# This is also used in unit tests!
MSG_FILE_UPDATED = '[File updated (edited at line {line_number}). Please review the changes and make sure they are correct (correct indentation, no duplicate lines, etc). Edit the file again if necessary.]'
LINTER_ERROR_MSG = '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'


# ==================================================================================================


def _output_error(error_msg: str) -> bool:
    print(f'ERROR: {error_msg}')
    return False


def _is_valid_filename(file_name) -> bool:
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        return False
    invalid_chars = '<>:"/\\|?*'
    if os.name == 'nt':  # Windows
        invalid_chars = '<>:"/\\|?*'
    elif os.name == 'posix':  # Unix-like systems
        invalid_chars = '\0'

    for char in invalid_chars:
        if char in file_name:
            return False
    return True


def _is_valid_path(path) -> bool:
    if not path or not isinstance(path, str):
        return False
    try:
        return os.path.exists(os.path.normpath(path))
    except PermissionError:
        return False


def _create_paths(file_name) -> bool:
    try:
        dirname = os.path.dirname(file_name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return True
    except PermissionError:
        return False


def _check_current_file(file_path: str | None = None) -> bool:
    global CURRENT_FILE
    if not file_path:
        file_path = CURRENT_FILE
    if not file_path or not os.path.isfile(file_path):
        return _output_error('No file open. Use the open_file function first.')
    return True


def _clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def _lint_file(file_path: str) -> tuple[str | None, int | None]:
    """Lint the file at the given path and return a tuple with a boolean indicating if there are errors,
    and the line number of the first error, if any.

    Returns:
        tuple[str | None, int | None]: (lint_error, first_error_line_number)
    """
    linter = DefaultLinter()
    lint_error: list[LintResult] = linter.lint(file_path)
    if not lint_error:
        # Linting successful. No issues found.
        return None, None
    first_error_line = lint_error[0].line if len(lint_error) > 0 else None
    error_text = 'ERRORS:\n' + '\n'.join(
        [f'{file_path}:{err.line}:{err.column}: {err.message}' for err in lint_error]
    )
    return error_text, first_error_line


def _print_window(
    file_path, targeted_line, window, return_str=False, ignore_window=False
):
    global CURRENT_LINE
    _check_current_file(file_path)
    with open(file_path) as file:
        content = file.read()

        # Ensure the content ends with a newline character
        if not content.endswith('\n'):
            content += '\n'

        lines = content.splitlines(True)  # Keep all line ending characters
        total_lines = len(lines)

        # cover edge cases
        CURRENT_LINE = _clamp(targeted_line, 1, total_lines)
        half_window = max(1, window // 2)
        if ignore_window:
            # Use CURRENT_LINE as starting line (for e.g. scroll_down)
            start = max(1, CURRENT_LINE)
            end = min(total_lines, CURRENT_LINE + window)
        else:
            # Ensure at least one line above and below the targeted line
            start = max(1, CURRENT_LINE - half_window)
            end = min(total_lines, CURRENT_LINE + half_window)

        # Adjust start and end to ensure at least one line above and below
        if start == 1:
            end = min(total_lines, start + window - 1)
        if end == total_lines:
            start = max(1, end - window + 1)

        output = ''

        # only display this when there's at least one line above
        if start > 1:
            output += f'({start - 1} more lines above)\n'
        else:
            output += '(this is the beginning of the file)\n'
        for i in range(start, end + 1):
            _new_line = f'{i}|{lines[i-1]}'
            if not _new_line.endswith('\n'):
                _new_line += '\n'
            output += _new_line
        if end < total_lines:
            output += f'({total_lines - end} more lines below)\n'
        else:
            output += '(this is the end of the file)\n'
        output = output.rstrip()

        if return_str:
            return output
        else:
            print(output)


def _cur_file_header(current_file, total_lines) -> str:
    if not current_file:
        return ''
    return f'[File: {os.path.abspath(current_file)} ({total_lines} lines total)]\n'


def open_file(
    path: str, line_number: int | None = 1, context_lines: int | None = WINDOW
) -> None:
    """Opens the file at the given path in the editor. IF the file is to be edited, first use `scroll_down` repeatedly to read the full file!
    If line_number is provided, the window will be moved to include that line.
    It only shows the first 100 lines by default! `context_lines` is the max number of lines to be displayed, up to 100. Use `scroll_up` and `scroll_down` to view more content up or down.

    Args:
        path: str: The path to the file to open, preferred absolute path.
        line_number: int | None = 1: The line number to move to. Defaults to 1.
        context_lines: int | None = 100: Only shows this number of lines in the context window (usually from line 1), with line_number as the center (if possible). Defaults to 100.
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW

    if not os.path.isfile(path):
        _output_error(f'File {path} not found.')
        return

    CURRENT_FILE = os.path.abspath(path)
    with open(CURRENT_FILE) as file:
        total_lines = max(1, sum(1 for _ in file))

    if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
        _output_error(f'Line number must be between 1 and {total_lines}')
        return
    CURRENT_LINE = line_number

    # Override WINDOW with context_lines
    if context_lines is None or context_lines < 1:
        context_lines = WINDOW

    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE,
        CURRENT_LINE,
        _clamp(context_lines, 1, 300),
        return_str=True,
        ignore_window=False,
    )
    if output.strip().endswith('more lines below)'):
        output += '\n[Use `scroll_down` to view the next 100 lines of the file!]'
    print(output)


def goto_line(line_number: int) -> None:
    """Moves the window to show the specified line number.

    Args:
        line_number: int: The line number to move to.
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    _check_current_file()

    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    if not isinstance(line_number, int) or line_number < 1 or line_number > total_lines:
        _output_error(f'Line number must be between 1 and {total_lines}.')
        return

    CURRENT_LINE = _clamp(line_number, 1, total_lines)

    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=False
    )
    print(output)


def scroll_down() -> None:
    """Moves the window down by 100 lines.

    Args:
        None
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    _check_current_file()

    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    CURRENT_LINE = _clamp(CURRENT_LINE + WINDOW, 1, total_lines)
    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=True
    )
    print(output)


def scroll_up() -> None:
    """Moves the window up by 100 lines.

    Args:
        None
    """
    global CURRENT_FILE, CURRENT_LINE, WINDOW
    _check_current_file()

    with open(str(CURRENT_FILE)) as file:
        total_lines = max(1, sum(1 for _ in file))
    CURRENT_LINE = _clamp(CURRENT_LINE - WINDOW, 1, total_lines)
    output = _cur_file_header(CURRENT_FILE, total_lines)
    output += _print_window(
        CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True, ignore_window=True
    )
    print(output)


def create_file(filename: str) -> None:
    """Creates and opens a new file with the given name.

    Args:
        filename: str: The name of the file to create.
    """
    if os.path.exists(filename):
        _output_error(f"File '{filename}' already exists.")
        return

    with open(filename, 'w') as file:
        file.write('\n')

    open_file(filename)
    print(f'[File {filename} created.]')


class LineNumberError(Exception):
    pass


def _append_impl(lines, content):
    """Internal method to handle appending to a file.

    Args:
        lines: list[str]: The lines in the original file.
        content: str: The content to append to the file.

    Returns:
        content: str: The new content of the file.
        n_added_lines: int: The number of lines added to the file.
    """
    content_lines = content.splitlines(keepends=True)
    n_added_lines = len(content_lines)
    if lines and not (len(lines) == 1 and lines[0].strip() == ''):
        # file is not empty
        if not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        new_lines = lines + content_lines
        content = ''.join(new_lines)
    else:
        # file is empty
        content = ''.join(content_lines)

    return content, n_added_lines


def _insert_impl(lines, start, content):
    """Internal method to handle inserting to a file.

    Args:
        lines: list[str]: The lines in the original file.
        start: int: The start line number for inserting.
        content: str: The content to insert to the file.

    Returns:
        content: str: The new content of the file.
        n_added_lines: int: The number of lines added to the file.

    Raises:
        LineNumberError: If the start line number is invalid.
    """
    inserted_lines = [content + '\n' if not content.endswith('\n') else content]
    if len(lines) == 0:
        new_lines = inserted_lines
    elif start is not None:
        if len(lines) == 1 and lines[0].strip() == '':
            # if the file with only 1 line and that line is empty
            lines = []

        if len(lines) == 0:
            new_lines = inserted_lines
        else:
            new_lines = lines[: start - 1] + inserted_lines + lines[start - 1 :]
    else:
        raise LineNumberError(
            f'Invalid line number: {start}. Line numbers must be between 1 and {len(lines)} (inclusive).'
        )

    content = ''.join(new_lines)
    n_added_lines = len(inserted_lines)
    return content, n_added_lines


def _edit_impl(lines, start, end, content):
    """Internal method to handle editing a file.

    REQUIRES (should be checked by caller):
        start <= end
        start and end are between 1 and len(lines) (inclusive)
        content ends with a newline

    Args:
        lines: list[str]: The lines in the original file.
        start: int: The start line number for editing.
        end: int: The end line number for editing.
        content: str: The content to replace the lines with.

    Returns:
        content: str: The new content of the file.
        n_added_lines: int: The number of lines added to the file.
    """
    # Handle cases where start or end are None
    if start is None:
        start = 1  # Default to the beginning
    if end is None:
        end = len(lines)  # Default to the end
    # Check arguments
    if not (1 <= start <= len(lines)):
        raise LineNumberError(
            f'Invalid start line number: {start}. Line numbers must be between 1 and {len(lines)} (inclusive).'
        )
    if not (1 <= end <= len(lines)):
        raise LineNumberError(
            f'Invalid end line number: {end}. Line numbers must be between 1 and {len(lines)} (inclusive).'
        )
    if start > end:
        raise LineNumberError(
            f'Invalid line range: {start}-{end}. Start must be less than or equal to end.'
        )

    if not content.endswith('\n'):
        content += '\n'
    content_lines = content.splitlines(True)
    n_added_lines = len(content_lines)
    new_lines = lines[: start - 1] + content_lines + lines[end:]
    content = ''.join(new_lines)
    return content, n_added_lines


def _subtract_strings(str1, str2) -> str:
    """Extract the linter errors caused by the modification.

    Args:
        str1: str: The original linter errors.
        str2: str: The linter errors after the modification.

    Returns:
        str: The linter errors caused by the modification.
    """

    def _extract_last_part(line):
        parts = line.split(':')
        if len(parts) > 1:
            return parts[-1].strip()
        return line.strip()

    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    last_parts1 = [_extract_last_part(line) for line in lines1]

    remaining_lines = [
        line for line in lines2 if _extract_last_part(line) not in last_parts1
    ]

    result = '\n'.join(remaining_lines)
    return result


def _edit_file_impl(
    file_name: str,
    start: int | None = None,
    end: int | None = None,
    content: str = '',
    is_insert: bool = False,
    is_append: bool = False,
) -> str | None:
    """Internal method to handle common logic for edit_/append_file methods.

    Args:
        file_name: str: The name of the file to edit or append to.
        start: int | None = None: The start line number for editing. Ignored if is_append is True.
        end: int | None = None: The end line number for editing. Ignored if is_append is True.
        content: str: The content to replace the lines with or to append.
        is_insert: bool = False: Whether to insert content at the given line number instead of editing.
        is_append: bool = False: Whether to append content to the file instead of editing.
    """
    ret_str = ''
    global CURRENT_FILE, CURRENT_LINE, WINDOW

    ERROR_MSG = f'[Error editing file {file_name}. Please confirm the file is correct.]'
    ERROR_MSG_SUFFIX = (
        'Your changes have NOT been applied. Please fix your edit command and try again.\n'
        'You either need to 1) Open the correct file and try again or 2) Specify the correct line number arguments.\n'
        'DO NOT re-run the same failed edit command. Running it again will lead to the same error.'
    )

    if not _is_valid_filename(file_name):
        _output_error('Invalid file name.')
        return None

    if not _is_valid_path(file_name):
        _output_error('Invalid path or file name.')
        return None

    if not _create_paths(file_name):
        _output_error('Could not access or create directories.')
        return None

    if not os.path.isfile(file_name):
        _output_error(f'File {file_name} not found.')
        return None

    if is_insert and is_append:
        _output_error('Cannot insert and append at the same time.')
        return None

    # Use a temporary file to write changes
    content = str(content or '')
    temp_file_path = ''
    first_error_line = None

    try:
        n_added_lines = None

        # lint the original file
        enable_auto_lint = os.getenv('ENABLE_AUTO_LINT', 'false').lower() == 'true'
        if enable_auto_lint:
            # Copy the original file to a temporary file (with the same ext) and lint it
            suffix = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix) as orig_file_clone:
                shutil.copy2(file_name, orig_file_clone.name)
                original_lint_error, _ = _lint_file(orig_file_clone.name)

        # Create a temporary file in the same directory as the original file
        original_dir = os.path.dirname(file_name)
        original_ext = os.path.splitext(file_name)[1]
        temp_file_name = f'.temp_{uuid.uuid4().hex}{original_ext}'
        temp_file_path = os.path.join(original_dir, temp_file_name)

        with open(temp_file_path, 'w') as temp_file:
            # Read the original file and check if empty and for a trailing newline
            with open(file_name) as original_file:
                lines = original_file.readlines()

            if is_append:
                content, n_added_lines = _append_impl(lines, content)
            elif is_insert:
                try:
                    content, n_added_lines = _insert_impl(lines, start, content)
                except LineNumberError as e:
                    ret_str += (f'{ERROR_MSG}\n' f'{e}\n' f'{ERROR_MSG_SUFFIX}') + '\n'
                    return ret_str
            else:
                try:
                    content, n_added_lines = _edit_impl(lines, start, end, content)
                except LineNumberError as e:
                    ret_str += (f'{ERROR_MSG}\n' f'{e}\n' f'{ERROR_MSG_SUFFIX}') + '\n'
                    return ret_str

            if not content.endswith('\n'):
                content += '\n'

            # Write the new content to the temporary file
            temp_file.write(content)

        # Replace the original file with the temporary file
        os.replace(temp_file_path, file_name)

        # Handle linting
        # NOTE: we need to get env var inside this function
        # because the env var will be set AFTER the agentskills is imported
        if enable_auto_lint:
            # Generate a random temporary file path
            suffix = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tfile:
                original_file_backup_path = tfile.name

            with open(original_file_backup_path, 'w') as f:
                f.writelines(lines)

            file_name_abs = os.path.abspath(file_name)
            lint_error, first_error_line = _lint_file(file_name_abs)

            # Select the errors caused by the modification
            def extract_last_part(line):
                parts = line.split(':')
                if len(parts) > 1:
                    return parts[-1].strip()
                return line.strip()

            def subtract_strings(str1, str2) -> str:
                lines1 = str1.splitlines()
                lines2 = str2.splitlines()

                last_parts1 = [extract_last_part(line) for line in lines1]

                remaining_lines = [
                    line
                    for line in lines2
                    if extract_last_part(line) not in last_parts1
                ]

                result = '\n'.join(remaining_lines)
                return result

            if original_lint_error and lint_error:
                lint_error = _subtract_strings(original_lint_error, lint_error)
                if lint_error == '':
                    lint_error = None
                    first_error_line = None

            if lint_error is not None:
                if first_error_line is not None:
                    show_line = int(first_error_line)
                elif is_append:
                    # original end-of-file
                    show_line = len(lines)
                # insert OR edit WILL provide meaningful line numbers
                elif start is not None and end is not None:
                    show_line = int((start + end) / 2)
                else:
                    raise ValueError('Invalid state. This should never happen.')

                ret_str += LINTER_ERROR_MSG
                ret_str += lint_error + '\n'

                editor_lines = n_added_lines + 20
                sep = '-' * 49 + '\n'
                ret_str += (
                    f'[This is how your edit would have looked if applied]\n{sep}'
                )
                ret_str += (
                    _print_window(file_name, show_line, editor_lines, return_str=True)
                    + '\n'
                )
                ret_str += f'{sep}\n'

                ret_str += '[This is the original code before your edit]\n'
                ret_str += sep
                ret_str += (
                    _print_window(
                        original_file_backup_path,
                        show_line,
                        editor_lines,
                        return_str=True,
                    )
                    + '\n'
                )
                ret_str += sep
                ret_str += (
                    'Your changes have NOT been applied. Please fix your edit command and try again.\n'
                    'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.\n'
                    'DO NOT re-run the same failed edit command. Running it again will lead to the same error.'
                )

                # recover the original file
                with open(original_file_backup_path) as fin, open(
                    file_name, 'w'
                ) as fout:
                    fout.write(fin.read())

                # Don't forget to remove the temporary file after you're done
                os.unlink(original_file_backup_path)
                return ret_str

    except FileNotFoundError as e:
        ret_str += f'File not found: {e}\n'
    except PermissionError as e:
        ret_str += f'Permission error during file operation: {str(e)}\n'
    except IOError as e:
        ret_str += f'An error occurred while handling the file: {e}\n'
    except ValueError as e:
        ret_str += f'Invalid input: {e}\n'
    except Exception as e:
        # Clean up the temporary file if an error occurs
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f'An unexpected error occurred: {e}')
        raise e

    # Update the file information and print the updated content
    with open(file_name, 'r', encoding='utf-8') as file:
        n_total_lines = max(1, len(file.readlines()))
    if first_error_line is not None and int(first_error_line) > 0:
        CURRENT_LINE = first_error_line
    else:
        if is_append:
            CURRENT_LINE = max(1, len(lines))  # end of original file
        else:
            CURRENT_LINE = start or n_total_lines or 1
    ret_str += f'[File: {os.path.abspath(file_name)} ({n_total_lines} lines total after edit)]\n'
    CURRENT_FILE = file_name
    ret_str += _print_window(CURRENT_FILE, CURRENT_LINE, WINDOW, return_str=True) + '\n'
    ret_str += MSG_FILE_UPDATED.format(line_number=CURRENT_LINE)
    return ret_str


def edit_file_by_replace(file_name: str, to_replace: str, new_content: str) -> None:
    """Edit an existing file. This will search for non-empty `to_replace` in the given file and replace it with non-empty `new_content`.
    `to_replace` and `new_content` must be different! Split large edits into multiple smaller edits if necessary!
    Use `append_file` method for writing after `create_file`!

    Every *to_replace* must *EXACTLY MATCH* the existing source code, character for character, including all comments, docstrings, etc.

    Include enough lines to make code in `to_replace` unique. `to_replace` should NOT be empty.

    For example, given a file "/workspace/example.txt" with the following content:
    ```
    line 1
    line 2
    line 2
    line 3
    ```

    EDITING: If you want to replace the second occurrence of "line 2", you can make `to_replace` unique:

    edit_file_by_replace(
        '/workspace/example.txt',
        to_replace='line 2\nline 3',
        new_content='new line\nline 3',
    )

    This will replace only the second "line 2" with "new line". The first "line 2" will remain unchanged.

    The resulting file will be:
    ```
    line 1
    line 2
    new line
    line 3
    ```

    REMOVAL: If you want to remove "line 2" and "line 3", you can set `new_content` to an empty string:

    edit_file_by_replace(
        '/workspace/example.txt',
        to_replace='line 2\nline 3',
        new_content='',
    )

    Args:
        file_name: str: The name of the file to edit.
        to_replace: str: The content to search for and replace.
        new_content: str: The new content to replace the old content with.
    """
    # FIXME: support replacing *all* occurrences
    if to_replace is None or to_replace.strip() == '':
        _output_error('`to_replace` must not be empty.')
        return

    if to_replace == new_content:
        _output_error('`to_replace` and `new_content` must be different.')
        return

    if not os.path.isfile(file_name):
        _output_error(f'File {file_name} not found.')
        return None

    # search for `to_replace` in the file
    # if found, replace it with `new_content`
    # if not found, perform a fuzzy search to find the closest match and replace it with `new_content`
    with open(file_name, 'r') as file:
        file_content = file.read()

    if file_content.count(to_replace) > 1:
        _output_error(
            '`to_replace` appears more than once, please include enough lines to make code in `to_replace` unique.'
        )
        return

    start = file_content.find(to_replace)
    if start != -1:
        # Convert start from index to line number
        start_line_number = file_content[:start].count('\n') + 1
        end_line_number = start_line_number + len(to_replace.splitlines()) - 1
    else:

        def _fuzzy_transform(s: str) -> str:
            # remove all space except newline
            return re.sub(r'[^\S\n]+', '', s)

        # perform a fuzzy search (remove all spaces except newlines)
        to_replace_fuzzy = _fuzzy_transform(to_replace)
        file_content_fuzzy = _fuzzy_transform(file_content)
        # find the closest match
        start = file_content_fuzzy.find(to_replace_fuzzy)
        if start == -1:
            print(
                f'[No exact match found in {file_name} for\n```\n{to_replace}\n```\n]'
            )
            return
        # Convert start from index to line number for fuzzy match
        start_line_number = file_content_fuzzy[:start].count('\n') + 1
        end_line_number = start_line_number + len(to_replace.splitlines()) - 1

    ret_str = _edit_file_impl(
        file_name,
        start=start_line_number,
        end=end_line_number,
        content=new_content,
        is_insert=False,
    )
    # lint_error = bool(LINTER_ERROR_MSG in ret_str)
    # TODO: automatically tries to fix linter error (maybe involve some static analysis tools on the location near the edit to figure out indentation)
    if ret_str is not None:
        print(ret_str)


def insert_content_at_line(file_name: str, line_number: int, content: str) -> None:
    """Insert content at the given line number in a file.
    This will NOT modify the content of the lines before OR after the given line number.

    For example, if the file has the following content:
    ```
    line 1
    line 2
    line 3
    ```
    and you call `insert_content_at_line('file.txt', 2, 'new line')`, the file will be updated to:
    ```
    line 1
    new line
    line 2
    line 3
    ```

    Args:
        file_name: str: The name of the file to edit.
        line_number: int: The line number (starting from 1) to insert the content after.
        content: str: The content to insert.
    """
    ret_str = _edit_file_impl(
        file_name,
        start=line_number,
        end=line_number,
        content=content,
        is_insert=True,
        is_append=False,
    )
    if ret_str is not None:
        print(ret_str)


def append_file(file_name: str, content: str) -> None:
    """Append content to the given file.
    It appends text `content` to the end of the specified file, ideal after a `create_file`!

    Args:
        file_name: str: The name of the file to edit.
        content: str: The content to insert.
    """
    ret_str = _edit_file_impl(
        file_name,
        start=None,
        end=None,
        content=content,
        is_insert=False,
        is_append=True,
    )
    if ret_str is not None:
        print(ret_str)


def search_dir(search_term: str, dir_path: str = './') -> None:
    """Searches for search_term in all files in dir. If dir is not provided, searches in the current directory.

    Args:
        search_term: str: The term to search for.
        dir_path: str: The path to the directory to search.
    """
    if not os.path.isdir(dir_path):
        _output_error(f'Directory {dir_path} not found')
        return
    matches = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.startswith('.'):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if search_term in line:
                        matches.append((file_path, line_num, line.strip()))

    if not matches:
        print(f'No matches found for "{search_term}" in {dir_path}')
        return

    num_matches = len(matches)
    num_files = len(set(match[0] for match in matches))

    if num_files > 100:
        print(
            f'More than {num_files} files matched for "{search_term}" in {dir_path}. Please narrow your search.'
        )
        return

    print(f'[Found {num_matches} matches for "{search_term}" in {dir_path}]')
    for file_path, line_num, line in matches:
        print(f'{file_path} (Line {line_num}): {line}')
    print(f'[End of matches for "{search_term}" in {dir_path}]')


def search_file(search_term: str, file_path: str | None = None) -> None:
    """Searches for search_term in file. If file is not provided, searches in the current open file.

    Args:
        search_term: str: The term to search for.
        file_path: str | None: The path to the file to search.
    """
    global CURRENT_FILE
    if file_path is None:
        file_path = CURRENT_FILE
    if file_path is None:
        _output_error('No file specified or open. Use the open_file function first.')
        return
    if not os.path.isfile(file_path):
        _output_error(f'File {file_path} not found.')
        return

    matches = []
    with open(file_path) as file:
        for i, line in enumerate(file, 1):
            if search_term in line:
                matches.append((i, line.strip()))

    if matches:
        print(f'[Found {len(matches)} matches for "{search_term}" in {file_path}]')
        for match in matches:
            print(f'Line {match[0]}: {match[1]}')
        print(f'[End of matches for "{search_term}" in {file_path}]')
    else:
        print(f'[No matches found for "{search_term}" in {file_path}]')


def find_file(file_name: str, dir_path: str = './') -> None:
    """Finds all files with the given name in the specified directory.

    Args:
        file_name: str: The name of the file to find.
        dir_path: str: The path to the directory to search.
    """
    if not os.path.isdir(dir_path):
        _output_error(f'Directory {dir_path} not found')
        return

    matches = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file_name in file:
                matches.append(os.path.join(root, file))

    if matches:
        print(f'[Found {len(matches)} matches for "{file_name}" in {dir_path}]')
        for match in matches:
            print(f'{match}')
        print(f'[End of matches for "{file_name}" in {dir_path}]')
    else:
        print(f'[No matches found for "{file_name}" in {dir_path}]')


__all__ = [
    'open_file',
    'goto_line',
    'scroll_down',
    'scroll_up',
    'search_dir',
    'search_file',
    'find_file',
]
