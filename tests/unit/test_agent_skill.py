import contextlib
import io
import os
import sys
from unittest.mock import patch

import docx
import pytest

from openhands.runtime.plugins.agent_skills.file_ops.file_ops import (
    MSG_FILE_UPDATED,
    WINDOW,
    _print_window,
    append_file,
    create_file,
    edit_file_by_replace,
    find_file,
    goto_line,
    insert_content_at_line,
    open_file,
    scroll_down,
    scroll_up,
    search_dir,
    search_file,
)
from openhands.runtime.plugins.agent_skills.file_reader.file_readers import (
    parse_docx,
    parse_latex,
    parse_pdf,
    parse_pptx,
)
from openhands.runtime.plugins.agent_skills.utils.aider import Linter


# CURRENT_FILE must be reset for each test
@pytest.fixture(autouse=True)
def reset_current_file():
    from openhands.runtime.plugins.agent_skills import agentskills

    agentskills.CURRENT_FILE = None


def _numbered_test_lines(start, end) -> str:
    return ('\n'.join(f'{i}|' for i in range(start, end + 1))) + '\n'


def _generate_test_file_with_lines(temp_path, num_lines) -> str:
    file_path = temp_path / 'test_file.py'
    file_path.write_text('\n' * num_lines)
    return file_path


def _generate_ruby_test_file_with_lines(temp_path, num_lines) -> str:
    file_path = temp_path / 'test_file.rb'
    file_path.write_text('\n' * num_lines)
    return file_path


def _calculate_window_bounds(current_line, total_lines, window_size):
    """Calculate the bounds of the window around the current line."""
    half_window = window_size // 2
    if current_line - half_window < 0:
        start = 1
        end = window_size
    else:
        start = current_line - half_window
        end = current_line + half_window
    return start, end


def _generate_ruby_test_file_with_lines(temp_path, num_lines) -> str:
    file_path = temp_path / 'test_file.rb'
    file_path.write_text('\n' * num_lines)
    return file_path


def _capture_file_operation_error(operation, expected_error_msg):
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            operation()
        result = buf.getvalue().strip()
    assert result == expected_error_msg


SEP = '-' * 49 + '\n'


# =============================================================================


def test_open_file_unexist_path():
    _capture_file_operation_error(
        lambda: open_file('/unexist/path/a.txt'),
        'ERROR: File /unexist/path/a.txt not found.',
    )


def test_open_file(tmp_path):
    assert tmp_path is not None
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))
        result = buf.getvalue()
    assert result is not None
    expected = (
        f'[File: {temp_file_path} (5 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|Line 1\n'
        '2|Line 2\n'
        '3|Line 3\n'
        '4|Line 4\n'
        '5|Line 5\n'
        '(this is the end of the file)\n'
    )
    assert result.split('\n') == expected.split('\n')


def test_open_file_with_indentation(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\n    Line 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))
        result = buf.getvalue()
    assert result is not None
    expected = (
        f'[File: {temp_file_path} (5 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|Line 1\n'
        '2|    Line 2\n'
        '3|Line 3\n'
        '4|Line 4\n'
        '5|Line 5\n'
        '(this is the end of the file)\n'
    )
    assert result.split('\n') == expected.split('\n')


def test_open_file_long(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = '\n'.join([f'Line {i}' for i in range(1, 1001)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path), 1, 50)
        result = buf.getvalue()
    assert result is not None
    expected = f'[File: {temp_file_path} (1000 lines total)]\n'
    expected += '(this is the beginning of the file)\n'
    for i in range(1, 51):
        expected += f'{i}|Line {i}\n'
    expected += '(950 more lines below)\n'
    expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
    assert result.split('\n') == expected.split('\n')


def test_open_file_long_with_lineno(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = '\n'.join([f'Line {i}' for i in range(1, 1001)])
    temp_file_path.write_text(content)

    cur_line = 100

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path), cur_line)
        result = buf.getvalue()
    assert result is not None
    expected = f'[File: {temp_file_path} (1000 lines total)]\n'
    # since 100 is < WINDOW and 100 - WINDOW//2 < 0, so it should show all lines from 1 to WINDOW

    start, end = _calculate_window_bounds(cur_line, 1000, WINDOW)
    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == 1000:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({1000 - end} more lines below)\n'
        expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
    assert result.split('\n') == expected.split('\n')


def test_create_file_unexist_path():
    with pytest.raises(FileNotFoundError):
        create_file('/unexist/path/a.txt')


def test_create_file(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            create_file(str(temp_file_path))
        result = buf.getvalue()

    expected = (
        f'[File: {temp_file_path} (1 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|\n'
        '(this is the end of the file)\n'
        f'[File {temp_file_path} created.]\n'
    )
    assert result.split('\n') == expected.split('\n')


def test_goto_line(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    total_lines = 1000
    content = '\n'.join([f'Line {i}' for i in range(1, total_lines + 1)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    expected += '(this is the beginning of the file)\n'
    for i in range(1, WINDOW + 1):
        expected += f'{i}|Line {i}\n'
    expected += f'({total_lines - WINDOW} more lines below)\n'
    expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
    assert result.split('\n') == expected.split('\n')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            goto_line(500)
        result = buf.getvalue()
    assert result is not None

    cur_line = 500
    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    start, end = _calculate_window_bounds(cur_line, total_lines, WINDOW)
    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == total_lines:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({total_lines - end} more lines below)\n'
    assert result.split('\n') == expected.split('\n')


def test_goto_line_negative(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = '\n'.join([f'Line {i}' for i in range(1, 5)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))

    _capture_file_operation_error(
        lambda: goto_line(-1), 'ERROR: Line number must be between 1 and 4.'
    )


def test_goto_line_out_of_bound(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = '\n'.join([f'Line {i}' for i in range(1, 10)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))

    _capture_file_operation_error(
        lambda: goto_line(100), 'ERROR: Line number must be between 1 and 9.'
    )


def test_scroll_down(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    total_lines = 1000
    content = '\n'.join([f'Line {i}' for i in range(1, total_lines + 1)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    start, end = _calculate_window_bounds(1, total_lines, WINDOW)
    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == total_lines:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({total_lines - end} more lines below)\n'
        expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
    assert result.split('\n') == expected.split('\n')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            scroll_down()
        result = buf.getvalue()
    assert result is not None

    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    start = WINDOW + 1
    end = 2 * WINDOW + 1
    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == total_lines:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({total_lines - end} more lines below)\n'
    assert result.split('\n') == expected.split('\n')


def test_scroll_up(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    total_lines = 1000
    content = '\n'.join([f'Line {i}' for i in range(1, total_lines + 1)])
    temp_file_path.write_text(content)

    cur_line = 300
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path), cur_line)
        result = buf.getvalue()
    assert result is not None

    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    start, end = _calculate_window_bounds(cur_line, total_lines, WINDOW)
    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == total_lines:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({total_lines - end} more lines below)\n'
        expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
    assert result.split('\n') == expected.split('\n')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            scroll_up()
        result = buf.getvalue()
    assert result is not None

    cur_line = cur_line - WINDOW

    expected = f'[File: {temp_file_path} ({total_lines} lines total)]\n'
    start = cur_line
    end = cur_line + WINDOW

    if start == 1:
        expected += '(this is the beginning of the file)\n'
    else:
        expected += f'({start - 1} more lines above)\n'
    for i in range(start, end + 1):
        expected += f'{i}|Line {i}\n'
    if end == total_lines:
        expected += '(this is the end of the file)\n'
    else:
        expected += f'({total_lines - end} more lines below)\n'
    assert result.split('\n') == expected.split('\n')


def test_scroll_down_edge(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = '\n'.join([f'Line {i}' for i in range(1, 10)])
    temp_file_path.write_text(content)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            open_file(str(temp_file_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[File: {temp_file_path} (9 lines total)]\n'
    expected += '(this is the beginning of the file)\n'
    for i in range(1, 10):
        expected += f'{i}|Line {i}\n'
    expected += '(this is the end of the file)\n'

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            scroll_down()
        result = buf.getvalue()
    assert result is not None

    # expected should be unchanged
    assert result.split('\n') == expected.split('\n')


def test_print_window_internal(tmp_path):
    test_file_path = tmp_path / 'a.txt'
    create_file(str(test_file_path))
    open_file(str(test_file_path))
    with open(test_file_path, 'w') as file:
        for i in range(1, 101):
            file.write(f'Line `{i}`\n')

    # Define the parameters for the test
    current_line = 50
    window = 2

    # Test _print_window especially with backticks
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            _print_window(str(test_file_path), current_line, window, return_str=False)
        result = buf.getvalue()
        expected = (
            '(48 more lines above)\n'
            '49|Line `49`\n'
            '50|Line `50`\n'
            '51|Line `51`\n'
            '(49 more lines below)\n'
        )
        assert result == expected


def test_open_file_large_line_number(tmp_path):
    test_file_path = tmp_path / 'a.txt'
    create_file(str(test_file_path))
    open_file(str(test_file_path))
    with open(test_file_path, 'w') as file:
        for i in range(1, 1000):
            file.write(f'Line `{i}`\n')

    # Define the parameters for the test
    current_line = 800
    window = 100

    # Test _print_window especially with backticks
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            # _print_window(str(test_file_path), current_line, window, return_str=False)
            open_file(str(test_file_path), current_line, window)
        result = buf.getvalue()
        expected = f'[File: {test_file_path} (999 lines total)]\n'
        expected += '(749 more lines above)\n'
        for i in range(750, 850 + 1):
            expected += f'{i}|Line `{i}`\n'
        expected += '(149 more lines below)\n'
        expected += '[Use `scroll_down` to view the next 100 lines of the file!]\n'
        assert result == expected


def test_edit_file_by_replace_window(tmp_path):
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        content = """def any_int(a, b, c):
    return isinstance(a, int) and isinstance(b, int) and isinstance(c, int)

def test_any_int():
    assert any_int(1, 2, 3) == True
    assert any_int(1.5, 2, 3) == False
    assert any_int(1, 2.5, 3) == False
    assert any_int(1, 2, 3.5) == False
    assert any_int(1.0, 2, 3) == False
    assert any_int(1, 2.0, 3) == False
    assert any_int(1, 2, 3.0) == False
    assert any_int(0, 0, 0) == True
    assert any_int(-1, -2, -3) == True
    assert any_int(1, -2, 3) == True
    assert any_int(1.5, -2, 3) == False
    assert any_int(1, -2.5, 3) == False

def check(any_int):
    # Check some simple cases
    assert any_int(2, 3, 1)==True, "This prints if this assert fails 1 (good for debugging!)"
    assert any_int(2.5, 2, 3)==False, "This prints if this assert fails 2 (good for debugging!)"
    assert any_int(1.5, 5, 3.5)==False, "This prints if this assert fails 3 (good for debugging!)"
    assert any_int(2, 6, 2)==False, "This prints if this assert fails 4 (good for debugging!)"
    assert any_int(4, 2, 2)==True, "This prints if this assert fails 5 (good for debugging!)"
    assert any_int(2.2, 2.2, 2.2)==False, "This prints if this assert fails 6 (good for debugging!)"
    assert any_int(-4, 6, 2)==True, "This prints if this assert fails 7 (good for debugging!)"

    # Check some edge cases that are easy to work out by hand.
    assert any_int(2,1,1)==True, "This prints if this assert fails 8 (also good for debugging!)"
    assert any_int(3,4,7)==True, "This prints if this assert fails 9 (also good for debugging!)"
    assert any_int(3.0,4,7)==False, "This prints if this assert fails 10 (also good for debugging!)"

check(any_int)"""

        temp_file_path = tmp_path / 'error-test.py'
        temp_file_path.write_text(content)

        open_file(str(temp_file_path))

        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                edit_file_by_replace(
                    str(temp_file_path),
                    to_replace='    assert any_int(1.0, 2, 3) == False',
                    new_content='        assert any_int(1.0, 2, 3) == False',
                )
            result = buf.getvalue()
            expected = (
                '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
                'ERRORS:\n'
                + str(temp_file_path)
                + ':9:9: '
                + 'E999 IndentationError: unexpected indent\n'
                '[This is how your edit would have looked if applied]\n'
                + SEP
                + '(this is the beginning of the file)\n'
                '1|def any_int(a, b, c):\n'
                '2|    return isinstance(a, int) and isinstance(b, int) and isinstance(c, int)\n'
                '3|\n'
                '4|def test_any_int():\n'
                '5|    assert any_int(1, 2, 3) == True\n'
                '6|    assert any_int(1.5, 2, 3) == False\n'
                '7|    assert any_int(1, 2.5, 3) == False\n'
                '8|    assert any_int(1, 2, 3.5) == False\n'
                '9|        assert any_int(1.0, 2, 3) == False\n'
                '10|    assert any_int(1, 2.0, 3) == False\n'
                '11|    assert any_int(1, 2, 3.0) == False\n'
                '12|    assert any_int(0, 0, 0) == True\n'
                '13|    assert any_int(-1, -2, -3) == True\n'
                '14|    assert any_int(1, -2, 3) == True\n'
                '15|    assert any_int(1.5, -2, 3) == False\n'
                '16|    assert any_int(1, -2.5, 3) == False\n'
                '17|\n'
                '18|def check(any_int):\n'
                '19|    # Check some simple cases\n'
                '20|    assert any_int(2, 3, 1)==True, "This prints if this assert fails 1 (good for debugging!)"\n'
                '21|    assert any_int(2.5, 2, 3)==False, "This prints if this assert fails 2 (good for debugging!)"\n'
                '(12 more lines below)\n' + SEP + '\n'
                '[This is the original code before your edit]\n'
                + SEP
                + '(this is the beginning of the file)\n'
                '1|def any_int(a, b, c):\n'
                '2|    return isinstance(a, int) and isinstance(b, int) and isinstance(c, int)\n'
                '3|\n'
                '4|def test_any_int():\n'
                '5|    assert any_int(1, 2, 3) == True\n'
                '6|    assert any_int(1.5, 2, 3) == False\n'
                '7|    assert any_int(1, 2.5, 3) == False\n'
                '8|    assert any_int(1, 2, 3.5) == False\n'
                '9|    assert any_int(1.0, 2, 3) == False\n'
                '10|    assert any_int(1, 2.0, 3) == False\n'
                '11|    assert any_int(1, 2, 3.0) == False\n'
                '12|    assert any_int(0, 0, 0) == True\n'
                '13|    assert any_int(-1, -2, -3) == True\n'
                '14|    assert any_int(1, -2, 3) == True\n'
                '15|    assert any_int(1.5, -2, 3) == False\n'
                '16|    assert any_int(1, -2.5, 3) == False\n'
                '17|\n'
                '18|def check(any_int):\n'
                '19|    # Check some simple cases\n'
                '20|    assert any_int(2, 3, 1)==True, "This prints if this assert fails 1 (good for debugging!)"\n'
                '21|    assert any_int(2.5, 2, 3)==False, "This prints if this assert fails 2 (good for debugging!)"\n'
                '(12 more lines below)\n'
                + SEP
                + 'Your changes have NOT been applied. Please fix your edit command and try again.\n'
                'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.\n'
                'DO NOT re-run the same failed edit command. Running it again will lead to the same error.\n'
            )
            assert result == expected


# ================================


def test_edit_file_by_replace(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            edit_file_by_replace(
                file_name=str(temp_file_path),
                to_replace='Line 1\nLine 2\nLine 3',
                new_content='REPLACE TEXT',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|REPLACE TEXT\n'
            '2|Line 4\n'
            '3|Line 5\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == 'REPLACE TEXT'
    assert lines[1].rstrip() == 'Line 4'
    assert lines[2].rstrip() == 'Line 5'


def test_edit_file_by_replace_sameline(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2\nLine 2\nLine 4\nLine 5'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            edit_file_by_replace(
                file_name=str(temp_file_path),
                to_replace='Line 2\nLine 2',
                new_content='Line 2\nREPLACE TEXT',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (5 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|Line 1\n'
            '2|Line 2\n'
            '3|REPLACE TEXT\n'
            '4|Line 4\n'
            '5|Line 5\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=2)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 5
    assert lines[0].rstrip() == 'Line 1'
    assert lines[1].rstrip() == 'Line 2'
    assert lines[2].rstrip() == 'REPLACE TEXT'
    assert lines[3].rstrip() == 'Line 4'
    assert lines[4].rstrip() == 'Line 5'


def test_edit_file_by_replace_multiline(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2\nLine 2\nLine 4\nLine 5'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            edit_file_by_replace(
                file_name=str(temp_file_path),
                to_replace='Line 2',
                new_content='REPLACE TEXT',
            )
            result = buf.getvalue()
            assert result.strip().startswith(
                'ERROR: `to_replace` appears more than once, please include enough lines to make code in `to_replace` unique'
            )


def test_edit_file_by_replace_no_diff(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2\nLine 2\nLine 4\nLine 5'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            edit_file_by_replace(
                file_name=str(temp_file_path),
                to_replace='Line 1',
                new_content='Line 1',
            )
            result = buf.getvalue()
            assert result.strip().startswith(
                'ERROR: `to_replace` and `new_content` must be different'
            )


def test_edit_file_by_replace_toreplace_empty(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2\nLine 2\nLine 4\nLine 5'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    _capture_file_operation_error(
        lambda: edit_file_by_replace(
            file_name=str(temp_file_path),
            to_replace='',
            new_content='Line 1',
        ),
        'ERROR: `to_replace` must not be empty.',
    )


def test_edit_file_by_replace_unknown_file():
    _capture_file_operation_error(
        lambda: edit_file_by_replace(
            str('unknown file'),
            'ORIGINAL TEXT',
            'REPLACE TEXT',
        ),
        'ERROR: File unknown file not found.',
    )


def test_insert_content_at_line(tmp_path):
    temp_file_path = tmp_path / 'b.txt'
    content = 'Line 1\nLine 2\nLine 3'
    temp_file_path.write_text(content)
    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                file_name=str(temp_file_path),
                line_number=2,
                content='Inserted Line',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (4 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|Line 1\n'
            '2|Inserted Line\n'
            '3|Line 2\n'
            '4|Line 3\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=2)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 4
    assert lines[0].rstrip() == 'Line 1'
    assert lines[1].rstrip() == 'Inserted Line'
    assert lines[2].rstrip() == 'Line 2'
    assert lines[3].rstrip() == 'Line 3'


def test_insert_content_at_line_from_scratch(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    create_file(str(temp_file_path))
    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                file_name=str(temp_file_path),
                line_number=1,
                content='REPLACE TEXT',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (1 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|REPLACE TEXT\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 1
    assert lines[0].rstrip() == 'REPLACE TEXT'


def test_insert_content_at_line_from_scratch_emptyfile(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    with open(temp_file_path, 'w') as file:
        file.write('')
    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                file_name=str(temp_file_path),
                line_number=1,
                content='REPLACE TEXT',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (1 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|REPLACE TEXT\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 1
    assert lines[0].rstrip() == 'REPLACE TEXT'


def test_insert_content_at_line_emptyline(tmp_path):
    temp_file_path = tmp_path / 'b.txt'
    content = 'Line 1\n\n'
    temp_file_path.write_text(content)
    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                file_name=str(temp_file_path),
                line_number=2,
                content='Inserted Line',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|Line 1\n'
            '2|Inserted Line\n'
            '3|\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=2)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == 'Line 1'
    assert lines[1].rstrip() == 'Inserted Line'


def test_insert_content_at_line_from_scratch_multiline_with_backticks_and_second_edit(
    tmp_path,
):
    temp_file_path = tmp_path / 'a.txt'
    create_file(str(temp_file_path))
    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                str(temp_file_path),
                1,
                '`REPLACE TEXT1`\n`REPLACE TEXT2`\n`REPLACE TEXT3`',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|`REPLACE TEXT1`\n'
            '2|`REPLACE TEXT2`\n'
            '3|`REPLACE TEXT3`\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == '`REPLACE TEXT1`'
    assert lines[1].rstrip() == '`REPLACE TEXT2`'
    assert lines[2].rstrip() == '`REPLACE TEXT3`'

    # Check that no backticks are escaped in the edit_file_by_replace call
    assert '\\`' not in result

    # Perform a second edit
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            edit_file_by_replace(
                str(temp_file_path),
                '`REPLACE TEXT1`\n`REPLACE TEXT2`\n`REPLACE TEXT3`',
                '`REPLACED TEXT1`\n`REPLACED TEXT2`\n`REPLACED TEXT3`',
            )
        second_result = buf.getvalue()
        second_expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|`REPLACED TEXT1`\n'
            '2|`REPLACED TEXT2`\n'
            '3|`REPLACED TEXT3`\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert second_result.split('\n') == second_expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == '`REPLACED TEXT1`'
    assert lines[1].rstrip() == '`REPLACED TEXT2`'
    assert lines[2].rstrip() == '`REPLACED TEXT3`'

    # Check that no backticks are escaped in the second edit_file_by_replace call
    assert '\\`' not in second_result


def test_insert_content_at_line_from_scratch_multiline(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    create_file(str(temp_file_path))
    open_file(temp_file_path)

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            insert_content_at_line(
                str(temp_file_path),
                1,
                content='REPLACE TEXT1\nREPLACE TEXT2\nREPLACE TEXT3',
            )
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|REPLACE TEXT1\n'
            '2|REPLACE TEXT2\n'
            '3|REPLACE TEXT3\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == 'REPLACE TEXT1'
    assert lines[1].rstrip() == 'REPLACE TEXT2'
    assert lines[2].rstrip() == 'REPLACE TEXT3'


def test_insert_content_at_line_not_opened():
    _capture_file_operation_error(
        lambda: insert_content_at_line(
            str('unknown file'),
            1,
            'REPLACE TEXT',
        ),
        'ERROR: Invalid path or file name.',
    )


def test_append_file(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    content = 'Line 1\nLine 2'
    temp_file_path.write_text(content)

    open_file(str(temp_file_path))

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            append_file(str(temp_file_path), content='APPENDED TEXT')
        result = buf.getvalue()
        expected = (
            f'[File: {temp_file_path} (3 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|Line 1\n'
            '2|Line 2\n'
            '3|APPENDED TEXT\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=2)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')

    with open(temp_file_path, 'r') as file:
        lines = file.readlines()
    assert len(lines) == 3
    assert lines[0].rstrip() == 'Line 1'
    assert lines[1].rstrip() == 'Line 2'
    assert lines[2].rstrip() == 'APPENDED TEXT'


def test_append_file_from_scratch(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    create_file(str(temp_file_path))
    try:
        open_file(str(temp_file_path))
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                append_file(str(temp_file_path), content='APPENDED TEXT')
            result = buf.getvalue()
            expected = (
                f'[File: {temp_file_path} (1 lines total after edit)]\n'
                '(this is the beginning of the file)\n'
                '1|APPENDED TEXT\n'
                '(this is the end of the file)\n'
                + MSG_FILE_UPDATED.format(line_number=1)
                + '\n'
            )
            assert result.split('\n') == expected.split('\n')

        with open(temp_file_path, 'r') as file:
            lines = file.readlines()
        assert len(lines) == 1
        assert lines[0].rstrip() == 'APPENDED TEXT'
    finally:
        os.remove(temp_file_path)


def test_append_file_from_scratch_multiline(tmp_path):
    temp_file_path = tmp_path / 'a3.txt'
    create_file(str(temp_file_path))
    try:
        open_file(temp_file_path)
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                append_file(
                    str(temp_file_path),
                    content='APPENDED TEXT1\nAPPENDED TEXT2\nAPPENDED TEXT3',
                )
            result = buf.getvalue()
            expected = (
                f'[File: {temp_file_path} (3 lines total after edit)]\n'
                '(this is the beginning of the file)\n'
                '1|APPENDED TEXT1\n'
                '2|APPENDED TEXT2\n'
                '3|APPENDED TEXT3\n'
                '(this is the end of the file)\n'
                + MSG_FILE_UPDATED.format(line_number=1)
                + '\n'
            )
            assert result.split('\n') == expected.split('\n')

        with open(temp_file_path, 'r') as file:
            lines = file.readlines()
        assert len(lines) == 3
        assert lines[0].rstrip() == 'APPENDED TEXT1'
        assert lines[1].rstrip() == 'APPENDED TEXT2'
        assert lines[2].rstrip() == 'APPENDED TEXT3'
    finally:
        os.remove(temp_file_path)


def test_append_file_not_opened():
    _capture_file_operation_error(
        lambda: append_file('unknown file', content='APPENDED TEXT'),
        'ERROR: Invalid path or file name.',
    )


def test_search_dir(tmp_path):
    # create files with the search term "bingo"
    for i in range(1, 101):
        temp_file_path = tmp_path / f'a{i}.txt'
        with open(temp_file_path, 'w') as file:
            file.write('Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n')
            if i == 50:
                file.write('bingo')

    # test
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_dir('bingo', str(tmp_path))
        result = buf.getvalue()
    assert result is not None

    expected = (
        f'[Found 1 matches for "bingo" in {tmp_path}]\n'
        f'{tmp_path}/a50.txt (Line 6): bingo\n'
        f'[End of matches for "bingo" in {tmp_path}]\n'
    )
    assert result.split('\n') == expected.split('\n')


def test_search_dir_not_exist_term(tmp_path):
    # create files with the search term "bingo"
    for i in range(1, 101):
        temp_file_path = tmp_path / f'a{i}.txt'
        with open(temp_file_path, 'w') as file:
            file.write('Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n')

    # test
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_dir('non-exist', str(tmp_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'No matches found for "non-exist" in {tmp_path}\n'
    assert result.split('\n') == expected.split('\n')


def test_search_dir_too_much_match(tmp_path):
    # create files with the search term "Line 5"
    for i in range(1, 1000):
        temp_file_path = tmp_path / f'a{i}.txt'
        with open(temp_file_path, 'w') as file:
            file.write('Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_dir('Line 5', str(tmp_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'More than 999 files matched for "Line 5" in {tmp_path}. Please narrow your search.\n'
    assert result.split('\n') == expected.split('\n')


def test_search_dir_cwd(tmp_path, monkeypatch):
    # Using pytest's monkeypatch to change directory without affecting other tests
    monkeypatch.chdir(tmp_path)
    # create files with the search term "bingo"
    for i in range(1, 101):
        temp_file_path = tmp_path / f'a{i}.txt'
        with open(temp_file_path, 'w') as file:
            file.write('Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n')
            if i == 50:
                file.write('bingo')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_dir('bingo')
        result = buf.getvalue()
    assert result is not None

    expected = (
        '[Found 1 matches for "bingo" in ./]\n'
        './a50.txt (Line 6): bingo\n'
        '[End of matches for "bingo" in ./]\n'
    )
    assert result.split('\n') == expected.split('\n')


def test_search_file(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_file('Line 5', str(temp_file_path))
        result = buf.getvalue()
    assert result is not None
    expected = f'[Found 1 matches for "Line 5" in {temp_file_path}]\n'
    expected += 'Line 5: Line 5\n'
    expected += f'[End of matches for "Line 5" in {temp_file_path}]\n'
    assert result.split('\n') == expected.split('\n')


def test_search_file_not_exist_term(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            search_file('Line 6', str(temp_file_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[No matches found for "Line 6" in {temp_file_path}]\n'
    assert result.split('\n') == expected.split('\n')


def test_search_file_not_exist_file():
    _capture_file_operation_error(
        lambda: search_file('Line 6', '/unexist/path/a.txt'),
        'ERROR: File /unexist/path/a.txt not found.',
    )


def test_find_file(tmp_path):
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            find_file('a.txt', str(tmp_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[Found 1 matches for "a.txt" in {tmp_path}]\n'
    expected += f'{tmp_path}/a.txt\n'
    expected += f'[End of matches for "a.txt" in {tmp_path}]\n'
    assert result.split('\n') == expected.split('\n')


def test_find_file_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    temp_file_path = tmp_path / 'a.txt'
    temp_file_path.write_text('Line 1\nLine 2\nLine 3\nLine 4\nLine 5')

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            find_file('a.txt')
        result = buf.getvalue()
    assert result is not None


def test_find_file_not_exist_file():
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            find_file('nonexist.txt')
        result = buf.getvalue()
    assert result is not None

    expected = '[No matches found for "nonexist.txt" in ./]\n'
    assert result.split('\n') == expected.split('\n')


def test_find_file_not_exist_file_specific_path(tmp_path):
    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            find_file('nonexist.txt', str(tmp_path))
        result = buf.getvalue()
    assert result is not None

    expected = f'[No matches found for "nonexist.txt" in {tmp_path}]\n'
    assert result.split('\n') == expected.split('\n')


def test_edit_lint_file_pass(tmp_path):
    # Enable linting
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        file_path = _generate_test_file_with_lines(tmp_path, 1)

        # Test linting functionality
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                open_file(str(file_path))
                insert_content_at_line(str(file_path), 1, "print('hello')\n")
            result = buf.getvalue()
        assert result is not None
        expected = (
            f'[File: {file_path} (1 lines total)]\n'
            '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            f'[File: {file_path} (1 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            "1|print('hello')\n"
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')


def test_lint_file_fail_undefined_name(tmp_path, capsys):
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        current_line = 1

        file_path = _generate_test_file_with_lines(tmp_path, 1)

        open_file(str(file_path), current_line)
        insert_content_at_line(str(file_path), 1, 'undefined_name()\n')

        result = capsys.readouterr().out
        assert result is not None

        expected = (
            f'[File: {file_path} (1 lines total)]\n'
            '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
            'ERRORS:\n'
            f"{file_path}:1:1: F821 undefined name 'undefined_name'\n"
            '[This is how your edit would have looked if applied]\n'
            + SEP
            + '(this is the beginning of the file)\n'
            '1|undefined_name()\n'
            '(this is the end of the file)\n'
            + SEP
            + '\n[This is the original code before your edit]\n'
            + SEP
            + '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            + SEP
            + 'Your changes have NOT been applied. Please fix your edit command and try again.\n'
            'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.\n'
            'DO NOT re-run the same failed edit command. Running it again will lead to the same error.\n'
        )
        assert result.split('\n') == expected.split('\n')


def test_lint_file_fail_undefined_name_long(tmp_path, capsys):
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        num_lines = 1000
        error_line = 500

        file_path = _generate_test_file_with_lines(tmp_path, num_lines)

        error_message = (
            f"{file_path}:{error_line}:1: F821 undefined name 'undefined_name'"
        )

        open_file(str(file_path))
        insert_content_at_line(str(file_path), error_line, 'undefined_name()\n')

        result = capsys.readouterr().out
        assert result is not None

        open_lines = '\n'.join([f'{i}|' for i in range(1, WINDOW + 1)])
        expected = (
            f'[File: {file_path} ({num_lines} lines total)]\n'
            '(this is the beginning of the file)\n'
            f'{open_lines}\n'
            f'({num_lines - WINDOW} more lines below)\n'
            f'[Use `scroll_down` to view the next 100 lines of the file!]\n'
            '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
            f'ERRORS:\n{error_message}\n'
            '[This is how your edit would have looked if applied]\n'
            + SEP
            + '(489 more lines above)\n'
            + _numbered_test_lines(error_line - 10, error_line - 1)
            + '500|undefined_name()\n'
            + _numbered_test_lines(error_line + 1, error_line + 10)
            + '(491 more lines below)\n'
            + SEP
            + '\n[This is the original code before your edit]\n'
            + SEP
            + '(489 more lines above)\n'
            + _numbered_test_lines(error_line - 10, error_line + 10)
            + '(490 more lines below)\n'
            + SEP
            + 'Your changes have NOT been applied. Please fix your edit command and try again.\n'
            'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.\n'
            'DO NOT re-run the same failed edit command. Running it again will lead to the same error.\n'
        )
        assert result.split('\n') == expected.split('\n')


def test_lint_file_disabled_undefined_name(tmp_path, capsys):
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'False'}):
        file_path = _generate_test_file_with_lines(tmp_path, 1)

        open_file(str(file_path))
        insert_content_at_line(str(file_path), 1, 'undefined_name()\n')

        result = capsys.readouterr().out
        assert result is not None
        expected = (
            f'[File: {file_path} (1 lines total)]\n'
            '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            f'[File: {file_path} (1 lines total after edit)]\n'
            '(this is the beginning of the file)\n'
            '1|undefined_name()\n'
            '(this is the end of the file)\n'
            + MSG_FILE_UPDATED.format(line_number=1)
            + '\n'
        )
        assert result.split('\n') == expected.split('\n')


def test_lint_file_ignore_existing_errors(tmp_path):
    """
    Make sure we allow edits as long as it does not introduce new errors. In other
    words, we don't care about existing linting errors. Although they might be
    real syntax issues, sometimes they are just false positives, or errors that
    we don't care about.
    """
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        content = """def some_valid_but_weird_function():
    # this function is legitimate, yet static analysis tools like flake8
    # reports 'F821 undefined name'
    if 'variable' in locals():
        print(variable)


def some_wrong_but_unused_function():
    # this function has a linting error, but it is not modified by us, and
    # who knows, this function might be completely dead code
    x = 1


def sum(a, b):
    return a - b
"""

        temp_file_path = tmp_path / 'problematic-file-test.py'
        temp_file_path.write_text(content)

        open_file(str(temp_file_path))

        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                edit_file_by_replace(
                    str(temp_file_path),
                    to_replace='    return a - b',
                    new_content='    return a + b',
                )
            result = buf.getvalue()
            expected = (
                f'[File: {temp_file_path} (15 lines total after edit)]\n'
                """(this is the beginning of the file)
1|def some_valid_but_weird_function():
2|    # this function is legitimate, yet static analysis tools like flake8
3|    # reports 'F821 undefined name'
4|    if 'variable' in locals():
5|        print(variable)
6|
7|
8|def some_wrong_but_unused_function():
9|    # this function has a linting error, but it is not modified by us, and
10|    # who knows, this function might be completely dead code
11|    x = 1
12|
13|
14|def sum(a, b):
15|    return a + b
(this is the end of the file)
""" + MSG_FILE_UPDATED.format(line_number=15) + '\n'
            )
            assert result.split('\n') == expected.split('\n')


def test_lint_file_catch_new_errors_in_edits(tmp_path):
    """
    Make sure we catch new linting errors in our edit chunk, and at the same
    time, ignore old linting errors (in this case, the old linting error is
    a false positive)
    """
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        content = """def some_valid_but_weird_function():
    # this function is legitimate, yet static analysis tools like flake8
    # reports 'F821 undefined name'
    if 'variable' in locals():
        print(variable)


def sum(a, b):
    return a - b
"""

        temp_file_path = tmp_path / 'problematic-file-test.py'
        temp_file_path.write_text(content)

        open_file(str(temp_file_path))

        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                edit_file_by_replace(
                    str(temp_file_path),
                    to_replace='    return a - b',
                    # we deliberately make a mistake here (LLMs can make mistakes like this)
                    # note 'variable' is an undefined name
                    new_content='    return a + variable',
                )
            result = buf.getvalue()

            expected = (
                '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
                'ERRORS:\n'
                f"{temp_file_path}:9:16: F821 undefined name 'variable'\n"
                """[This is how your edit would have looked if applied]
-------------------------------------------------
(this is the beginning of the file)
1|def some_valid_but_weird_function():
2|    # this function is legitimate, yet static analysis tools like flake8
3|    # reports 'F821 undefined name'
4|    if 'variable' in locals():
5|        print(variable)
6|
7|
8|def sum(a, b):
9|    return a + variable
(this is the end of the file)
-------------------------------------------------

[This is the original code before your edit]
-------------------------------------------------
(this is the beginning of the file)
1|def some_valid_but_weird_function():
2|    # this function is legitimate, yet static analysis tools like flake8
3|    # reports 'F821 undefined name'
4|    if 'variable' in locals():
5|        print(variable)
6|
7|
8|def sum(a, b):
9|    return a - b
(this is the end of the file)
-------------------------------------------------
Your changes have NOT been applied. Please fix your edit command and try again.
You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.
DO NOT re-run the same failed edit command. Running it again will lead to the same error.
"""
            )
            assert result.split('\n') == expected.split('\n')


def test_lint_file_catch_new_errors_outside_edits(tmp_path):
    """
    Make sure we catch new linting errors induced by our edits, even
    though the error itself is not in the edit chunk
    """
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        content = """def valid_func1():
    print(my_sum(1, 2))


def my_sum(a, b):
    return a - b


def valid_func2():
    print(my_sum(0, 0))


"""

        # Add 100 lines of invalid code, which linter shall ignore
        # because they are not being edited. For testing purpose, we
        # must add these existing linting errors, otherwise the pre-edit
        # linting would pass, and thus there won't be any comparison
        # between pre-edit and post-edit linting.
        for _ in range(100):
            content += '\ninvalid_func()'

        temp_file_path = tmp_path / 'problematic-file-test.py'
        temp_file_path.write_text(content)

        open_file(str(temp_file_path))

        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                edit_file_by_replace(
                    str(temp_file_path),
                    to_replace='def my_sum(a, b):',
                    # we deliberately make a mistake here (LLMs can make mistakes like this)
                    # by changing the function name, existing function calls would become invalid
                    new_content='def my_sum2(a, b):',
                )
            result = buf.getvalue()

            expected = (
                '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
                'ERRORS:\n'
                f"{temp_file_path}:2:11: F821 undefined name 'my_sum'\n"
                f"{temp_file_path}:10:11: F821 undefined name 'my_sum'\n"
                """[This is how your edit would have looked if applied]
-------------------------------------------------
(this is the beginning of the file)
1|def valid_func1():
2|    print(my_sum(1, 2))
3|
4|
5|def my_sum2(a, b):
6|    return a - b
7|
8|
9|def valid_func2():
10|    print(my_sum(0, 0))
11|
12|
13|
14|invalid_func()
15|invalid_func()
16|invalid_func()
17|invalid_func()
18|invalid_func()
19|invalid_func()
20|invalid_func()
21|invalid_func()
(92 more lines below)
-------------------------------------------------

[This is the original code before your edit]
-------------------------------------------------
(this is the beginning of the file)
1|def valid_func1():
2|    print(my_sum(1, 2))
3|
4|
5|def my_sum(a, b):
6|    return a - b
7|
8|
9|def valid_func2():
10|    print(my_sum(0, 0))
11|
12|
13|
14|invalid_func()
15|invalid_func()
16|invalid_func()
17|invalid_func()
18|invalid_func()
19|invalid_func()
20|invalid_func()
21|invalid_func()
(92 more lines below)
-------------------------------------------------
Your changes have NOT been applied. Please fix your edit command and try again.
You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.
DO NOT re-run the same failed edit command. Running it again will lead to the same error.
"""
            )
            assert result.split('\n') == expected.split('\n')


def test_parse_docx(tmp_path):
    # Create a DOCX file with some content
    test_docx_path = tmp_path / 'test.docx'
    doc = docx.Document()
    doc.add_paragraph('Hello, this is a test document.')
    doc.add_paragraph('This is the second paragraph.')
    doc.save(str(test_docx_path))

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the parse_docx function
    parse_docx(str(test_docx_path))

    # Capture the output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Check if the output is correct
    expected_output = (
        f'[Reading DOCX file from {test_docx_path}]\n'
        '@@ Page 1 @@\nHello, this is a test document.\n\n'
        '@@ Page 2 @@\nThis is the second paragraph.\n\n\n'
    )
    assert output == expected_output, f'Expected output does not match. Got: {output}'


def test_parse_latex(tmp_path):
    # Create a LaTeX file with some content
    test_latex_path = tmp_path / 'test.tex'
    with open(test_latex_path, 'w') as f:
        f.write(r"""
        \documentclass{article}
        \begin{document}
        Hello, this is a test LaTeX document.
        \end{document}
        """)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the parse_latex function
    parse_latex(str(test_latex_path))

    # Capture the output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Check if the output is correct
    expected_output = (
        f'[Reading LaTex file from {test_latex_path}]\n'
        'Hello, this is a test LaTeX document.\n'
    )
    assert output == expected_output, f'Expected output does not match. Got: {output}'


def test_parse_pdf(tmp_path):
    # Create a PDF file with some content
    test_pdf_path = tmp_path / 'test.pdf'
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(test_pdf_path), pagesize=letter)
    c.drawString(100, 750, 'Hello, this is a test PDF document.')
    c.save()

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Call the parse_pdf function
    parse_pdf(str(test_pdf_path))

    # Capture the output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Check if the output is correct
    expected_output = (
        f'[Reading PDF file from {test_pdf_path}]\n'
        '@@ Page 1 @@\n'
        'Hello, this is a test PDF document.\n'
    )
    assert output == expected_output, f'Expected output does not match. Got: {output}'


def test_parse_pptx(tmp_path):
    test_pptx_path = tmp_path / 'test.pptx'
    from pptx import Presentation

    pres = Presentation()

    slide1 = pres.slides.add_slide(pres.slide_layouts[0])
    title1 = slide1.shapes.title
    title1.text = 'Hello, this is the first test PPTX slide.'

    slide2 = pres.slides.add_slide(pres.slide_layouts[0])
    title2 = slide2.shapes.title
    title2.text = 'Hello, this is the second test PPTX slide.'

    pres.save(str(test_pptx_path))

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    parse_pptx(str(test_pptx_path))

    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    expected_output = (
        f'[Reading PowerPoint file from {test_pptx_path}]\n'
        '@@ Slide 1 @@\n'
        'Hello, this is the first test PPTX slide.\n\n'
        '@@ Slide 2 @@\n'
        'Hello, this is the second test PPTX slide.\n\n'
    )
    assert output == expected_output, f'Expected output does not match. Got: {output}'


def test_lint_file_fail_non_python(tmp_path, capsys):
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        current_line = 1
        file_path = _generate_ruby_test_file_with_lines(tmp_path, 1)

        open_file(str(file_path), current_line)
        insert_content_at_line(
            str(file_path), 1, "def print_hello_world()\n    puts 'Hello World'"
        )
        result = capsys.readouterr().out
        assert result is not None
        expected = (
            f'[File: {file_path} (1 lines total)]\n'
            '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]\n'
            'ERRORS:\n'
            f'{file_path}:1:1: Syntax error\n'
            '[This is how your edit would have looked if applied]\n'
            + SEP
            + '(this is the beginning of the file)\n'
            '1|def print_hello_world()\n'
            "2|    puts 'Hello World'\n"
            '(this is the end of the file)\n'
            '-------------------------------------------------\n\n'
            '[This is the original code before your edit]\n'
            + SEP
            + '(this is the beginning of the file)\n'
            '1|\n'
            '(this is the end of the file)\n'
            + SEP
            + 'Your changes have NOT been applied. Please fix your edit command and try again.\n'
            'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.\n'
            'DO NOT re-run the same failed edit command. Running it again will lead to the same error.\n'
        )
        assert result.split('\n') == expected.split('\n')


def test_lint_file_fail_typescript(tmp_path, capsys):
    linter = Linter()
    with patch.dict(os.environ, {'ENABLE_AUTO_LINT': 'True'}):
        current_line = 1
        file_path = tmp_path / 'test.ts'
        file_path.write_text('')

        open_file(str(file_path), current_line)
        insert_content_at_line(
            str(file_path),
            1,
            "function greet(name: string) {\n    console.log('Hello, ' + name)",
        )
        result = capsys.readouterr().out
        assert result is not None

        # Note: the tsc (typescript compiler) message is different from a
        # compared to a python linter message, like line and column in brackets:
        expected_lines = [
            f'[File: {file_path} (1 lines total)]',
            '(this is the beginning of the file)',
            '1|',
            '(this is the end of the file)',
            '[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]',
            'ERRORS:',
            f"{file_path}(3,1): error TS1005: '}}' expected.",
            '[This is how your edit would have looked if applied]',
            '-------------------------------------------------',
            '(this is the beginning of the file)',
            '1|function greet(name: string) {',
            "2|    console.log('Hello, ' + name)",
            '(this is the end of the file)',
            '-------------------------------------------------',
            '',
            '[This is the original code before your edit]',
            '-------------------------------------------------',
            '(this is the beginning of the file)',
            '1|',
            '(this is the end of the file)',
            '-------------------------------------------------',
            'Your changes have NOT been applied. Please fix your edit command and try again.',
            'You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.',
            'DO NOT re-run the same failed edit command. Running it again will lead to the same error.',
            '',
        ]
        result_lines = result.split('\n')
        assert len(result_lines) == len(expected_lines), "Number of lines doesn't match"

        for i, (result_line, expected_line) in enumerate(
            zip(result_lines, expected_lines)
        ):
            if i == 6:
                if linter.ts_installed and result_line != expected_lines[6]:
                    assert (
                        'ts:1:20:' in result_line or '(3,1):' in result_line
                    ), f"Line {i+1} doesn't match"
            else:
                assert result_line.lstrip('./') == expected_line.lstrip(
                    './'
                ), f"Line {i+1} doesn't match"
