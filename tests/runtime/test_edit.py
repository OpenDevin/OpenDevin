"""Edit-related tests for the EventStreamRuntime."""

import os

import pytest
from conftest import (
    TEST_IN_CI,
    _close_test_runtime,
    _load_runtime,
)

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import FileEditAction, FileReadAction
from openhands.events.observation import FileEditObservation
from openhands.utils.diff import get_diff

ORGINAL = """from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    numbers = list(range(1, 11))
    return str(numbers)

if __name__ == '__main__':
    app.run(port=5000)
"""


@pytest.mark.skipif(
    TEST_IN_CI != 'True',
    reason='This test requires LLM to run.',
)
def test_edit_from_scratch(temp_dir, box_class, run_as_openhands):
    runtime = _load_runtime(temp_dir, box_class, run_as_openhands)
    try:
        action = FileEditAction(
            content=ORGINAL,
            start=-1,
            path=os.path.join('/workspace', 'app.py'),
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(
            obs, FileEditObservation
        ), 'The observation should be a FileEditObservation.'

        action = FileReadAction(
            path=os.path.join('/workspace', 'app.py'),
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.content.strip() == ORGINAL.strip()

    finally:
        _close_test_runtime(runtime)


EDIT = """# above stays the same
@app.route('/')
def index():
    numbers = list(range(1, 11))
    return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
# below stays the same
"""


@pytest.mark.skipif(
    TEST_IN_CI != 'True',
    reason='This test requires LLM to run.',
)
def test_edit(temp_dir, box_class, run_as_openhands):
    runtime = _load_runtime(temp_dir, box_class, run_as_openhands)
    try:
        action = FileEditAction(
            content=ORGINAL,
            path=os.path.join('/workspace', 'app.py'),
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(
            obs, FileEditObservation
        ), 'The observation should be a FileEditObservation.'

        action = FileReadAction(
            path=os.path.join('/workspace', 'app.py'),
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.content.strip() == ORGINAL.strip()

        action = FileEditAction(
            content=EDIT,
            path=os.path.join('/workspace', 'app.py'),
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.content.strip()
            == (
                '--- /workspace/app.py\n'
                '+++ /workspace/app.py\n'
                '@@ -4,7 +4,7 @@\n'
                " @app.route('/')\n"
                ' def index():\n'
                '     numbers = list(range(1, 11))\n'
                '-    return str(numbers)\n'
                "+    return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'\n"
                '\n'
                " if __name__ == '__main__':\n"
                '     app.run(port=5000)\n'
            ).strip()
        )
    finally:
        _close_test_runtime(runtime)


ORIGINAL_LONG = '\n'.join([f'This is line {i}' for i in range(1, 1000)])
EDIT_LONG = """
This is line 100 + 10
This is line 101 + 10
"""


def test_edit_long_file(temp_dir, box_class, run_as_openhands):
    runtime = _load_runtime(temp_dir, box_class, run_as_openhands)
    try:
        action = FileEditAction(
            content=ORIGINAL_LONG,
            path=os.path.join('/workspace', 'app.py'),
            start=-1,
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(
            obs, FileEditObservation
        ), 'The observation should be a FileEditObservation.'

        action = FileReadAction(
            path=os.path.join('/workspace', 'app.py'),
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.content.strip() == ORIGINAL_LONG.strip()

        action = FileEditAction(
            content=EDIT_LONG,
            path=os.path.join('/workspace', 'app.py'),
            start=100,
            end=200,
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert (
            obs.content.strip()
            == (
                '--- /workspace/app.py\n'
                '+++ /workspace/app.py\n'
                '@@ -97,8 +97,8 @@\n'
                ' This is line 97\n'
                ' This is line 98\n'
                ' This is line 99\n'
                '-This is line 100\n'
                '-This is line 101\n'
                '+This is line 100 + 10\n'
                '+This is line 101 + 10\n'
                ' This is line 102\n'
                ' This is line 103\n'
                ' This is line 104\n'
            ).strip()
        )
    finally:
        _close_test_runtime(runtime)


# ======================================================================================
# Test FileEditObservation (things that are displayed to the agent)
# ======================================================================================


def test_edit_obs_insert_only():
    EDIT_LONG_INSERT_ONLY = (
        '\n'.join([f'This is line {i}' for i in range(1, 100)])
        + EDIT_LONG
        + '\n'.join([f'This is line {i}' for i in range(100, 1000)])
    )

    diff = get_diff(ORIGINAL_LONG, EDIT_LONG_INSERT_ONLY, '/workspace/app.py')
    obs = FileEditObservation(
        content=diff,
        path='/workspace/app.py',
        prev_exist=True,
        old_content=ORIGINAL_LONG,
        new_content=EDIT_LONG_INSERT_ONLY,
    )
    assert (
        str(obs).strip()
        == """
[Existing file /workspace/app.py is edited with 1 changes.]
[begin of edit 1 / 1]
(content before edits)
  98|This is line 98
  99|This is line 99
 100|This is line 100
 101|This is line 101
(content after edits)
  98|This is line 98
  99|This is line 99
+100|This is line 100 + 10
+101|This is line 101 + 10
 102|This is line 100
 103|This is line 101
[end of edit 1 / 1]
""".strip()
    )


def test_edit_obs_replace():
    _new_content = (
        '\n'.join([f'This is line {i}' for i in range(1, 100)])
        + EDIT_LONG
        + '\n'.join([f'This is line {i}' for i in range(102, 1000)])
    )

    diff = get_diff(ORIGINAL_LONG, _new_content, '/workspace/app.py')
    obs = FileEditObservation(
        content=diff,
        path='/workspace/app.py',
        prev_exist=True,
        old_content=ORIGINAL_LONG,
        new_content=_new_content,
    )
    print(str(obs))
    assert (
        str(obs).strip()
        == """
[Existing file /workspace/app.py is edited with 1 changes.]
[begin of edit 1 / 1]
(content before edits)
  98|This is line 98
  99|This is line 99
-100|This is line 100
-101|This is line 101
 102|This is line 102
 103|This is line 103
(content after edits)
  98|This is line 98
  99|This is line 99
+100|This is line 100 + 10
+101|This is line 101 + 10
 102|This is line 102
 103|This is line 103
[end of edit 1 / 1]
""".strip()
    )


def test_edit_obs_replace_with_empty_line():
    _new_content = (
        '\n'.join([f'This is line {i}' for i in range(1, 100)])
        + '\n'
        + EDIT_LONG
        + '\n'.join([f'This is line {i}' for i in range(102, 1000)])
    )

    diff = get_diff(ORIGINAL_LONG, _new_content, '/workspace/app.py')
    obs = FileEditObservation(
        content=diff,
        path='/workspace/app.py',
        prev_exist=True,
        old_content=ORIGINAL_LONG,
        new_content=_new_content,
    )
    print(str(obs))
    assert (
        str(obs).strip()
        == """
[Existing file /workspace/app.py is edited with 1 changes.]
[begin of edit 1 / 1]
(content before edits)
  98|This is line 98
  99|This is line 99
-100|This is line 100
-101|This is line 101
 102|This is line 102
 103|This is line 103
(content after edits)
  98|This is line 98
  99|This is line 99
+100|
+101|This is line 100 + 10
+102|This is line 101 + 10
 103|This is line 102
 104|This is line 103
[end of edit 1 / 1]
""".strip()
    )


def test_edit_obs_multiple_edits():
    _new_content = (
        '\n'.join([f'This is line {i}' for i in range(1, 50)])
        + '\nbalabala\n'
        + '\n'.join([f'This is line {i}' for i in range(50, 100)])
        + EDIT_LONG
        + '\n'.join([f'This is line {i}' for i in range(102, 1000)])
    )

    diff = get_diff(ORIGINAL_LONG, _new_content, '/workspace/app.py')
    obs = FileEditObservation(
        content=diff,
        path='/workspace/app.py',
        prev_exist=True,
        old_content=ORIGINAL_LONG,
        new_content=_new_content,
    )
    assert (
        str(obs).strip()
        == """
[Existing file /workspace/app.py is edited with 2 changes.]
[begin of edit 1 / 2]
(content before edits)
 48|This is line 48
 49|This is line 49
 50|This is line 50
 51|This is line 51
(content after edits)
 48|This is line 48
 49|This is line 49
+50|balabala
 51|This is line 50
 52|This is line 51
[end of edit 1 / 2]
-------------------------
[begin of edit 2 / 2]
(content before edits)
  98|This is line 98
  99|This is line 99
-100|This is line 100
-101|This is line 101
 102|This is line 102
 103|This is line 103
(content after edits)
  99|This is line 98
 100|This is line 99
+101|This is line 100 + 10
+102|This is line 101 + 10
 103|This is line 102
 104|This is line 103
[end of edit 2 / 2]
""".strip()
    )
