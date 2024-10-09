from openhands.linter import DefaultLinter, LintResult
from openhands.utils.diff import get_diff, parse_diff

OLD_CONTENT = """
def foo():
    print("Hello, World!")
    x = UNDEFINED_VARIABLE
foo()
"""

NEW_CONTENT_V1 = (
    OLD_CONTENT
    + """
def new_function_that_causes_error():
    y = ANOTHER_UNDEFINED_VARIABLE
"""
)

NEW_CONTENT_V2 = """
def foo():
    print("Hello, World!")
    x = UNDEFINED_VARIABLE
    y = ANOTHER_UNDEFINED_VARIABLE
foo()
"""


def test_get_and_parse_diff(tmp_path):
    diff = get_diff(OLD_CONTENT, NEW_CONTENT_V1, 'test.py')
    print(diff)
    assert (
        diff
        == """
--- test.py
+++ test.py
@@ -6,0 +7,3 @@
+def new_function_that_causes_error():
+    y = ANOTHER_UNDEFINED_VARIABLE
+
""".strip()
    )

    print(
        '\n'.join(
            [f'{i+1}|{line}' for i, line in enumerate(NEW_CONTENT_V1.splitlines())]
        )
    )
    changes = parse_diff(diff)
    assert len(changes) == 3
    assert (
        changes[0].old is None
        and changes[0].new == 7
        and changes[0].line == 'def new_function_that_causes_error():'
    )
    assert (
        changes[1].old is None
        and changes[1].new == 8
        and changes[1].line == '    y = ANOTHER_UNDEFINED_VARIABLE'
    )
    assert changes[2].old is None and changes[2].new == 9 and changes[2].line == ''


def test_lint_with_diff_append(tmp_path):
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(OLD_CONTENT)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(NEW_CONTENT_V1)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    print(result)
    assert len(result) == 1
    assert (
        result[0].line == 8
        and result[0].column == 9
        and result[0].message == "F821 undefined name 'ANOTHER_UNDEFINED_VARIABLE'"
    )


def test_lint_with_diff_insert(tmp_path):
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(OLD_CONTENT)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(NEW_CONTENT_V2)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 1
    assert (
        result[0].line == 5
        and result[0].column == 9
        and result[0].message == "F821 undefined name 'ANOTHER_UNDEFINED_VARIABLE'"
    )


def test_lint_with_multiple_changes_and_errors(tmp_path):
    old_content = """
def foo():
    print("Hello, World!")
    x = 10
foo()
"""
    new_content = """
def foo():
    print("Hello, World!")
    x = UNDEFINED_VARIABLE
    y = 20

def bar():
    z = ANOTHER_UNDEFINED_VARIABLE
    return z + 1

foo()
bar()
"""
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 2
    assert (
        result[0].line == 4
        and result[0].column == 9
        and result[0].message == "F821 undefined name 'UNDEFINED_VARIABLE'"
    )
    assert (
        result[1].line == 8
        and result[1].column == 9
        and result[1].message == "F821 undefined name 'ANOTHER_UNDEFINED_VARIABLE'"
    )


def test_lint_with_introduced_and_fixed_errors(tmp_path):
    old_content = """
x = UNDEFINED_VARIABLE
y = 10
"""
    new_content = """
x = 5
y = ANOTHER_UNDEFINED_VARIABLE
z = UNDEFINED_VARIABLE
"""
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 2
    assert (
        result[0].line == 3
        and result[0].column == 5
        and result[0].message == "F821 undefined name 'ANOTHER_UNDEFINED_VARIABLE'"
    )
    assert (
        result[1].line == 4
        and result[1].column == 5
        and result[1].message == "F821 undefined name 'UNDEFINED_VARIABLE'"
    )


def test_lint_with_multiline_changes(tmp_path):
    old_content = """
def complex_function(a, b, c):
    return (a +
            b +
            c)
"""
    new_content = """
def complex_function(a, b, c):
    return (a +
            UNDEFINED_VARIABLE +
            b +
            c)
"""
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 1
    assert (
        result[0].line == 4
        and result[0].column == 13
        and result[0].message == "F821 undefined name 'UNDEFINED_VARIABLE'"
    )


def test_lint_with_syntax_error(tmp_path):
    old_content = """
def foo():
    print("Hello, World!")
"""
    new_content = """
def foo():
    print("Hello, World!"
"""
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 1
    assert (
        result[0].line == 3
        and result[0].column == 11
        and result[0].message == "E999 SyntaxError: '(' was never closed"
    )


def test_lint_with_docstring_changes(tmp_path):
    old_content = '''
def foo():
    """This is a function."""
    print("Hello, World!")
'''
    new_content = '''
def foo():
    """
    This is a function.
    It now has a multi-line docstring with an UNDEFINED_VARIABLE.
    """
    print("Hello, World!")
'''
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    assert len(result) == 0  # Linter should ignore changes in docstrings


def test_lint_with_multiple_errors_on_same_line(tmp_path):
    old_content = """
def foo():
    print("Hello, World!")
    x = 10
foo()
"""
    new_content = """
def foo():
    print("Hello, World!")
    x = UNDEFINED_VARIABLE + ANOTHER_UNDEFINED_VARIABLE
foo()
"""
    with open(tmp_path / 'old.py', 'w') as f:
        f.write(old_content)
    with open(tmp_path / 'new.py', 'w') as f:
        f.write(new_content)

    linter = DefaultLinter()
    result: list[LintResult] = linter.lint_file_diff(
        str(tmp_path / 'old.py'),
        str(tmp_path / 'new.py'),
    )
    print(result)
    assert len(result) == 2
    assert (
        result[0].line == 4
        and result[0].column == 9
        and result[0].message == "F821 undefined name 'UNDEFINED_VARIABLE'"
    )
    assert (
        result[1].line == 4
        and result[1].column == 30
        and result[1].message == "F821 undefined name 'ANOTHER_UNDEFINED_VARIABLE'"
    )


def test_parse_diff_with_empty_patch():
    diff_patch = ''
    changes = parse_diff(diff_patch)
    assert len(changes) == 0
