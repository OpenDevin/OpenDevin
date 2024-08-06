import pytest

from opendevin.runtime.utils.bash import split_bash_commands


def test_split_commands_util():
    cmds = [
        'ls -l',
        'echo -e "hello\nworld"',
        """
echo -e "hello it\\'s me"
""".strip(),
        """
echo \\
    -e 'hello' \\
    -v
""".strip(),
        """
echo -e 'hello\\nworld\\nare\\nyou\\nthere?'
""".strip(),
        """
echo -e 'hello
world
are
you\\n
there?'
""".strip(),
        """
echo -e 'hello
world "
'
""".strip(),
        """
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep
spec:
  containers:
  - name: busybox
    image: busybox:1.28
    args:
    - sleep
    - "1000000"
EOF
""".strip(),
        """
mkdir -p _modules && \
for month in {01..04}; do
    for day in {01..05}; do
        touch "_modules/2024-${month}-${day}-sample.md"
    done
done
""".strip(),
    ]
    joined_cmds = '\n'.join(cmds)
    split_cmds = split_bash_commands(joined_cmds)
    for s in split_cmds:
        print('\nCMD')
        print(s)
    for i in range(len(cmds)):
        assert (
            split_cmds[i]
            .replace('\\\n', '')
            .replace('\\', '')
            .replace('\n', '')
            .strip()
            == cmds[i].replace('\\\n', '').replace('\\', '').replace('\n', '').strip()
        ), f'At index {i}: {split_cmds[i]} != {cmds[i]}.'


@pytest.mark.parametrize(
    'input_command, expected_output',
    [
        ('ls -l', ['ls -l']),
        ("echo 'Hello, world!'", ["echo 'Hello, world!'"]),
        ('cd /tmp && touch test.txt', ['cd /tmp && touch test.txt']),
        ("echo -e 'line1\\nline2\\nline3'", ["echo -e 'line1\\nline2\\nline3'"]),
        (
            "grep 'pattern' file.txt | sort | uniq",
            ["grep 'pattern' file.txt | sort | uniq"],
        ),
        ('for i in {1..5}; do echo $i; done', ['for i in {1..5}; do echo $i; done']),
        (
            "echo 'Single quotes don\\'t escape'",
            ["echo 'Single quotes don\\'t escape'"],
        ),
        (
            'echo "Double quotes \\"do\\" escape"',
            ['echo "Double quotes \\"do\\" escape"'],
        ),
    ],
)
def test_single_commands(input_command, expected_output):
    assert split_bash_commands(input_command) == expected_output


def test_heredoc():
    input_commands = """
cat <<EOF
multiline
text
EOF
echo "Done"
"""
    expected_output = ['cat <<EOF\nmultiline\ntext\nEOF', 'echo "Done"']
    assert split_bash_commands(input_commands) == expected_output


def test_jupyter_heredoc():
    """This tests specifically test the behavior of the bash parser
    when the input is a heredoc for a Jupyter cell (used in ServerRuntime).

    It will failed to parse bash commands AND fall back to the original input,
    which won't cause issues in actual execution.

    [input]: cat > /tmp/opendevin_jupyter_temp.py <<'EOL'
    print('Hello, `World`!
    ')
    EOL
    [warning]: here-document at line 0 delimited by end-of-file (wanted "'EOL'") (position 75)

    TODO: remove this tests after the deprecation of ServerRuntime
    """

    code = "print('Hello, `World`!\n')"
    input_commands = f"""cat > /tmp/opendevin_jupyter_temp.py <<'EOL'
{code}
EOL"""
    expected_output = [f"cat > /tmp/opendevin_jupyter_temp.py <<'EOL'\n{code}\nEOL"]
    assert split_bash_commands(input_commands) == expected_output


def test_backslash_continuation():
    input_commands = """
echo "This is a long \
command that spans \
multiple lines"
echo "Next command"
"""
    expected_output = [
        'echo "This is a long command that spans multiple lines"',
        'echo "Next command"',
    ]
    assert split_bash_commands(input_commands) == expected_output


def test_comments():
    input_commands = """
echo "Hello" # This is a comment
# This is another comment
ls -l
"""
    expected_output = [
        'echo "Hello" # This is a comment\n# This is another comment',
        'ls -l',
    ]
    assert split_bash_commands(input_commands) == expected_output


def test_complex_quoting():
    input_commands = """
echo "This is a \\"quoted\\" string"
echo 'This is a '\''single-quoted'\'' string'
echo "Mixed 'quotes' in \\"double quotes\\""
"""
    expected_output = [
        'echo "This is a \\"quoted\\" string"',
        "echo 'This is a '''single-quoted''' string'",
        'echo "Mixed \'quotes\' in \\"double quotes\\""',
    ]
    assert split_bash_commands(input_commands) == expected_output


def test_invalid_syntax():
    invalid_inputs = [
        'echo "Unclosed quote',
        "echo 'Unclosed quote",
        'cat <<EOF\nUnclosed heredoc',
    ]
    for input_command in invalid_inputs:
        # it will fall back to return the original input
        assert split_bash_commands(input_command) == [input_command]


@pytest.fixture
def sample_commands():
    return [
        'ls -l',
        'echo "Hello, world!"',
        'cd /tmp && touch test.txt',
        'echo -e "line1\\nline2\\nline3"',
        'grep "pattern" file.txt | sort | uniq',
        'for i in {1..5}; do echo $i; done',
        'cat <<EOF\nmultiline\ntext\nEOF',
        'echo "Escaped \\"quotes\\""',
        "echo 'Single quotes don\\'t escape'",
        'echo "Command with a trailing backslash \\\n  and continuation"',
    ]


def test_split_single_commands(sample_commands):
    for cmd in sample_commands:
        result = split_bash_commands(cmd)
        assert len(result) == 1, f'Expected single command, got: {result}'


def test_split_commands_with_heredoc():
    input_commands = """
cat <<EOF
multiline
text
EOF
echo "Done"
"""
    expected_output = ['cat <<EOF\nmultiline\ntext\nEOF', 'echo "Done"']
    result = split_bash_commands(input_commands)
    assert result == expected_output, f'Expected {expected_output}, got {result}'


def test_split_commands_with_backslash_continuation():
    input_commands = """
echo "This is a long \
command that spans \
multiple lines"
echo "Next command"
"""
    expected_output = [
        'echo "This is a long command that spans multiple lines"',
        'echo "Next command"',
    ]
    result = split_bash_commands(input_commands)
    assert result == expected_output, f'Expected {expected_output}, got {result}'


def test_split_commands_with_empty_lines():
    input_commands = """
ls -l

echo "Hello"

cd /tmp
"""
    expected_output = ['ls -l', 'echo "Hello"', 'cd /tmp']
    result = split_bash_commands(input_commands)
    assert result == expected_output, f'Expected {expected_output}, got {result}'


def test_split_commands_with_comments():
    input_commands = """
echo "Hello" # This is a comment
# This is another comment
ls -l
"""
    expected_output = [
        'echo "Hello" # This is a comment\n# This is another comment',
        'ls -l',
    ]
    result = split_bash_commands(input_commands)
    assert result == expected_output, f'Expected {expected_output}, got {result}'


def test_split_commands_with_complex_quoting():
    input_commands = """
echo "This is a \\"quoted\\" string"
echo "Mixed 'quotes' in \\"double quotes\\""
"""
    # echo 'This is a '\''single-quoted'\'' string'

    expected_output = [
        'echo "This is a \\"quoted\\" string"',
        'echo "Mixed \'quotes\' in \\"double quotes\\""',
    ]
    # "echo 'This is a '\\''single-quoted'\\'' string'",
    result = split_bash_commands(input_commands)
    assert result == expected_output, f'Expected {expected_output}, got {result}'


def test_split_commands_with_invalid_input():
    invalid_inputs = [
        'echo "Unclosed quote',
        "echo 'Unclosed quote",
        'cat <<EOF\nUnclosed heredoc',
    ]
    for input_command in invalid_inputs:
        # it will fall back to return the original input
        assert split_bash_commands(input_command) == [input_command]
