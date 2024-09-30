import json
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from grep_ast import TreeContext, filename_to_lang
from tree_sitter_languages import get_parser  # noqa: E402

# tree_sitter is throwing a FutureWarning
warnings.simplefilter('ignore', category=FutureWarning)


@dataclass
class LintResult:
    text: str
    lines: list


class Linter:
    def __init__(self, encoding='utf-8', root=None):
        self.encoding = encoding
        self.root = root

        self.ts_installed = self._check_tool_installed('tsc')
        self.eslint_installed = self._check_tool_installed('eslint')

        self.languages = dict(
            python=self.py_lint,
        )
        if self.eslint_installed:
            self.languages['javascript'] = self.ts_eslint
            self.languages['typescript'] = self.ts_eslint
        elif self.ts_installed:
            self.languages['javascript'] = self.ts_tsc_lint
            self.languages['typescript'] = self.ts_tsc_lint
        self.all_lint_cmd = None

    def set_linter(self, lang, cmd):
        if lang:
            self.languages[lang] = cmd
            return

        self.all_lint_cmd = cmd

    def get_rel_fname(self, fname):
        if self.root:
            return os.path.relpath(fname, self.root)
        else:
            return fname

    def run_cmd(self, cmd, rel_fname, code):
        cmd += ' ' + rel_fname
        cmd = cmd.split()

        process = subprocess.Popen(
            cmd,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,  # Add stdin parameter
        )
        stdout, _ = process.communicate(
            input=code.encode()
        )  # Pass the code to the process
        errors = stdout.decode().strip()
        self.returncode = process.returncode
        if self.returncode == 0:
            return  # zero exit status

        cmd = ' '.join(cmd)
        res = ''
        res += errors
        line_num = Linter.extract_first_error_line_from(res)
        return LintResult(text=res, lines=[line_num])

    def get_abs_fname(self, fname):
        if os.path.isabs(fname):
            return fname
        elif os.path.isfile(fname):
            rel_fname = self.get_rel_fname(fname)
            return os.path.abspath(rel_fname)
        else:  # if a temp file
            return self.get_rel_fname(fname)

    def lint(self, fname, cmd=None) -> LintResult | None:
        code = Path(fname).read_text(self.encoding)
        absolute_fname = self.get_abs_fname(fname)
        if cmd:
            cmd = cmd.strip()
        if not cmd:
            lang = filename_to_lang(fname)
            if not lang:
                return None
            if self.all_lint_cmd:
                cmd = self.all_lint_cmd
            else:
                cmd = self.languages.get(lang)
        if callable(cmd):
            linkres = cmd(fname, absolute_fname, code)
        elif cmd:
            linkres = self.run_cmd(cmd, absolute_fname, code)
        else:
            linkres = basic_lint(absolute_fname, code)
        return linkres

    def flake_lint(self, rel_fname, code):
        fatal = 'F821,F822,F831,E112,E113,E999,E902'
        flake8 = f'flake8 --select={fatal} --isolated'

        try:
            flake_res = self.run_cmd(flake8, rel_fname, code)
        except FileNotFoundError:
            flake_res = None
        return flake_res

    def py_lint(self, fname, rel_fname, code):
        error = self.flake_lint(rel_fname, code)
        if not error:
            error = lint_python_compile(fname, code)
        if not error:
            error = basic_lint(rel_fname, code)
        return error

    def _check_tool_installed(self, tool_name: str) -> bool:
        """Check if a tool is installed."""
        try:
            subprocess.run(
                [tool_name, '--version'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def print_lint_result(self, lint_result: LintResult) -> None:
        print(f'\n{lint_result.text.strip()}')
        if isinstance(lint_result.lines, list) and lint_result.lines:
            if isinstance(lint_result.lines[0], LintResult):
                self.print_lint_result(lint_result.lines[0])

    def ts_eslint(self, fname: str, rel_fname: str, code: str) -> Optional[LintResult]:
        """Use ESLint to check for errors. If ESLint is not installed return None."""
        if not self.eslint_installed:
            return None

        # Enhanced ESLint configuration with React support
        eslint_config = {
            'env': {'es6': True, 'browser': True, 'node': True},
            'extends': ['eslint:recommended', 'plugin:react/recommended'],
            'parserOptions': {
                'ecmaVersion': 2021,
                'sourceType': 'module',
                'ecmaFeatures': {'jsx': True},
            },
            'plugins': ['react'],
            'rules': {
                'no-unused-vars': 'warn',
                'no-console': 'off',
                'react/prop-types': 'warn',
                'semi': ['error', 'always'],
            },
            'settings': {'react': {'version': 'detect'}},
        }

        # Write config to a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as temp_config:
            json.dump(eslint_config, temp_config)
            temp_config_path = temp_config.name

        try:
            # Point to frontend node_modules directory
            if self.root:
                plugin_path = f'{self.root}/frontend/node_modules/'
            else:
                return None

            eslint_cmd = f'eslint --no-eslintrc --config {temp_config_path} --resolve-plugins-relative-to {plugin_path} --format json'
            eslint_res = ''
            try:
                eslint_res = self.run_cmd(eslint_cmd, rel_fname, code)
                if eslint_res and hasattr(eslint_res, 'text'):
                    # Parse the ESLint JSON output
                    eslint_output = json.loads(eslint_res.text)
                    error_lines = []
                    error_messages = []
                    for result in eslint_output:
                        for message in result.get('messages', []):
                            line = message.get('line', 0)
                            error_lines.append(line)
                            error_messages.append(
                                f"{rel_fname}:{line}:{message.get('column', 0)}: {message.get('message')} ({message.get('ruleId')})"
                            )
                    if not error_messages:
                        return None

                    return LintResult(text='\n'.join(error_messages), lines=error_lines)
            except json.JSONDecodeError as e:
                return LintResult(text=f'\nJSONDecodeError: {e}', lines=[eslint_res])
            except FileNotFoundError:
                return None
            except Exception as e:
                return LintResult(text=f'\nUnexpected error: {e}', lines=[])
        finally:
            os.unlink(temp_config_path)
        return None

    def ts_tsc_lint(self, fname, rel_fname, code):
        """Use typescript compiler to check for errors. If TypeScript is not installed return None."""
        if self.ts_installed:
            tsc_cmd = 'tsc --noEmit --allowJs --checkJs --strict --noImplicitAny --strictNullChecks --strictFunctionTypes --strictBindCallApply --strictPropertyInitialization --noImplicitThis --alwaysStrict'
            try:
                tsc_res = self.run_cmd(tsc_cmd, rel_fname, code)
                if tsc_res:
                    # Parse the TSC output
                    error_lines = []
                    for line in tsc_res.text.split('\n'):
                        # Extract lines and column numbers
                        if ': error TS' in line or ': warning TS' in line:
                            try:
                                location_part = line.split('(')[1].split(')')[0]
                                line_num, _ = map(int, location_part.split(','))
                                error_lines.append(line_num)
                            except (IndexError, ValueError):
                                continue
                    return LintResult(text=tsc_res.text, lines=error_lines)
            except FileNotFoundError:
                pass

        # If still no errors, check for missing semicolons
        lines = code.split('\n')
        error_lines = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if (
                stripped_line
                and not stripped_line.endswith(';')
                and not stripped_line.endswith('{')
                and not stripped_line.endswith('}')
                and not stripped_line.startswith('//')
            ):
                error_lines.append(i + 1)

        if error_lines:
            error_message = (
                f"{rel_fname}({error_lines[0]},1): error TS1005: ';' expected."
            )
            return LintResult(text=error_message, lines=error_lines)

        # If tsc is not available return None (basic_lint causes other problems!)
        return None

    @staticmethod
    def extract_error_line_from(line):
        # TODO: this is a temporary fix to extract the error line from the error message
        # it should be replaced with a more robust/unified solution
        # eslint, for example, doesn't follow the same format
        if line.strip():
            # The format of the error message is: <filename>:<line>:<column>: <error code> <error message>
            parts = line.split(':')
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def extract_first_error_line_from(lint_error):
        for line in lint_error.splitlines(True):
            error_lineno = Linter.extract_error_line_from(line)
            if error_lineno is not None:
                return error_lineno
        return None

    @staticmethod
    def extract_error_lines_from(lint_error):
        lines = []
        for line in lint_error.splitlines(True):
            error_lineno = Linter.extract_error_line_from(line)
            if error_lineno is not None:
                lines.append(error_lineno)
        return lines

    @staticmethod
    def refine_lint_error(
        original_lint_error: str,
        lint_error: str,
        is_append: bool,
        is_insert: bool,
        lines: list,
        start: int,
        end: int,
        n_added_lines: int,
    ):
        """
        Given original_lint_error (pre-edit-lint) and lint_error (post-edit-lint),
        select only new linting errors that are caused by the edit
        """
        original_error_lines = set(Linter.extract_error_lines_from(original_lint_error))
        new_lint_errors = []
        first_error_lineno = None

        # due to the edit, the old line numbers and new line numbers may not match
        # for every lint error, check if it is caused by modification
        for line in lint_error.splitlines():
            lineno = Linter.extract_error_line_from(line)
            if lineno is None:
                continue
            elif is_append:
                # append: only retain errors from appendication chunk, since
                # an appendication never causes new errors in other places
                if lineno > len(lines):
                    new_lint_errors.append(line)
                else:
                    continue
            elif is_insert:
                # insert: similar to append, only retain errors from insertion
                # chunk, since an insertion never causes new errors in other places,
                # except the line right below the insertion
                if lineno >= start and lineno <= start + n_added_lines:
                    new_lint_errors.append(line)
                else:
                    continue
            else:
                # edit:
                if lineno >= start and lineno <= start + n_added_lines:
                    # error coming from edit part itself
                    new_lint_errors.append(line)
                elif lineno < start and lineno not in original_error_lines:
                    # error coming before edit part, and is new, meaning the
                    # error is caused by the edit - e.g. edit changes a
                    # function signature and breaks existing code
                    new_lint_errors.append(line)
                elif (
                    lineno > start + n_added_lines
                    and (lineno - n_added_lines + end - start + 1)
                    not in original_error_lines
                ):
                    # error coming after edit part, and is new, meaning the
                    # error is caused by the edit - e.g. edit changes a
                    # function signature and breaks existing code
                    new_lint_errors.append(line)
                else:
                    continue

            if first_error_lineno is None:
                first_error_lineno = lineno

        if len(new_lint_errors) == 0:
            return None, None
        else:
            return '\n'.join(new_lint_errors), first_error_lineno


def lint_python_compile(fname, code):
    try:
        compile(code, fname, 'exec')  # USE TRACEBACK BELOW HERE
        return
    except IndentationError as err:
        end_lineno = getattr(err, 'end_lineno', err.lineno)
        if isinstance(end_lineno, int):
            line_numbers = list(range(end_lineno - 1, end_lineno))
        else:
            line_numbers = []

        tb_lines = traceback.format_exception(type(err), err, err.__traceback__)
        last_file_i = 0

        target = '# USE TRACEBACK'
        target += ' BELOW HERE'
        for i in range(len(tb_lines)):
            if target in tb_lines[i]:
                last_file_i = i
                break
        tb_lines = tb_lines[:1] + tb_lines[last_file_i + 1 :]

    res = ''.join(tb_lines)
    return LintResult(text=res, lines=line_numbers)


def basic_lint(fname, code):
    """Use tree-sitter to look for syntax errors, display them with tree context."""
    lang = filename_to_lang(fname)
    if not lang:
        return

    parser = get_parser(lang)
    tree = parser.parse(bytes(code, 'utf-8'))

    errors = traverse_tree(tree.root_node)
    if not errors:
        return

    error_messages = [
        f'{fname}:{line}:{col}: {error_details}' for line, col, error_details in errors
    ]
    return LintResult(
        text='\n'.join(error_messages), lines=[line for line, _, _ in errors]
    )


def tree_context(fname, code, line_nums):
    context = TreeContext(
        fname,
        code,
        color=False,
        line_number=True,
        child_context=False,
        last_line=False,
        margin=0,
        mark_lois=True,
        loi_pad=3,
        # header_max=30,
        show_top_of_file_parent_scope=False,
    )
    line_nums = set(line_nums)
    context.add_lines_of_interest(line_nums)
    context.add_context()
    output = context.format()

    return output


def traverse_tree(node):
    """Traverses the tree to find errors"""
    errors = []
    if node.type == 'ERROR' or node.is_missing:
        line_no = node.start_point[0] + 1
        col_no = node.start_point[1] + 1
        error_type = 'Missing node' if node.is_missing else 'Syntax error'
        errors.append((line_no, col_no, error_type))

    for child in node.children:
        errors += traverse_tree(child)

    return errors


def main():
    """Main function to parse files provided as command line arguments."""
    if len(sys.argv) < 2:
        print('Usage: python linter.py <file1> <file2> ...')
        sys.exit(1)

    linter = Linter(root=os.getcwd())
    for file_path in sys.argv[1:]:
        errors = linter.lint(file_path)
        if errors:
            print(errors)


if __name__ == '__main__':
    main()
