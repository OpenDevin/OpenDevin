from typing import List

from openhands.core.logger import openhands_logger as logger
from openhands.linter.base import BaseLinter, LintResult
from openhands.linter.utils import run_cmd


def python_compile_lint(fname: str) -> list[LintResult]:
    try:
        with open(fname, 'r') as f:
            code = f.read()
        compile(code, fname, 'exec')  # USE TRACEBACK BELOW HERE
        return []
    except SyntaxError as err:
        err_lineno = getattr(err, 'end_lineno', err.lineno)
        err_offset = getattr(err, 'end_offset', err.offset)
        if err_offset and err_offset < 0:
            err_offset = err.offset
        return [
            LintResult(
                file=fname, line=err_lineno, column=err_offset or 1, message=err.msg
            )
        ]


def flake_lint(filepath: str) -> list[LintResult]:
    fatal = 'F821,F822,F831,E112,E113,E999,E902'
    flake8_cmd = f'flake8 --select={fatal} --isolated {filepath}'

    try:
        cmd_outputs = run_cmd(flake8_cmd)
    except FileNotFoundError:
        return []
    results: list[LintResult] = []
    if not cmd_outputs:
        return results
    for line in cmd_outputs.splitlines():
        parts = line.split(':')
        if len(parts) >= 4:
            _msg = parts[3].strip()
            if len(parts) > 4:
                _msg += ': ' + parts[4].strip()

            try:
                line_num = int(parts[1])
            except ValueError as e:
                logger.warning(
                    f'Error parsing flake8 output for line: {e}. Parsed parts: {parts}. Skipping...'
                )
                continue

            try:
                column_num = int(parts[2])
            except ValueError as e:
                column_num = 1
                _msg = (
                    parts[2].strip() + ' ' + _msg
                )  # add the unparsed message to the original message
                logger.warning(
                    f'Error parsing flake8 output for column: {e}. Parsed parts: {parts}. Using default column 1.'
                )

            results.append(
                LintResult(
                    file=filepath,
                    line=line_num,
                    column=column_num,
                    message=_msg,
                )
            )
    return results


class PythonLinter(BaseLinter):
    @property
    def supported_extensions(self) -> List[str]:
        return ['.py']

    def lint(self, file_path: str) -> list[LintResult]:
        error = flake_lint(file_path)
        if not error:
            error = python_compile_lint(file_path)
        return error

    def compile_lint(self, file_path: str, code: str) -> List[LintResult]:
        try:
            compile(code, file_path, 'exec')
            return []
        except SyntaxError as e:
            return [
                LintResult(
                    file=file_path,
                    line=e.lineno,
                    column=e.offset,
                    message=str(e),
                    rule='SyntaxError',
                )
            ]
