import os
from collections import defaultdict

from openhands.linter.base import BaseLinter, LinterException, LintResult
from openhands.linter.languages.python import PythonLinter
from openhands.linter.languages.treesitter import TreesitterBasicLinter
from openhands.utils.diff import get_diff, parse_diff


class DefaultLinter(BaseLinter):
    def __init__(self):
        self.linters: dict[str, list[BaseLinter]] = defaultdict(list)
        self.linters['.py'] = [PythonLinter()]

        # Add treesitter linter as a fallback for all linters
        self.basic_linter = TreesitterBasicLinter()
        for extension in self.basic_linter.supported_extensions:
            self.linters[extension].append(self.basic_linter)
        self._supported_extensions = list(self.linters.keys())

    @property
    def supported_extensions(self) -> list[str]:
        return self._supported_extensions

    def lint(self, file_path: str) -> list[LintResult]:
        if not os.path.isabs(file_path):
            raise LinterException(f'File path {file_path} is not an absolute path')
        file_extension = os.path.splitext(file_path)[1]

        linters: list[BaseLinter] = self.linters.get(file_extension, [])
        for linter in linters:
            res = linter.lint(file_path)
            # We always return the first linter's result (higher priority)
            if res:
                return res
        return []

    def lint_file_diff(
        self, original_file_path: str, updated_file_path: str
    ) -> list[LintResult]:
        """Only return lint errors that are introduced by the diff.

        Args:
            original_file_path: The original file path.
            updated_file_path: The updated file path.

        Returns:
            A list of lint errors that are introduced by the diff.
        """
        updated_lint_error: list[LintResult] = self.lint(updated_file_path)

        with open(original_file_path, 'r') as f:
            original_file_content = f.read()
        with open(updated_file_path, 'r') as f:
            updated_file_content = f.read()
        diff = get_diff(original_file_content, updated_file_content)
        changes = parse_diff(diff)

        # Select errors that are introduced by the new diff
        selected_errors = []
        for change in changes:
            new_lineno = change.new
            for error in updated_lint_error:
                if error.line == new_lineno:
                    selected_errors.append(error)
        return selected_errors
