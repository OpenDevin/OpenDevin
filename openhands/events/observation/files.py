from dataclasses import dataclass
from difflib import SequenceMatcher

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class FileReadObservation(Observation):
    """This data class represents the content of a file."""

    path: str
    observation: str = ObservationType.READ

    @property
    def message(self) -> str:
        return f'I read the file {self.path}.'


@dataclass
class FileWriteObservation(Observation):
    """This data class represents a file write operation"""

    path: str
    observation: str = ObservationType.WRITE

    @property
    def message(self) -> str:
        return f'I wrote to the file {self.path}.'


@dataclass
class FileEditObservation(Observation):
    """This data class represents a file edit operation"""

    # content: str will be a unified diff patch string include NO context lines
    path: str
    prev_exist: bool
    old_content: str
    new_content: str
    observation: str = ObservationType.EDIT

    @property
    def message(self) -> str:
        return f'I edited the file {self.path}.'

    def visualize_diff(self, n_context_lines: int = 2) -> str:
        """Visualize the diff of the file edit.

        Instead of showing the diff line by line, this function
        shows each hunk of changes as a separate entity.

        Args:
            n_context_lines: The number of lines of context to show before and after the changes.
        """
        if self.content.strip() == '':
            # diff patch is empty
            return '(no changes detected. Please make sure your edits changes the content of the existing file.)\n'

        old_lines = self.old_content.split('\n')
        new_lines = self.new_content.split('\n')
        # Borrowed from difflib.unified_diff to directly parse into structured format.
        edit_groups: list[dict] = []
        for group in SequenceMatcher(None, old_lines, new_lines).get_grouped_opcodes(
            n_context_lines
        ):
            # take the max line number in the group
            _indent_pad_size = len(str(group[-1][3])) + 1  # +1 for the "*" prefix
            cur_group: dict[str, list[str]] = {
                'before_edits': [],
                'after_edits': [],
            }
            for tag, i1, i2, j1, j2 in group:
                if tag == 'equal':
                    for idx, line in enumerate(old_lines[i1:i2]):
                        cur_group['before_edits'].append(
                            f'{i1+idx+1:>{_indent_pad_size}}|{line}'
                        )
                    for idx, line in enumerate(new_lines[j1:j2]):
                        cur_group['after_edits'].append(
                            f'{j1+idx+1:>{_indent_pad_size}}|{line}'
                        )
                    continue
                if tag in {'replace', 'delete'}:
                    for idx, line in enumerate(old_lines[i1:i2]):
                        cur_group['before_edits'].append(
                            f'-{i1+idx+1:>{_indent_pad_size-1}}|{line}'
                        )
                if tag in {'replace', 'insert'}:
                    for idx, line in enumerate(new_lines[j1:j2]):
                        cur_group['after_edits'].append(
                            f'+{j1+idx+1:>{_indent_pad_size-1}}|{line}'
                        )
            edit_groups.append(cur_group)

        result = [
            f'[Existing file {self.path} is edited with {len(edit_groups)} changes.]'
        ]

        for i, cur_edit_group in enumerate(edit_groups):
            if i != 0:
                result.append('-------------------------')
            result.append(f'[begin of edit {i+1} / {len(edit_groups)}]')
            result.append('(content before edits)')
            result.extend(cur_edit_group['before_edits'])
            result.append('(content after edits)')
            result.extend(cur_edit_group['after_edits'])
            result.append(f'[end of edit {i+1} / {len(edit_groups)}]')
        return '\n'.join(result)

    def __str__(self) -> str:
        ret = ''
        if not self.prev_exist:
            assert (
                self.old_content == ''
            ), 'old_content should be empty if the file is new (prev_exist=False).'
            ret += f'[New file {self.path} is created with the provided content.]\n'
            return ret.rstrip() + '\n'
        ret += self.visualize_diff()
        return ret.rstrip() + '\n'
