import difflib

import whatthepatch


def get_diff(old_contents: str, new_contents: str, filepath: str = 'file') -> str:
    diff = list(
        difflib.unified_diff(
            old_contents.split('\n'),
            new_contents.split('\n'),
            fromfile=filepath,
            tofile=filepath,
            # do not output unchange lines
            # because they can cause `parse_diff` to fail
            n=0,
        )
    )
    return '\n'.join(map(lambda x: x.rstrip(), diff))


def parse_diff(diff_patch: str) -> list[whatthepatch.patch.Change]:
    # handle empty patch
    if diff_patch.strip() == '':
        return []

    patch = whatthepatch.parse_patch(diff_patch)
    patch_list = list(patch)
    assert len(patch_list) == 1, (
        'parse_diff only supports single file diff. But got:\nPATCH:\n'
        + diff_patch
        + '\nPATCH LIST:\n'
        + str(patch_list)
    )
    changes = patch_list[0].changes

    # ignore changes that are the same (i.e., old_lineno == new_lineno)
    output_changes = []
    for change in changes:
        if change.old != change.new:
            output_changes.append(change)
    return output_changes
