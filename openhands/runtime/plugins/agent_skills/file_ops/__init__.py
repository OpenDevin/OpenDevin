from openhands.runtime.plugins.agent_skills.file_ops import file_ops
from openhands.runtime.plugins.agent_skills.utils.dependency import import_functions

import_functions(
    module=file_ops, function_names=file_ops.__all__, target_globals=globals()
)
__all__ = file_ops.__all__

append_file = file_ops.append_file
edit_file_by_replace = file_ops.edit_file_by_replace
insert_content_at_line = file_ops.insert_content_at_line
