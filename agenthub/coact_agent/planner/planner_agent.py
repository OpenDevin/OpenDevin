import os

from agenthub.coact_agent.planner.action_parser import PlannerResponseParser
from agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM
from openhands.runtime.plugins.agent_skills.agentskills import (
    DOCUMENTATION_DICT as AGENTSKILLS_DOCS_DICT,
)
from openhands.utils.prompt import PromptManager


class GlobalPlannerAgent(CodeActAgent):
    VERSION = '1.0'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)

        self.action_parser = PlannerResponseParser(
            initial_task_str=self.initial_task_str
        )

        # Planner agent can do everything except file-editing operations
        planner_agentskills_exclude = [
            'create_file',
            'edit_file_by_replace',
            'insert_content_at_line',
            'append_file',
        ]
        planner_agentskills = [
            v
            for k, v in AGENTSKILLS_DOCS_DICT.items()
            if k not in planner_agentskills_exclude
        ]
        executor_editing_agentskills = [
            v
            for k, v in AGENTSKILLS_DOCS_DICT.items()
            if k in planner_agentskills_exclude
        ]
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__)),
            agent_skills_docs=''.join(planner_agentskills),
            system_extra_vars={
                'executor_editing_agent_skills_docs': ''.join(
                    executor_editing_agentskills
                )
            },
            micro_agent=self.micro_agent,
        )

        self.stop_sequences.append('</execute_global_plan>')
