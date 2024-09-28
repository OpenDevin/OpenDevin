import os

from agenthub.coact_agent.executor.action_parser import ExecutorResponseParser
from agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM
from openhands.runtime.plugins.agent_skills import AgentSkillsRequirement
from openhands.utils.prompt import PromptManager


class LocalExecutorAgent(CodeActAgent):
    VERSION = '1.0'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)

        self.action_parser = ExecutorResponseParser()
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__)),
            agent_skills_docs=AgentSkillsRequirement.documentation,
            micro_agent=self.micro_agent,
        )
        self.stop_sequences.append('</execute_request>')
