from typing import TypedDict

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.config import AgentConfig
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import (
    Action,
    AgentFinishAction,
    IPythonRunCellAction,
)
from opendevin.events.observation import (
    Observation,
)
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from opendevin.runtime.plugins.agent_skills.agentless.agentless import (
    AGENTLESS_FINAL_PATCH_OBSERVATION,
    AGENTLESS_FOUND_FILE_OBSERVATION,
    AGENTLESS_LINE_LOCATIONS_OBSERVATION,
    AGENTLESS_RELATED_LOCATIONS_OBSERVATION,
    AGENTLESS_REPAIR_OBSERVATION,
    extract_observation,
)

ActionObs = TypedDict(
    'ActionObs', {'action': Action, 'observations': list[Observation]}
)


class AgentlessAgent(Agent):
    VERSION = '0.0.1'
    """
    AgentlessAgent
    """
    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    # action_parser = CodeActResponseParser()
    def __init__(self, llm: LLM, config: AgentConfig):
        super().__init__(llm, config)
        self.steps: list[ActionObs] = []
        self.reset()

    def reset(self) -> None:
        """
        Resets the CodeAct Agent.
        """
        super().reset()

    def search_memory(self, query: str) -> list[str]:
        return []

    def step(self, state: State) -> Action:
        """
        Performs the Agentless localization and repair

        Parameters:
        - state (State): used to get updated info

        Returns:
        - AgentFinishAction() - end the interaction
        """

        problem_statement = state.history.get_last_user_message()

        last_observation = state.history.get_last_observation()

        if not last_observation:
            return IPythonRunCellAction(code='install_agentless_dependencies()')
        else:
            prev_observation = last_observation.content
            logger.info('prev_observation:')
            logger.info(prev_observation)
            logger.info('---------end of prev_observation---------------')

            if 'Agentless dependencies installed' in prev_observation:
                logger.info('Localizing files')
                return IPythonRunCellAction(
                    code=f"agentless_file_localization('''{problem_statement}''')"
                )

            elif AGENTLESS_FOUND_FILE_OBSERVATION in prev_observation:
                logger.info('Localizing related')
                return IPythonRunCellAction(
                    code=f"agentless_related_localization('''{problem_statement}''', ''' {prev_observation} ''')"
                )

            elif AGENTLESS_RELATED_LOCATIONS_OBSERVATION in prev_observation:
                logger.info('Localizing lines')
                return IPythonRunCellAction(
                    code=f"agentless_line_level_localization('''{problem_statement}''', ''' {prev_observation} ''', 4)"
                )

            elif AGENTLESS_LINE_LOCATIONS_OBSERVATION in prev_observation:
                logger.info('Reparing')
                return IPythonRunCellAction(
                    code=f"agentless_repair_multi_context('''{problem_statement}''', ''' {prev_observation} ''', 21)"
                )

            elif AGENTLESS_REPAIR_OBSERVATION in prev_observation:
                logger.info('Post-processing repair')
                return IPythonRunCellAction(
                    code=f"agentless_post_process_repair(''' {prev_observation} ''')"
                )

            elif AGENTLESS_FINAL_PATCH_OBSERVATION in prev_observation:
                get_patch = extract_observation(
                    prev_observation, AGENTLESS_FINAL_PATCH_OBSERVATION
                )
                logger.info('Applying patch')
                return IPythonRunCellAction(code=f"apply_git_patch('''{get_patch}''')")
            else:
                return AgentFinishAction()
