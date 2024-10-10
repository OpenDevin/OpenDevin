from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import Action, AgentDelegateAction, AgentFinishAction
from openhands.events.observation import AgentDelegateObservation
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory


class DelegatorAgent(Agent):
    VERSION = '1.0'
    """
    The Delegator Agent is responsible for delegating tasks to other agents based on the current task.
    """

    current_delegate: str = ''

    def __init__(self, llm: LLM, config: AgentConfig, memory: ConversationMemory):
        """Initialize the Delegator Agent with an LLM and memory

        Parameters:
        - llm: The llm to be used by this agent
        - config: The agent config
        - memory: The memory to be used by this agent
        """
        super().__init__(llm, config, memory=memory)

    def step(self, state: State) -> Action:
        """Checks to see if current step is completed, returns AgentFinishAction if True.
        Otherwise, delegates the task to the next agent in the pipeline.

        Parameters:
        - state: The current state given the previous actions and observations

        Returns:
        - AgentFinishAction: If the last state was 'completed', 'verified', or 'abandoned'
        - AgentDelegateAction: The next agent to delegate the task to
        """
        if self.current_delegate == '':
            self.current_delegate = 'study'
            task, _ = self.memory.get_current_user_intent()
            return AgentDelegateAction(
                agent='StudyRepoForTaskAgent', inputs={'task': task}
            )

        # last observation in history should be from the delegate
        last_observation = self.memory.get_last_observation()

        if not isinstance(last_observation, AgentDelegateObservation):
            raise Exception('Last observation is not an AgentDelegateObservation')

        goal, _ = self.memory.get_current_user_intent()
        if self.current_delegate == 'study':
            self.current_delegate = 'coder'
            return AgentDelegateAction(
                agent='CoderAgent',
                inputs={
                    'task': goal,
                    'summary': last_observation.outputs['summary'],
                },
            )
        elif self.current_delegate == 'coder':
            self.current_delegate = 'verifier'
            return AgentDelegateAction(
                agent='VerifierAgent',
                inputs={
                    'task': goal,
                },
            )
        elif self.current_delegate == 'verifier':
            if (
                'completed' in last_observation.outputs
                and last_observation.outputs['completed']
            ):
                return AgentFinishAction()
            else:
                self.current_delegate = 'coder'
                return AgentDelegateAction(
                    agent='CoderAgent',
                    inputs={
                        'task': goal,
                        'summary': last_observation.outputs['summary'],
                    },
                )
        else:
            raise Exception('Invalid delegate state')
