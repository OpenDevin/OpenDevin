from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.events.action import Action, AgentDelegateAction, AgentFinishAction
from opendevin.events.observation import AgentDelegateObservation
from opendevin.llm.llm import LLM


class DelegatorAgent(Agent):
    VERSION = '1.0'
    """
    The Delegator Agent is responsible for delegating tasks to other agents based on the current task.
    """

    current_delegate: str = ''

    def __init__(self, llm: LLM):
        """Initialize the Delegator Agent with an LLM

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)

    def step(self, state: State) -> Action:
        """Checks to see if current step is completed, returns AgentFinishAction if True.
        Otherwise, delegates the task to the next agent in the pipeline.

        Parameters:
        - state (State): The current state given the previous actions and observations

        Returns:
        - AgentFinishAction: If the last state was 'completed', 'verified', or 'abandoned'
        - AgentDelegateAction: The next agent to delegate the task to
        """
        if self.current_delegate == '':
            self.current_delegate = 'study'
            task, _ = state.get_current_user_intent()
            return AgentDelegateAction(
                agent='StudyRepoForTaskAgent', inputs={'task': task}
            )

        # last observation in history should be from the delegate
        last_observation = state.history.get_last_observation()

        if not isinstance(last_observation, AgentDelegateObservation):
            raise Exception('Last observation is not an AgentDelegateObservation')

        goal, _ = state.get_current_user_intent()
        if self.current_delegate == 'study':
            self.current_delegate = 'coder'
            summary = last_observation.outputs.get('summary', '')
            return AgentDelegateAction(
                agent='CoderAgent',
                inputs={
                    'task': goal,
                    'summary': summary,
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
                summary = last_observation.outputs.get('summary', '')
                return AgentDelegateAction(
                    agent='CoderAgent',
                    inputs={
                        'task': goal,
                        'summary': summary,
                    },
                )
        else:
            raise Exception('Invalid delegate state')
