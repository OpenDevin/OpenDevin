from typing import TypedDict, Union

from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.schema import AgentState
from openhands.events.action import (
    Action,
    AddTaskAction,
    AgentFinishAction,
    AgentRejectAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    MessageAction,
    ModifyTaskAction,
)
from openhands.events.observation import (
    AgentStateChangedObservation,
    CmdOutputObservation,
    FileReadObservation,
    FileWriteObservation,
    NullObservation,
    Observation,
)
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory

"""
FIXME: There are a few problems this surfaced
* FileWrites seem to add an unintended newline at the end of the file
* Browser not working
"""

ActionObs = TypedDict(
    'ActionObs', {'action': Action, 'observations': list[Observation]}
)


class DummyAgent(Agent):
    VERSION = '1.0'
    """
    The DummyAgent is used for e2e testing. It just sends the same set of actions deterministically,
    without making any LLM calls.
    """

    def __init__(self, llm: LLM, config: AgentConfig, memory: ConversationMemory):
        super().__init__(llm, config, memory)
        self.steps: list[ActionObs] = [
            {
                'action': AddTaskAction(
                    parent='None', goal='check the current directory'
                ),
                'observations': [],
            },
            {
                'action': AddTaskAction(parent='0', goal='run ls'),
                'observations': [],
            },
            {
                'action': ModifyTaskAction(task_id='0', state='in_progress'),
                'observations': [],
            },
            {
                'action': MessageAction('Time to get started!'),
                'observations': [],
            },
            {
                'action': CmdRunAction(command='echo "foo"'),
                'observations': [
                    CmdOutputObservation(
                        'foo', command_id=-1, command='echo "foo"', exit_code=0
                    )
                ],
            },
            {
                'action': FileWriteAction(
                    content='echo "Hello, World!"', path='hello.sh'
                ),
                'observations': [
                    FileWriteObservation(
                        content='echo "Hello, World!"', path='hello.sh'
                    )
                ],
            },
            {
                'action': FileReadAction(path='hello.sh'),
                'observations': [
                    FileReadObservation('echo "Hello, World!"\n', path='hello.sh')
                ],
            },
            {
                'action': CmdRunAction(command='bash hello.sh'),
                'observations': [
                    CmdOutputObservation(
                        'bash: hello.sh: No such file or directory',
                        command_id=-1,
                        command='bash workspace/hello.sh',
                        exit_code=127,
                    )
                ],
            },
            {
                'action': BrowseURLAction(url='https://google.com'),
                'observations': [
                    # BrowserOutputObservation('<html><body>Simulated Google page</body></html>',url='https://google.com',screenshot=''),
                ],
            },
            {
                'action': BrowseInteractiveAction(
                    browser_actions='goto("https://google.com")'
                ),
                'observations': [
                    # BrowserOutputObservation('<html><body>Simulated Google page after interaction</body></html>',url='https://google.com',screenshot=''),
                ],
            },
            {
                'action': AgentRejectAction(),
                'observations': [NullObservation('')],
            },
            {
                'action': AgentFinishAction(
                    outputs={}, thought='Task completed', action='finish'
                ),
                'observations': [AgentStateChangedObservation('', AgentState.FINISHED)],
            },
        ]

    def step(self, state: State) -> Action:
        if state.iteration >= len(self.steps):
            return AgentFinishAction()

        current_step = self.steps[state.iteration]
        action = current_step['action']

        # If the action is AddTaskAction or ModifyTaskAction, update the parent ID or task_id
        if isinstance(action, AddTaskAction):
            if action.parent == 'None':
                action.parent = ''  # Root task has no parent
            elif action.parent == '0':
                action.parent = state.root_task.id
            elif action.parent.startswith('0.'):
                action.parent = f'{state.root_task.id}{action.parent[1:]}'
        elif isinstance(action, ModifyTaskAction):
            if action.task_id == '0':
                action.task_id = state.root_task.id
            elif action.task_id.startswith('0.'):
                action.task_id = f'{state.root_task.id}{action.task_id[1:]}'
            # Ensure the task_id doesn't start with a dot
            if action.task_id.startswith('.'):
                action.task_id = action.task_id[1:]
        elif isinstance(action, (BrowseURLAction, BrowseInteractiveAction)):
            try:
                return self.simulate_browser_action(action)
            except (
                Exception
            ):  # This could be a specific exception for browser unavailability
                return self.handle_browser_unavailable(action)

        if state.iteration > 0:
            prev_step = self.steps[state.iteration - 1]

            if 'observations' in prev_step and prev_step['observations']:
                expected_observations = prev_step['observations']
                hist_events = self.memory.get_last_events(len(expected_observations))

                if len(hist_events) < len(expected_observations):
                    print(
                        f'Warning: Expected {len(expected_observations)} observations, but got {len(hist_events)}'
                    )

                for i in range(min(len(expected_observations), len(hist_events))):
                    hist_obs = event_to_dict(hist_events[i])
                    expected_obs = event_to_dict(expected_observations[i])

                    # Remove dynamic fields for comparison
                    for obs in [hist_obs, expected_obs]:
                        obs.pop('id', None)
                        obs.pop('timestamp', None)
                        obs.pop('cause', None)
                        obs.pop('source', None)
                        if 'extras' in obs:
                            obs['extras'].pop('command_id', None)

                    if hist_obs != expected_obs:
                        print(
                            f'Warning: Observation mismatch. Expected {expected_obs}, got {hist_obs}'
                        )

        return action

    def simulate_browser_action(
        self, action: Union[BrowseURLAction, BrowseInteractiveAction]
    ) -> Action:
        # Instead of simulating, we'll reject the browser action
        return self.handle_browser_unavailable(action)

    def handle_browser_unavailable(
        self, action: Union[BrowseURLAction, BrowseInteractiveAction]
    ) -> Action:
        # Create a message action to inform that browsing is not available
        message = 'Browser actions are not available in the DummyAgent environment.'
        if isinstance(action, BrowseURLAction):
            message += f' Unable to browse URL: {action.url}'
        elif isinstance(action, BrowseInteractiveAction):
            message += (
                f' Unable to perform interactive browsing: {action.browser_actions}'
            )
        return MessageAction(content=message)
