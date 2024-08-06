from dataclasses import dataclass, field

from opendevin.core.schema import ActionType

from .action import Action


@dataclass
class ChangeAgentStateAction(Action):
    """Fake action, just to notify the client that a task state has changed."""

    agent_state: str
    thought: str = ''
    status_message: str = ''
    action: str = ActionType.CHANGE_AGENT_STATE

    @property
    def message(self) -> str:
        return (
            self.status_message
            if self.status_message
            else f'Agent state changed to {self.agent_state}'
        )


@dataclass
class AgentSummarizeAction(Action):
    summary: str
    action: str = ActionType.SUMMARIZE

    @property
    def message(self) -> str:
        return self.summary

    def __str__(self) -> str:
        ret = '**AgentSummarizeAction**\n'
        ret += f'SUMMARY: {self.summary}'
        return ret


@dataclass
class AgentFinishAction(Action):
    outputs: dict = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.FINISH

    @property
    def message(self) -> str:
        if self.thought != '':
            return self.thought
        return "All done! What's next on the agenda?"


@dataclass
class AgentRejectAction(Action):
    outputs: dict = field(default_factory=dict)
    thought: str = ''
    action: str = ActionType.REJECT

    @property
    def message(self) -> str:
        msg: str = 'Task is rejected by the agent.'
        if 'reason' in self.outputs:
            msg += ' Reason: ' + self.outputs['reason']
        return msg


@dataclass
class AgentDelegateAction(Action):
    agent: str
    inputs: dict
    thought: str = ''
    action: str = ActionType.DELEGATE

    @property
    def message(self) -> str:
        return f"I'm asking {self.agent} for help with this task."
