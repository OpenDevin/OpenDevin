from opendevin.controller.action_parser import ActionParser, ResponseParser
from opendevin.core.utils import json
from opendevin.events.action import (
    Action,
    AgentDelegateAction,
    MessageAction,
)

from .agent_state_machine import SelfDiscoverState
from .prompt import TASK_KEY


def to_markdown(d: str | list | dict, indent=0) -> str:
    markdown = ''
    if isinstance(d, dict):
        for key, value in d.items():
            markdown += ' ' * indent + f'- **{key}**:'
            if isinstance(value, dict) or isinstance(value, list):
                markdown += '\n' + to_markdown(value, indent + 2)
            else:
                markdown += f' {value}\n'
    elif isinstance(d, list):
        for item in d:
            markdown += ' ' * indent + f'- {item}\n'
    else:
        markdown += ' ' * indent + f'- {d}\n'
    return markdown


class SelfDiscoverResponseToActionParser(ResponseParser):
    """
    Parser action:
        - AgentDelegateAction(agent, inputs) - delegate action to BrowserAgent for browsing or CodeActAgent for plan execution
        - MessageAction(content) - Message action for user interaction if 'wait_for_response' == True
        - MessageAction(content) - Message action to advance to next self discovery state if 'wait_for_response' == False
    """

    def __init__(self):
        super().__init__()
        # Order matters as plan is considered last step
        self.action_parsers = [
            SelfDiscoverActionParserExecutePlan(),
        ]
        self.default_parser = SelfDiscoverActionParserAdvance()

    def parse(self, response: str) -> Action:
        action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        return action

    def parse_action(self, action_str: str) -> Action:
        for action_parser in self.action_parsers:
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)


class SelfDiscoverActionParserExecutePlan(ActionParser):
    """
    Parser action:
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def check_condition(self, action_str: str) -> bool:
        action_dict = json.loads(action_str)
        return (
            SelfDiscoverState.IMPLEMENT.value in action_dict and TASK_KEY in action_dict
        )

    def parse(self, action_str: str) -> Action:
        action_dict = json.loads(action_str)
        task = action_dict[TASK_KEY]
        plan = to_markdown(action_dict[SelfDiscoverState.IMPLEMENT.value])

        input_task = task + (
            f'\n\nTo solve the task you must follow the step-by-step plan below\n{plan}'
        )
        thought = (
            f'I am proposing the following step-by-step plan to solve the task:\n{plan}'
        )
        return AgentDelegateAction(
            agent='CodeActAgent', inputs={'task': input_task}, thought=thought
        )


class SelfDiscoverActionParserAdvance(ActionParser):
    """
    Parser action:
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
    """

    def __init__(
        self,
    ):
        pass

    def check_condition(self, action_str: str) -> bool:
        return True

    def parse(self, action_str: str) -> Action:
        action_dict = json.loads(action_str)
        if (
            SelfDiscoverState.SELECT.value in action_dict
            and action_dict[SelfDiscoverState.SELECT.value] is not None
        ):
            content = (
                'Selection of reasoning modules finished. Moving on to adaptation ...'
            )
        elif (
            SelfDiscoverState.ADAPT.value in action_dict
            and action_dict[SelfDiscoverState.ADAPT.value] is not None
        ):
            content = 'Adaptation of reasoning modules finished. Moving on to building a reasoning structure ...'
        else:
            content = ''
        return MessageAction(content=content)
