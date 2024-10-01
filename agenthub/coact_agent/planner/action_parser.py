import re

from agenthub.codeact_agent.action_parser import (
    CodeActActionParserAgentDelegate,
    CodeActActionParserCmdRun,
    CodeActActionParserFinish,
    CodeActActionParserIPythonRunCell,
    CodeActActionParserMessage,
    CodeActResponseParser,
)
from openhands.controller.action_parser import ActionParser
from openhands.events.action import (
    Action,
    AgentDelegateAction,
)


class PlannerResponseParser(CodeActResponseParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - IPythonRunCellAction(code) - IPython code to run
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    - AgentFinishAction() - end the interaction
    """

    def __init__(self, initial_task_str=None):
        # Need pay attention to the item order in self.action_parsers
        super().__init__()
        self.action_parsers = [
            CodeActActionParserFinish(),
            CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            CodeActActionParserAgentDelegate(),
            CoActActionParserGlobalPlan(initial_task_str=initial_task_str),
        ]
        self.default_parser = CodeActActionParserMessage()

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        for action_suffix in [
            'bash',
            'ipython',
            'browse',
            'global_plan',
            'decide',
            'revise',
            'overrule',
            'collation',
        ]:
            if (
                f'<execute_{action_suffix}>' in action
                and f'</execute_{action_suffix}>' not in action
            ):
                action += f'</execute_{action_suffix}>'
        return action


class CoActActionParserGlobalPlan(ActionParser):
    """Parser action:
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def __init__(
        self,
        initial_task_str: list | None = None,
    ):
        self.global_plan: re.Match | None = None
        self.initial_task_str = initial_task_str or ['']

    def check_condition(self, action_str: str) -> bool:
        self.global_plan = re.search(
            r'<execute_global_plan>(.*)</execute_global_plan>', action_str, re.DOTALL
        )
        return self.global_plan is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.global_plan is not None
        ), 'self.global_plan should not be None when parse is called'
        thought = action_str.replace(self.global_plan.group(0), '').strip()
        global_plan_actions = self.global_plan.group(1).strip()

        # Some extra processing when doing swe-bench eval: extract text up to and including '--- END ISSUE ---'
        issue_text_pattern = re.compile(r'(.*--- END ISSUE ---)', re.DOTALL)
        issue_text_match = issue_text_pattern.match(self.initial_task_str[0])

        if issue_text_match:
            self.initial_task_str[0] = issue_text_match.group(1)

        return AgentDelegateAction(
            agent='CoActExecutorAgent',
            thought=thought,
            inputs={
                'task': f'The user message is: {self.initial_task_str[0]}.\nExecute the following plan to fulfill it:\n{global_plan_actions}'
            },
            action_suffix='global_plan',
        )
