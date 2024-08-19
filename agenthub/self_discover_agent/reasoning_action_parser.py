from typing import Dict, List, Union

from opendevin.core.utils import json

from .agent_state_machine import SelfDiscoverState
from .reasoning_action import REASONING_MODULE_LIST, NestedDict


class SelfDiscoverResponseToReasoningActionParser:
    """This class is an interface for a response parser dedicated to parsing the reasoning
    module selection, adaption and final structure from the response from the LLM.
    """

    def __init__(self):
        pass

    def parse(
        self, response: str
    ) -> Dict[str, Union[List[str], Dict[str, str], NestedDict]]:
        response_str = self.parse_response(response)
        return self.parse_reasoning(response_str)

    def parse_response(self, response) -> str:
        return response.choices[0].message.content

    def parse_reasoning(
        self, reasoning_str: str
    ) -> Dict[str, Union[List[str], Dict[str, str], NestedDict]]:
        reasoning_dict = json.loads(reasoning_str)
        if SelfDiscoverState.SELECT.value in reasoning_dict:
            reasoning_dict[SelfDiscoverState.SELECT.value] = [
                REASONING_MODULE_LIST[i]
                for i in reasoning_dict[SelfDiscoverState.SELECT.value]
            ]
        return reasoning_dict
