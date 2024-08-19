from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeGuard, Union

from .agent_state_machine import SelfDiscoverState

# Order is important as it is used in prompt
REASONING_MODULE_LIST: List[str] = [
    'How could I devise an experiment to help solve that problem?',
    'Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.',
    'How could I measure progress on this problem?',
    'How can I simplify the problem so that it is easier to solve?',
    'What are the key assumptions underlying this problem?',
    'What are the potential risks and drawbacks of each solution?',
    'What are the alternative perspectives or viewpoints on this problem?',
    'What are the long-term implications of this problem and its solutions?',
    'How can I break down this problem into smaller, more manageable parts?',
    'Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.',
    'Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.',
    'Seek input and collaboration from others to solve the problem. Empha- size teamwork, open communication, and leveraging the diverse per- spectives and expertise of a group to come up with effective solutions.',
    'Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdepen- dencies that influence the problem, and developing holistic solutions that address the system as a whole.',
    'Use Risk Analysis: Evaluate potential risks, uncertainties, and trade- offs associated with different solutions or approaches to a problem. Em- phasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.',
    'Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assump- tions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.',
    'What is the core issue or problem that needs to be addressed?',
    'What are the underlying causes or factors contributing to the problem?',
    'Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?',
    'What are the potential obstacles or challenges that might arise in solving this problem?',
    'Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?',
    'Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?',
    'What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?',
    'How can progress or success in solving the problem be measured or evaluated?',
    'What indicators or metrics can be used?',
    'Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?',
    'Does the problem involve a physical constraint, such as limited re- sources, infrastructure, or space?',
    'Is the problem related to human behavior, such as a social, cultural, or psychological issue?',
    'Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?',
    'Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?',
    'Is the problem a design challenge that requires creative solutions and innovation?',
    'Does the problem require addressing systemic or structural issues rather than just individual instances?',
    'Is the problem time-sensitive or urgent, requiring immediate attention and action?',
    'What kinds of solution typically are produced for this kind of problem specification?',
    'Given the problem specification and the current best solution, have a guess about other possible solutions.',
    'Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?',
    'What is the best way to modify this current best solution, given what you know about these kinds of problem specification?',
    'Ignoring the current best solution, create an entirely new solution to the problem.',
    'Let’s think step by step.',
    'Let’s make a step by step plan and implement it with good notion and explanation.',
]

NestedDict = Dict[str, Union[str, 'NestedDict']]


def is_nested_dict(obj: Any) -> TypeGuard[NestedDict]:
    return isinstance(obj, dict) and all(
        isinstance(k, str) and (isinstance(v, str) or is_nested_dict(v))
        for k, v in obj.items()
    )


def is_str_dict(obj: Any) -> TypeGuard[Dict[str, str]]:
    return isinstance(obj, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in obj.items()
    )


@dataclass
class ReasoningAction:
    _selected_reasoning_modules: List[str] = field(default_factory=list)
    _adapted_reasoning_modules: Dict[str, str] = field(default_factory=dict)
    _reasoning_structure: NestedDict = field(default_factory=dict)

    @property
    def selected_reasoning_modules(self) -> List[str] | None:
        return (
            self._selected_reasoning_modules
            if self._selected_reasoning_modules
            else None
        )

    @property
    def adapted_reasoning_modules(self) -> Dict[str, str] | None:
        return (
            self._adapted_reasoning_modules if self._adapted_reasoning_modules else None
        )

    @property
    def reasoning_structure(self) -> NestedDict | None:
        return self._reasoning_structure if self._reasoning_structure else None

    def reset(self) -> None:
        self._selected_reasoning_modules.clear()
        self._adapted_reasoning_modules.clear()
        self._reasoning_structure.clear()

    def update_data(
        self,
        data: Dict[str, Union[List[str], Dict[str, str], NestedDict]],
    ) -> None:
        for k, v in data.items():
            if k == SelfDiscoverState.SELECT.value:
                if isinstance(v, list) and all(isinstance(item, str) for item in v):
                    self._selected_reasoning_modules = v
                else:
                    raise TypeError(
                        f'Data must be a list of strings for Self Discover state {SelfDiscoverState.SELECT.value}.'
                    )
            elif k == SelfDiscoverState.ADAPT.value:
                if is_str_dict(v):
                    self._adapted_reasoning_modules = v
                else:
                    raise TypeError(
                        f'Data must be a dictionary with strings keys and values for Self Discover state {SelfDiscoverState.ADAPT.value}.'
                    )
            elif k == SelfDiscoverState.IMPLEMENT.value:
                if is_nested_dict(v):
                    self._reasoning_structure = v
                else:
                    raise TypeError(
                        f'Data must be a nested dictionary with strings keys and values for Self Discover state {SelfDiscoverState.IMPLEMENT.value}.'
                    )
