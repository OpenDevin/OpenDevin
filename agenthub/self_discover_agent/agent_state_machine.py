from enum import Enum

from opendevin.events.action import (
    Action,
    AgentDelegateAction,
    MessageAction,
)


class SelfDiscoverState(str, Enum):
    SELECT = 'select'
    ADAPT = 'adapt'
    IMPLEMENT = 'implement'
    DELEGATE = 'delegate'


class SelfDiscoverStateMachine:
    def __init__(self) -> None:
        self.current_state: SelfDiscoverState = SelfDiscoverState.SELECT
        self.prev_state: SelfDiscoverState | None = None

    def reset(self) -> None:
        self.current_state = SelfDiscoverState.SELECT
        self.prev_state = None

    def transition(self, action: Action) -> None:
        self.prev_state = self.current_state
        # Only move to next state if not browsing or user inquiry (for now deactivated)
        if not (
            (
                isinstance(action, AgentDelegateAction)
                and action.agent == 'BrowsingAgent'
            )
            or (isinstance(action, MessageAction) and action.wait_for_response)
        ):
            state_order = list(SelfDiscoverState)
            current_index = state_order.index(self.current_state)

            # Move to the next state if it exists
            if current_index < len(state_order) - 1:
                self.current_state = state_order[current_index + 1]
