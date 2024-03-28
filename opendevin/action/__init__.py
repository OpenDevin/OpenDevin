from .base import Action, NullAction
from .bash import CmdRunAction, CmdKillAction
from .browse import BrowseURLAction
from .fileop import FileReadAction, FileWriteAction
from .agent import AgentRecallAction, AgentThinkAction, AgentFinishAction, AgentEchoAction, AgentSummarizeAction

actions = (
    CmdKillAction,
    CmdRunAction,
    BrowseURLAction,
    FileReadAction,
    FileWriteAction,
    AgentRecallAction,
    AgentThinkAction,
    AgentFinishAction
)

ACTION_TYPE_TO_CLASS = {action_class.action:action_class for action_class in actions} # type: ignore[attr-defined]

def action_class_initialize_dispatcher(action: str, *args: str, **kwargs: str) -> Action:
    action_class = ACTION_TYPE_TO_CLASS.get(action)
    if action_class is None:
        raise KeyError(f"'{action=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}")
    return action_class(*args, **kwargs)

CLASS_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_TO_CLASS.items()}

__all__ = [
    "Action",
    "NullAction",
    "CmdRunAction",
    "CmdKillAction",
    "BrowseURLAction",
    "FileReadAction",
    "FileWriteAction",
    "AgentRecallAction",
    "AgentThinkAction",
    "AgentFinishAction",
    "AgentEchoAction",
    "AgentSummarizeAction",
]
