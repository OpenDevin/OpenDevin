import json
from typing import List, Tuple, Dict, Type

from opendevin.plan import Plan
from opendevin.action import Action
from opendevin.observation import Observation

from opendevin.action import (
    NullAction,
    CmdRunAction,
    CmdKillAction,
    BrowseURLAction,
    FileReadAction,
    FileWriteAction,
    AgentRecallAction,
    AgentThinkAction,
    AgentFinishAction,
    AgentSummarizeAction,
    AddSubtaskAction,
    ModifySubtaskAction,
)

from opendevin.observation import (
    NullObservation,
)

ACTION_TYPE_TO_CLASS: Dict[str, Type[Action]] = {
    "run": CmdRunAction,
    "kill": CmdKillAction,
    "browse": BrowseURLAction,
    "read": FileReadAction,
    "write": FileWriteAction,
    "recall": AgentRecallAction,
    "think": AgentThinkAction,
    "summarize": AgentSummarizeAction,
    "finish": AgentFinishAction,
    "add_subtask": AddSubtaskAction,
    "modify_subtask": ModifySubtaskAction,
}

HISTORY_SIZE = 10

prompt = """
# Task
You're a diligent software engineer. You've been given the following task:

%(task)s

## Plan
As you complete this task, you're building a plan and keeping
track of your progress. Here's a JSON representation of your plan:
```json
%(plan)s
```

%(plan_status)s

## History
Here is a recent history of actions you've taken in service of this plan,
as well as observations you've made.
```json
%(history)s
```

Your most recent action is at the bottom of that history.

## Action
What is your next thought or action? Your response must be in JSON format.

It must be an object, and it must contain two fields:
* `action`, which is one of the actions below
* `args`, which is a map of key-value pairs, specifying the arguments for that action

* `read` - reads the contents of a file. Arguments:
  * `path` - the path of the file to read
* `write` - writes the contents to a file. Arguments:
  * `path` - the path of the file to write
  * `contents` - the contents to write to the file
* `run` - runs a command on the command line in a Linux shell. Arguments:
  * `command` - the command to run
  * `background` - if true, run the command in the background, so that other commands can be run concurrently. Useful for e.g. starting a server. You won't be able to see the logs. You don't need to end the command with `&`, just set this to true.
* `kill` - kills a background command
  * `id` - the ID of the background command to kill
* `browse` - opens a web page. Arguments:
  * `url` - the URL to open
* `think` - make a plan, set a goal, or record your thoughts. Arguments:
  * `thought` - the thought to record
* `add_subtask` - add a task to your plan. Arguments:
  * `parent` - the ID of the parent task
  * `goal` - the goal of the subtask
* `modify_subtask` - close a subtask. Arguments:
  * `id` - the ID of the subtask to close
  * `state` - set to 'in_progress' to start the task, 'completed' to finish it, 'abandoned' to give up on it permanently, or `open` to stop working on it for now.
* `finish` - if you're absolutely certain that you've completed your task and have tested your work, use the finish action to stop working.

You MUST take time to think in between read, write, run, browse, and recall actions.
You should never act twice in a row without thinking. But if your last several
actions are all `think` actions, you should consider taking a different action.

Based on the history above, if ANY of your open subtasks have been completed,
you MUST close them with the `modify_subtask` action.

What is your next thought or action? Again, you must reply with JSON, and only with JSON.

%(hint)s
"""

def get_prompt(plan: Plan, history: List[Tuple[Action, Observation]]):
    plan_str = json.dumps(plan.task.to_dict(), indent=2)
    sub_history = history[-HISTORY_SIZE:]
    history_dicts = []
    latest_action = ""
    for action, observation in sub_history:
        if not isinstance(action, NullAction):
            action_dict = action.to_dict()
            action_dict["action"] = convert_action(action_dict["action"])
            if 'base_dir' in action_dict:
                action_dict.pop('base_dir')
            history_dicts.append(action_dict)
            latest_action = action_dict["action"]
        if not isinstance(observation, NullObservation):
            observation_dict = observation.to_dict()
            observation_dict["observation"] = convert_observation(observation_dict["observation"])
            if 'base_dir' in observation_dict:
                observation_dict.pop('base_dir')
            history_dicts.append(observation_dict)
    history_str = json.dumps(history_dicts, indent=2)

    hint = ""
    current_task = plan.get_current_task()
    if current_task is not None:
        plan_status = f"You're currently working on this subtask: {current_task.goal}."
    else:
        plan_status = "You're not currently working on any subtasks. Your next action MUST be to mark a subtask as in_progress."
        hint = plan_status

    if current_task is not None:
        if latest_action == "":
            hint = "You haven't taken any actions yet. Start by using `ls` to check out what files you're working with."
        elif latest_action == "run":
            hint = "You should think about the command you just ran, and what output it gave. Maybe it's time to mark a subtask as complete."
        elif latest_action == "read":
            hint = "You should think about the file you just read, and what you learned from it."
        elif latest_action == "write":
            hint = "You just changed a file. You should probably run a command to check if your changes were successful, and have the intended behavior. Or maybe you should mark a subtask as complete."
        elif latest_action == "browse":
            hint = "You should think about the page you just visited, and what you learned from it."
        elif latest_action == "think":
            hint = "You should take some action, or adjust your plan."
        elif latest_action == "recall":
            hint = "You should think about the information you just recalled, and how it fits into your plan."
        elif latest_action == "add_subtask":
            hint = "You should take some action, or think about your next step."
        elif latest_action == "modify_subtask":
            hint = "You should take some action, or think about your next step."
        elif latest_action == "summarize":
            hint = ""
        elif latest_action == "finish":
            hint = ""

    return prompt % {
        'task': plan.main_goal,
        'plan': plan_str,
        'history': history_str,
        'hint': hint,
        'plan_status': plan_status,
    }

def parse_response(response: str) -> Action:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    response = response[json_start:json_end]
    action_dict = json.loads(response)
    if 'content' in action_dict:
        # The LLM gets confused here. Might as well be robust
        action_dict['contents'] = action_dict.pop('content')

    action = ACTION_TYPE_TO_CLASS[action_dict["action"]](**action_dict["args"])
    return action

def convert_action(action):
    if action == "CmdRunAction":
        action = "run"
    elif action == "CmdKillAction":
        action = "kill"
    elif action == "BrowseURLAction":
        action = "browse"
    elif action == "FileReadAction":
        action = "read"
    elif action == "FileWriteAction":
        action = "write"
    elif action == "AgentFinishAction":
        action = "finish"
    elif action == "AgentRecallAction":
        action = "recall"
    elif action == "AgentThinkAction":
        action = "think"
    elif action == "AgentSummarizeAction":
        action = "summarize"
    elif action == "AddSubtaskAction":
        action = "add_subtask"
    elif action == "ModifySubtaskAction":
        action = "modify_subtask"
    return action

def convert_observation(observation):
    if observation == "UserMessageObservation":
        observation = "chat"
    elif observation == "AgentMessageObservation":
        observation = "chat"
    elif observation == "CmdOutputObservation":
        observation = "run"
    elif observation == "FileReadObservation":
        observation = "read"
    return observation
