from openhands.controller.state.state import State
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import ActionType
from openhands.core.utils import json
from openhands.events.action import (
    Action,
)
from openhands.events.serialization.action import action_from_dict
from openhands.events.serialization.event import event_to_memory

HISTORY_SIZE = 20

prompt = """
# Task
You're a diligent software engineer AI. You can't see, draw, or interact with a
browser, but you can read and write files, and you can run commands, and you can think.

You've been given the following task:

%(task)s

## Plan
As you complete this task, you're building a plan and keeping
track of your progress. Here's a JSON representation of your plan:

%(plan)s


%(plan_status)s

You're responsible for managing this plan and the status of tasks in
it, by using the `add_task` and `modify_task` actions described below.

If the History below contradicts the state of any of these tasks, you
MUST modify the task using the `modify_task` action described below.

Be sure NOT to duplicate any tasks. Do NOT use the `add_task` action for
a task that's already represented. Every task must be represented only once.

Tasks that are sequential MUST be siblings. They must be added in order
to their parent task.

If you mark a task as 'completed', 'verified', or 'abandoned',
all non-abandoned subtasks will be marked the same way.
So before closing a task this way, you MUST not only be sure that it has
been completed successfully--you must ALSO be sure that all its subtasks
are ready to be marked the same way.

If, and only if, ALL tasks have already been marked verified,
you MUST respond with the `finish` action.

## History
Here is a recent history of actions you've taken in service of this plan,
as well as observations you've made. This only includes the MOST RECENT
ten actions--more happened before that.

%(history)s


Your most recent action is at the bottom of that history.

## Action
What is your next thought or action? Your response must be in JSON format.

It must be an object, and it must contain two fields:
* `action`, which is one of the actions below
* `args`, which is a map of key-value pairs, specifying the arguments for that action

* `read` - reads the content of a file. Arguments:
  * `path` - the path of the file to read
* `write` - writes the content to a file. Arguments:
  * `path` - the path of the file to write
  * `content` - the content to write to the file
* `run` - runs a command on the command line in a Linux shell. Arguments:
  * `command` - the command to run
* `browse` - opens a web page. Arguments:
  * `url` - the URL to open
* `message` - make a plan, set a goal, record your thoughts, or ask for more input from the user. Arguments:
  * `content` - the message to record
  * `wait_for_response` - set to `true` to wait for the user to respond before proceeding
* `add_task` - add a task to your plan. Arguments:
  * `parent` - the ID of the parent task (leave empty if it should go at the top level)
  * `goal` - the goal of the task
  * `subtasks` - a list of subtasks, each of which is a map with a `goal` key.
* `modify_task` - close a task. Arguments:
  * `task_id` - the ID of the task to close
  * `state` - set to 'in_progress' to start the task, 'completed' to finish it, 'verified' to assert that it was successful, 'abandoned' to give up on it permanently, or `open` to stop working on it for now.
* `finish` - if ALL of your tasks and subtasks have been verified or abandoned, and you're absolutely certain that you've completed your task and have tested your work, use the finish action to stop working.

You MUST take time to think in between read, write, run, and browse actions--do this with the `message` action.
You should never act twice in a row without thinking. But if your last several
actions are all `message` actions, you should consider taking a different action.

What is your next thought or action? Again, you must reply with JSON, and only with JSON.

%(hint)s
"""


def get_hint(latest_action_id: str) -> str:
    """Returns action type hint based on given action_id"""
    hints = {
        '': "You haven't taken any actions yet. Start by using `ls` to check out what files you're working with.",
        ActionType.RUN: 'You should think about the command you just ran, what output it gave, and how that affects your plan.',
        ActionType.READ: 'You should think about the file you just read, what you learned from it, and how that affects your plan.',
        ActionType.WRITE: 'You just changed a file. You should think about how it affects your plan.',
        ActionType.BROWSE: 'You should think about the page you just visited, and what you learned from it.',
        ActionType.MESSAGE: "Look at your last thought in the history above. What does it suggest? Don't think anymore--take action.",
        ActionType.ADD_TASK: 'You should think about the next action to take.',
        ActionType.MODIFY_TASK: 'You should think about the next action to take.',
        ActionType.SUMMARIZE: '',
        ActionType.FINISH: '',
    }
    return hints.get(latest_action_id, '')


def get_prompt(
    state: State,
    task: str,
    latest_action: Action | None,
    history_str: str,
    max_message_chars: int,
) -> str:
    """Gets the prompt for the planner agent.

    Formatted with the most recent action-observation pairs, current task, and hint based on last action

    Parameters:
    - state (State): The state of the current agent

    Returns:
    - str: The formatted string prompt with historical values
    """
    # the plan
    plan_str = json.dumps(state.root_task.to_dict(), indent=2)

    # the plan status
    current_task = state.root_task.get_current_task()
    if current_task is not None:
        plan_status = f"You're currently working on this task:\n{current_task.goal}."
        if len(current_task.subtasks) == 0:
            plan_status += "\nIf it's not achievable AND verifiable with a SINGLE action, you MUST break it down into subtasks NOW."
    else:
        plan_status = "You're not currently working on any tasks. Your next action MUST be to mark a task as in_progress."

    # the hint, based on the last action
    hint = (
        get_hint(event_to_memory(latest_action, max_message_chars).get('action', ''))
        if latest_action
        else ''
    )
    logger.info('HINT:\n' + hint, extra={'msg_type': 'DETAIL'})

    # finally, fill in the prompt
    return prompt % {
        'task': task,
        'plan': plan_str,
        'history': history_str,
        'hint': hint,
        'plan_status': plan_status,
    }


def parse_response(response: str) -> Action:
    """Parses the model output to find a valid action to take
    Parameters:
    - response (str): A response from the model that potentially contains an Action.

    Returns:
    - Action: A valid next action to perform from model output
    """
    action_dict = json.loads(response)
    if 'contents' in action_dict:
        # The LLM gets confused here. Might as well be robust
        action_dict['content'] = action_dict.pop('contents')
    action = action_from_dict(action_dict)
    return action
