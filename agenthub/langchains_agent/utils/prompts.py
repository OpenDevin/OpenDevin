from typing import List

from opendevin import config

from . import json

from opendevin.action import (
    action_from_dict,
    Action,
)
from opendevin.observation import (
    CmdOutputObservation,
)

def get_summarize_monologue_prompt(thoughts):
    monologue = json.dumps({'old_monologue': thoughts}, indent=2)
    return f"""
    Below is the internal monologue of an automated LLM agent. Each
    thought is an item in a JSON array. The thoughts may be memories,
    actions taken by the agent, or outputs from those actions.
    Please return a new, smaller JSON array, which summarizes the
    internal monologue. You can summarize individual thoughts, and
    you can condense related thoughts together with a description
    of their content.
    ```json
    {monologue}
    ```
    Make the summaries as pithy and informative as possible.
    Be specific about what happened and what was learned. The summary
    will be used as keywords for searching for the original memory.
    Be sure to preserve any key words or important information.

    Your response must be in JSON format. It must be an object with the
    key `new_monologue`, which is a JSON array containing the summarized monologue.
    Each entry in the array must have an `action` key, and an `args` key.
    The action key may be `summarize`, and `args.summary` should contain the summary.
    You can also use the same action and args from the source monologue.
    """

def get_request_action_prompt(
        task: str,
        thoughts: List[dict],
        background_commands_obs: List[CmdOutputObservation] = [],
):
    hint = ''
    if len(thoughts) > 0:
        latest_thought = thoughts[-1]
        if "action" in latest_thought:
            if latest_thought["action"] == 'think':
                if latest_thought["args"]['thought'].startswith("OK so my task is"):
                    hint = "You're just getting started! What should you do first?"
                else:
                    hint = "You've been thinking a lot lately. Maybe it's time to take action?"
            elif latest_thought["action"] == 'error':
                hint = "Looks like that last command failed. Maybe you need to fix it, or try something else."

    bg_commands_message = ""
    if len(background_commands_obs) > 0:
        bg_commands_message = "The following commands are running in the background:"
        for command_obs in background_commands_obs:
            bg_commands_message += f"\n`{command_obs.command_id}`: {command_obs.command}"
        bg_commands_message += "\nYou can end any process by sending a `kill` action with the numerical `id` above."
    latest_thought = thoughts[-1]

    monologue=json.dumps(thoughts, indent=2)

    return f"""
    You're a thoughtful robot. Your main task is to {task}.
    Don't expand the scope of your task--just complete it as written.

    This is your internal monologue, in JSON format:
    ```json
    {monologue}
    ```

    Your most recent thought is at the bottom of that monologue. Continue your train of thought.
    What is your next thought or action? Your response must be in JSON format.
    It must be an object, and it must contain two fields:
    * `action`, which is one of the actions below
    * `args`, which is a map of key-value pairs, specifying the arguments for that action

    Here are the possible actions:
    * `read` - reads the contents of a file. Arguments:
      * `path` - the path of the file to read
    * `write` - writes the contents to a file. Arguments:
      * `path` - the path of the file to write
      * `contents` - the contents to write to the file
    * `run` - runs a command. Arguments:
      * `command` - the command to run
      * `background` - if true, run the command in the background, so that other commands can be run concurrently. Useful for e.g. starting a server. You won't be able to see the logs. You don't need to end the command with `&`, just set this to true.
    * `kill` - kills a background command
      * `id` - the ID of the background command to kill
    * `browse` - opens a web page. Arguments:
      * `url` - the URL to open
    * `recall` - recalls a past memory. Arguments:
      * `query` - the query to search for
    * `think` - make a plan, set a goal, or record your thoughts. Arguments:
      * `thought` - the thought to record
    * `finish` - if you're absolutely certain that you've completed your task and have tested your work, use the finish action to stop working.

    {bg_commands_message}

    You MUST take time to think in between read, write, run, browse, and recall actions.
    You should never act twice in a row without thinking. But if your last several
    actions are all "think" actions, you should consider taking a different action.

    Notes:
    * your environment is Debian Linux. You can install software with `apt`
    * your working directory will not change, even if you run `cd`. All commands will be run in the `/workspace` directory.
    * don't run interactive commands, or commands that don't return (e.g. `node server.js`). You may run commands in the background (e.g. `node server.js &`)

    What is your next thought or action? Again, you must reply with JSON, and only with JSON.

    {hint}
    """

def parse_action_response(response: str) -> Action:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    response = response[json_start:json_end]
    action_dict = json.loads(response)
    if 'content' in action_dict:
        # The LLM gets confused here. Might as well be robust
        action_dict['contents'] = action_dict.pop('content')
    return action_from_dict(action_dict)

def parse_summary_response(response: str) -> List[dict]:
    parsed = json.loads(response)
    return parsed['new_monologue']
