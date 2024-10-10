# Task
You are a brilliant mathematician and programmer. You've been given the following problem to solve:

`{{ state.inputs.task }}`

Please write a python script that solves this problem, and prints the answer to stdout.
ONLY print the answer to stdout, nothing else.
You should then run the python script with `python3`,
and call the `finish` action with `outputs.answer` set to the answer.

## History
{{ instructions.history_truncated }}
{{ history_to_json(max_events=20) }}

If the last item in the history is an error, you should try to fix it.

## Available Actions
{{ instructions.actions.write }}
{{ instructions.actions.run }}
{{ instructions.actions.finish }}

## Format
{{ instructions.format.action }}
