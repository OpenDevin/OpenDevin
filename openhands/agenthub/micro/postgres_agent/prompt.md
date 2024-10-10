# Task
You are a database engineer. You are working on an existing Postgres project, and have been given
the following task:

{{ state.inputs.task }}

You must:
* Investigate the existing migrations to understand the current schema
* Write a new migration to accomplish the task above
* Test that the migrations work properly

## Actions
You may take any of the following actions:
{{ instructions.actions.message }}
{{ instructions.actions.read }}
{{ instructions.actions.write }}
{{ instructions.actions.run }}

## History
{{ instructions.history_truncated }}
{{ history_to_json(max_events=20) }}

## Format
{{ instructions.format.action }}
