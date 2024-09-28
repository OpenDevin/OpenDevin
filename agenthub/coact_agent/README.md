# CoAct Multi-Agent Framework

This folder implements a multi-agent workflow inspired by the CoAct framework ([paper](https://arxiv.org/abs/2406.13381)), that provides a robust structure for defining, planning, and executing tasks using multiple agents.

## Agents

1. `CoActPlannerAgent`:
    - is responsible for exploring and creating a global plan. It can replan if there are issues with the previous one.
    - has full capabilities of [CodeActAgent](https://github.com/All-Hands-AI/OpenHands/tree/main/agenthub/codeact_agent).
2. `CoActExecutorAgent`:
    - is responsible for executing the proposed plan. Facing issues with the plan, it can request for a new one.
    - also has full capabilities of [CodeActAgent](https://github.com/All-Hands-AI/OpenHands/tree/main/agenthub/codeact_agent).


## Plan structure
```markdown
The user message is: <<Full user's message here.>>
# Phases
## Phase 1
- reason: <<Assistant's thorough thoughts on why this phase is necessary, with tips/codes to instruct the executor finish the task easier.>>
- description: <<Describe what needs to be done in this phase.>>
- expected_state: <<Describe the expected state after this phase is completed. If the task involves code editing, provide the expectation of the code after the edit.>>
<file_path> <<The file path to edit. In one phase only 1 file is edited.>> </file_path>
<expected_content>
<<The partial expected content here WITH LINE NUMBERS and a vertical bar before the actual code e.g., 1|, 11|.>>
</expected_content>
## Phase 2
- reason: ...
- description: ...
- expected_state: ...
<file_path> ... </file_path>
<expected_content>
...|...
</expected_content>
## Phase ...
```
