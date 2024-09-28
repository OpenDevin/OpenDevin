PLANNER_CONSTRUCT_PLAN = """Your role is to construct a multi-stage global plan, providing separate subtask descriptions and expected states for each phase. Ensure that your plan is comprehensive and covers all aspects of the task. Use the following format:

**Subtask 1:**
- **Subtask**: Describe the first subtask you are about to plan.
- **Expected State**: Define the expected state of the task after completing the first subtask.

**Subtask 2:**
- **Subtask**: Describe the second subtask you are about to plan.
- **Expected State**: Define the expected state of the task after completing the second subtask.

...

**Subtask n:**
- **Subtask**: Describe the nth subtask you are about to plan.
- **Expected State**: Define the expected state of the task after completing the nth subtask.
"""

PLANNER_DECIDE = """The local agent encountered issues with the global planner's global plan and he believes it's necessary to replan globally. Here is the reasons he proposed: {reasons}. Your current task is to decide whether to agree to the re-planning based on the description in the request for a new global plan submitted by the local agent. If you agree, the next action is to ```revise```; if not, the next action is to ```overrule```. Consider the implications of your decision and provide clear reasoning for it. Make sure to give clear and constructive guidance for plan adjustments.
    Follow this structured output template:

**Output:**
- Output the action decision and reasons for your decision. Use the following format:
    ```
    Action: [action]
    Reasons: [reasons]
    ```
"""

PLANNER_REVISE = """In response to the local agent's request, you have chosen to revise your previously made global plan. Reconsider the characteristics of the task based on the description in the request and create a new global plan. Ensure that your revised plan is well-detailed and addresses any issues identified. And make sure to follow the format for action generation as mentioned earlier.
"""

PLANNER_COLLATE = """Your role now involves collating the final result to meet the needs of the task. Ensure that the final output aligns with the global plan and that any necessary adjustments have been made. Provide a comprehensive summary of the task's completion.
"""
