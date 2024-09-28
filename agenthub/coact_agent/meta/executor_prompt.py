EXECUTOR_PASS_CHECK = """Now, your role is to ensure the successful execution of actions in the global plan. Verify the results of these actions and compare them to the global plan. If they align, proceed to the next phase and output the action decision as ```move```. If discrepancies arise, you have two options:
1) If you suspect issues with your local plan, output the action ```revise```, or
2) If you suspect problems with the global planner's plan, trigger a request for replanning by outputting the action ```request```.
If the actions align with the global plan, explain the reasons for this alignment. If discrepancies arise, provide detailed reasons for your action decision:
- If you suspect issues with your local plan, explain why and use the action ```revise```.
- If you suspect problems with the global planner's plan, explain why and use the action ```request```.
Follow this structured output template:

**Output:**
- Output the action decision and reasons for your decision. Use the following format:
    ```
    Action: [action]
    Reasons: [reasons]
    ```
Next, you will be provided with the expected state of this phase and the execution results of your local plan.
"""  # FIXME: delete this as we may not use it

EXECUTOR_FALSE_CHECK = """You have encountered an exception in the execution process. Your current responsibility is to meticulously inspect the execution results of actions and identify the root causes of these exceptions. You have two options:
1) Suspect issues within your local plan and employ the action ```revise```, or
2) Suspect problems with the global planner's plan and trigger a request for replanning by executing the action ```request```.
Provide detailed reasons for your action decision:
- If you suspect issues with your local plan, explain why you decide to use the action ```revise```.
- If you suspect problems with the global planner's plan, explain why you decide to use the action ```request```.
Follow this structured output template:

**Output:**
- Output the action decision and reasons for your decision. Use the following format:
    ```
    Action: [action]
    Reasons: [reasons]
    ```
Next, you will be provided with the expected state of this phase and the execution exceptions during the execution of your local plan.
"""

EXECUTOR_SELF_REVISE = """Now, you have analyzed the situation and decided adjustments are needed to the local plan. Here is the reasons you proposed: {reasons}. Provide a revised plan using Page Operation Actions, and make sure to follow the format for action generation as mentioned earlier"""

EXECUTOR_OVERRULED_REVISE = """Facing your request, the global planner believes his previous global plan is correct, and refuse to adjust it and overrule your request. Here is the reasons he proposed: {reasons}. Based these information and your past experience, provide a revised plan using Page Operation Actions, and make sure to follow the format for action generation as mentioned earlier."""
