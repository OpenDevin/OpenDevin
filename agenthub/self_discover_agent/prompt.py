from opendevin.core.message import Message, TextContent

from .agent_state_machine import SelfDiscoverState
from .reasoning_action import REASONING_MODULE_LIST, ReasoningAction

TASK_KEY = 'task'

ROLE_DESCRIPTION = """You are a seasoned manager at a Fortune 500 company with a team of capable software engineers.
Your responsibility is to plan the given task for your team. You must only plan and must not code yourself."""

INTERACTION_SKILL = """You are working with a curious user that needs your help. You can directly interact with the user by replying with
<execute_ask> and </execute_ask>. For example, <execute_ask> Are my assumptions correct? </execute_ask>.
"""

BROWSING_SKILL = """You can also browse the Internet with <execute_browse> and </execute_browse>.
For example, <execute_browse> Tell me the usa's president using google search </execute_browse>.
Or <execute_browse> Tell me what is in http://example.com </execute_browse>.
"""

SYSTEM_SUFFIX = """Let the following principles guide your response:
- Before responding read all information carefully.
- Take time to think.
- Responses should be concise.
- Use only information that's explicitly stated by the user.
"""

# TODO: allow agent to ask questions before specifying a plan
# SYSTEM_MESSAGE = ROLE_DESCRIPTION + INTERACTION_SKILL + BROWSING_SKILL + SYSTEM_SUFFIX
SYSTEM_MESSAGE = ROLE_DESCRIPTION + SYSTEM_SUFFIX

SELECT_PROMPT = """Select several reasoning moduls that are crucial to solve the task provided by the user.

## Task
Recall from above the task: {task}

## Reasoning modules list (0-based indexing)
{reasoning_module_list}

## Constraints
Select between 3 and 5 task-relevant reasoning modules.

Your response MUST be in JSON format. It must be an object with the field "{select_state_key}",
which contains the indeces of selected reasoning modules. For example, for selecting reasoning modules
"How could I devise an experiment to help solve that problem?" and "How could I measure progress on this problem?",
you would reply with

{{
  "{select_state_key}": [0, 2]
}}

You MUST NOT include any other text besides the JSON response.
"""

ADAPT_PROMPT = """Rephrase and specify each previously selected reasoning module so that it better helps solving the task.

## Selected reasoning modules
{selected_reasoning_modules}

## Task
Recall from above the task is as follows: {task}

## Constraints
Your response MUST be in JSON format. It must be an object with the field "{adapt_state_key}", which contains key-value pairs describing the adapted reasoning modules.
The key represents the title for the adapted reasoning module.
For example,

{{
  "{adapt_state_key}": {{
    "Title adapted reasoning module X": "Description adapted reasoning module X",
    "Title adapted reasoning module Y": "Description adapted reasoning module Y",
    }}
}}

You MUST NOT include any other text besides the JSON response.
"""

# TODO: modify example so that it is consistent with prompting
# ADAPT_EXAMPLE = """
# ## Example
# USER:
# This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
# (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle

# ASSISTANT:
# To solve the given task I selected the following candidate reasoning modules:
# - Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.
# - How can I break down this problem into smaller, more manageable parts?

# USER:
# Rephrase and specify each previously selected reasoning module so that it better helps solving the task.

# ## Task
# Recall from above the task is as follows: This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
# (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle

# ## Constraints
# Your response MUST be in JSON format. It must be an object with the field "{adapt_state_key}", which contains a list of key-value pairs describing the adapted reasoning modules.
# The key represents a title for the adapted reasoning module.
# For example,

# {{
#   "{adapt_state_key}": {{
#     {{"Title adapted reasoning module X": "Description adapted reasoning module X"}},
#     {{"Title adapted reasoning module Y": "Description adapted reasoning module Y"}},
#     }}
# }}

# The list must be in the same order as the list of previously selected reasoning modules.
# Each adapted reasoning module must be a separate key-value pair item in the list.
# You MUST NOT include any other text besides the JSON response.

# ASSISTANT:
# {{
#   "{adapt_state_key}":
#     {{
#       "Critical Thinking: Analyzing SVG Path Element": "This style involves analyzing the SVG path element from different perspectives, questioning assumptions about its shape, and evaluating the coordinates and lines described. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in interpreting the shape."
#     }},
#     {{
#       "Breaking Down the SVG Path into Parts": "How can I break down this SVG path element into smaller, more manageable parts? Identify and analyze each segment described in the path data, understand how they connect, and deduce the overall shape they form by systematically examining each line and vertex."
#     }}
# }}
# """


IMPLEMENT_PROMPT = """Based on the adapted reasoning modules implement a reasoning structure for your team members to follow step-by-step and arrive at the correct answer.

## Adapted reasoning structure
{adapted_reasoning_modules}

## Task
Recall from above the task: {task}

## Constraints
Do not solve the actual task yourself. Instead, you must plan the task by deriving a step-by-step reasoning structure, which guides your team to the correct answer.
Your response MUST be in JSON format. It must be an object with the fields "{task_key}" and "{implement_state_key}":
* {task_key}: this field must repeat the initial task from the user
* {implement_state_key}: key-values pairs which represent the step-by-by plan.
"""

IMPLEMENT_EXAMPLE = """
## Example
USER:
This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle

USER:
Based on the adapted reasoning modules implement a reasoning structure for your team members to follow step-by-step and arrive at the correct answer.

## Task
Recall from above the task: This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle

## Constraints
Do not solve the actual task yourself. Instead, you must plan the task by deriving a step-by-step reasoning structure, which guides your team to the correct answer.
Your response MUST be in JSON format. It must be an object with the fields "{task_key}" and "{implement_state_key}":
* {task_key}: this field must repeat the initial task from the user
* {implement_state_key}: key-values pairs which represent the step-by-by plan.

ASSISTANT:
{{
  "{task_key}": "This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
(A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle",
  "{implement_state_key}": {{
    "Step 1: Analyze the SVG Path Element": {{
      "Step 1.1: Review the SVG Path Data": "Examine the coordinates and line segments defined in the SVG path element.",
      "Step 1.2: Identify the Start and End Points": "Note the starting point and the ending point of the path to understand the sequence of drawing."
    }},
    "Step 2: Break Down the SVG Path into Parts": {{
      "Step 2.1: Divide the Path into Segments": {{
        "Step 2.1.1: Segment 1": "Analyze the coordinates and direction of the first line segment.",
        "Step 2.1.2: Segment 2": "Analyze the coordinates and direction of the second line segment.",
        "Step 2.1.3: Segment 3": "Analyze the coordinates and direction of the third line segment.",
        "Step 2.1.4: Continue until all segments are analyzed": "Proceed with analyzing each segment sequentially."
      }}
    }},
    "Step 3: Evaluate the Connections": {{
      "Step 3.1: Determine the Connections Between Segments": "Check how each segment connects to the next one.",
      "Step 3.2: Identify Closed Shapes or Open Paths": "Determine if the segments form a closed shape or an open path."
    }},
    "Step 4: Deduce the Overall Shape": {{
      "Step 4.1: Compile the Analyzed Segments": "Combine the information from all segments to understand the overall structure.",
      "Step 4.2: Match with Given Options": "Compare the deduced shape with the provided options (circle, heptagon, etc.) and identify the closest match."
    }}
  }}
}}
"""


def get_prompt(
    task: str, current_state: SelfDiscoverState, reasoning_data: ReasoningAction
) -> Message | None:
    if current_state == SelfDiscoverState.SELECT:
        content = SELECT_PROMPT.format(
            task=task,
            reasoning_module_list=REASONING_MODULE_LIST,
            select_state_key=current_state.value,
        )
    elif current_state == SelfDiscoverState.ADAPT:
        content = ADAPT_PROMPT.format(
            task=task,
            selected_reasoning_modules=reasoning_data.selected_reasoning_modules,
            adapt_state_key=current_state.value,
        )
    elif current_state == SelfDiscoverState.IMPLEMENT:
        content = IMPLEMENT_PROMPT.format(
            task=task,
            adapted_reasoning_modules=reasoning_data.adapted_reasoning_modules,
            task_key=TASK_KEY,
            implement_state_key=current_state.value,
        )
    else:
        return None

    return Message(role='user', content=[TextContent(text=content)])
