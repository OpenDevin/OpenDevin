import re
import traceback
from typing import Dict, Optional

from datatypes import ParseError, StepOutput, TaskState
from tasks.base import Task

from opendevin.controller.state.state import State


class SimplifiedEnv:
    INVALID_INPUT_MESSAGE = (
        "I don't understand your input. \n"
        'If you want to execute code, please use <execute_ipython> YOUR_CODE_HERE </execute_ipython>.\n'
        'If you want to give me an answer, please use <solution> YOUR_SOLUTION_HERE </solution>.\n'
        'For example: The answer to the question is <solution> 42 </solution>. \n'
    )

    def __init__(self, agent_state: State, task: Task, task_config: Dict[str, int]):
        self.agent_state = agent_state
        self.task = task

        agent_action_count = {
            'propose_solution': 0,
            'use_tool': 0,
            'invalid_action': 0,
        }
        # check if agent_state has attribute turn_info set
        if hasattr(self.agent_state, 'propose_solution_count'):
            agent_action_count['propose_solution'] = (
                self.agent_state.propose_solution_count
            )

        self.task_state = TaskState(agent_action_count=agent_action_count)

        self.task_config = task_config

    async def step(self, lm_message: str):
        observation = self.handle_propose_solution(lm_message)

        self.check_max_iteration()

        turn_info = (
            self.task_config['max_iterations'] - self.agent_state.iteration,
            self.task_config['max_propose_solution']
            - self.task_state.agent_action_count['propose_solution'],
        )

        output = StepOutput(
            observation=observation,
            success=self.task_state.success,
            turn_info=turn_info,
        )

        self.agent_state.propose_solution_count = self.task_state.agent_action_count[
            'propose_solution'
        ]
        self.log_output(output)
        return self.task_state

    def handle_propose_solution(self, lm_message) -> Optional[str]:
        """Propose answer to check the task success.

        It might set self.state.finished = True if the task is successful.
        """
        self.task_state.agent_action_count['propose_solution'] += 1
        try:
            parsed = self.parse_propose_solution(lm_message)
            task_success = self.check_task_success(parsed['answer'])
            if task_success:
                self.task_state.finished = True
                self.task_state.success = True
                self.task_state.terminate_reason = 'task_success'
                # NOTE: should not return the function now, because we need to log the output
                # Set state.finished = True will terminate the episode
        except ParseError:
            return SimplifiedEnv.INVALID_INPUT_MESSAGE
        except Exception:
            error_traceback = traceback.format_exc()
            return f'{error_traceback}'

    def parse_propose_solution(self, lm_message: str) -> dict:
        """Define the parsing logic."""
        lm_output = '\n' + lm_message + '\n'

        answer = '\n'.join(
            [
                i.strip()
                for i in re.findall(r'<solution>(.*?)</solution>', lm_output, re.DOTALL)
            ]
        )
        if answer == '':
            raise ParseError('No answer found.')

        return {'answer': answer}

    def log_output(self, output: StepOutput) -> None:
        if self.task_state.finished:
            return

        content = output.to_str()
        self.task_state.latest_output = output.to_dict()
        self.task_state.latest_output['content'] = content

    def check_task_success(self, answer: str) -> bool:
        # log_message.info(f"STUDENT ANSWER: [{answer}]")
        # log_message.info(f"REFERENCE ANSWER: [{self.task.reference}]")
        return self.task.success(answer)

    def check_max_iteration(self):
        """Check if the agent has reached the max iteration limit.

        It might set self.state.finished = True if the agent has reached the max iteration limit.
        """
        if self.task_state.finished:
            # ignore if the episode is already finished (e.g., task success)
            return

        if (
            # propose solution > max output solution
            self.task_state.agent_action_count['propose_solution']
            >= self.task_config['max_propose_solution']
        ):
            self.task_state.finished = True
            self.task_state.success = False
            self.task_state.terminate_reason = 'max_propose_steps'
        elif self.agent_state.iteration >= self.task_config['max_iterations']:
            self.task_state.finished = True
            self.task_state.success = False
            self.task_state.terminate_reason = 'max_iterations'
