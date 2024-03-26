import asyncio
import traceback
from typing import List, Callable, Tuple, Any, Literal, Mapping
from termcolor import colored

from opendevin.state import State
from opendevin.agent import Agent
from opendevin.action import (
    Action,
    NullAction,
    FileReadAction,
    FileWriteAction,
    AgentFinishAction,
)
from opendevin.observation import Observation, NullObservation

from .command_manager import CommandManager

ColorType = Literal['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'light_grey', 'dark_grey', 'light_red', 'light_green', 'light_yellow', 'light_blue', 'light_magenta', 'light_cyan', 'white']

def print_with_color(text: Any, print_type: str = "INFO"):
    TYPE_TO_COLOR: Mapping[str, ColorType] = {
        "BACKGROUND LOG": "blue",
        "ACTION": "green",
        "OBSERVATION": "yellow",
        "INFO": "cyan",
    }
    color = TYPE_TO_COLOR.get(print_type.upper(), TYPE_TO_COLOR["INFO"])
    print(
        colored(f"\n{print_type.upper()}:\n", color, attrs=["bold"])
        + colored(str(text), color),
        flush=True,
    )


class AgentController:
    def __init__(
        self,
        agent: Agent,
        workdir: str,
        max_iterations: int = 100,
        callbacks: List[Callable] = [],
    ):
        self.agent = agent
        self.max_iterations = max_iterations
        self.workdir = workdir
        self.command_manager = CommandManager(workdir)
        self.callbacks = callbacks
        self.state_updated_info: List[Tuple[Action, Observation]] = []

    def get_current_state(self) -> State:
        # update observations & actions
        state = State(
            background_commands_obs=self.command_manager.get_background_obs(),
            updated_info=self.state_updated_info,
        )
        self.state_updated_info = []
        return state

    def add_observation(self, observation: Observation):
        self.state_updated_info.append((NullAction(), observation))

    async def start_loop(self, task_instruction: str):
        finished = False
        self.agent.instruction = task_instruction
        print_with_color(task_instruction, "OBSERVATION")
        for i in range(self.max_iterations):
            try:
                finished = await self.step(i)
            except Exception as e:
                print("Error in loop", e, flush=True)
                traceback.print_exc()
                break
            if finished:
                break
        if not finished:
            print("Exited before finishing", flush=True)

    async def step(self, i: int):
        print("\n==============", flush=True)
        print("STEP", i, flush=True)
        log_obs = self.command_manager.get_background_obs()
        for obs in log_obs:
            self.add_observation(obs)
            await self._run_callbacks(obs)
            print_with_color(obs, "BACKGROUND LOG")

        state: State = self.get_current_state()
        action: Action = self.agent.step(state)

        print_with_color(action, "ACTION")
        await self._run_callbacks(action)

        if isinstance(action, AgentFinishAction):
            print_with_color("FINISHED", "INFO")
            return True
        if isinstance(action, (FileReadAction, FileWriteAction)):
            action_cls = action.__class__
            _kwargs = action.__dict__
            _kwargs["base_path"] = self.workdir
            action = action_cls(**_kwargs)
        if action.executable:
            observation: Observation = action.run(self)
        else:
            observation = NullObservation("[Previous action was not executable, no observation produced.]")
        print_with_color(observation, "OBSERVATION")
        self.state_updated_info.append((action, observation))
        await self._run_callbacks(observation)

    async def _run_callbacks(self, event):
        if event is None:
            return
        for callback in self.callbacks:
            idx = self.callbacks.index(callback)
            try:
                callback(event)
            except Exception as e:
                print("Callback error:" + str(idx), e, flush=True)
                pass
        await asyncio.sleep(
            0.001
        )  # Give back control for a tick, so we can await in callbacks
