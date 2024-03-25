import asyncio
from typing import Callable, List, Tuple

from opendevin.action import (
    Action,
    AgentFinishAction,
    FileReadAction,
    FileWriteAction,
    NullAction,
)
from opendevin.agent import Agent
from opendevin.observation import NullObservation, Observation
from opendevin.state import State

from .command_manager import CommandManager


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
        try:
            self.agent.instruction = task_instruction
            for i in range(self.max_iterations):
                print("STEP", i, flush=True)

                state: State = self.get_current_state()
                action: Action = self.agent.step(state)

                print("ACTION", action, flush=True)
                for _callback_fn in self.callbacks:
                    _callback_fn(action)

                if isinstance(action, AgentFinishAction):
                    print("FINISHED", flush=True)
                    break
                if isinstance(action, (FileReadAction, FileWriteAction)):
                    action_cls = action.__class__
                    _kwargs = action.__dict__
                    _kwargs["base_path"] = self.workdir
                    action = action_cls(**_kwargs)
                    print(action, flush=True)
                print("---", flush=True)

                if action.executable:
                    observation: Observation = action.run(self)
                else:
                    print("ACTION NOT EXECUTABLE", flush=True)
                    observation = NullObservation("")
                print("OBSERVATION", observation, flush=True)
                self.state_updated_info.append((action, observation))

                print(observation, flush=True)
                for _callback_fn in self.callbacks:
                    _callback_fn(observation)

                print("==============", flush=True)

                await asyncio.sleep(0.001)
        except Exception as e:
            print("Error in loop", e, flush=True)
            pass
