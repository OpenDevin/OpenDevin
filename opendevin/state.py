from dataclasses import dataclass
from typing import List, Tuple

from opendevin.action import Action
from opendevin.observation import CmdOutputObservation, Observation


@dataclass
class State:
    background_commands_obs: List[CmdOutputObservation]
    updated_info: List[Tuple[Action, Observation]]
