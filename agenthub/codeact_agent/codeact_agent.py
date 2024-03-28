import re
from typing import List, Mapping

from termcolor import colored

from opendevin.agent import Agent
from opendevin.state import State
from opendevin.action import (
    Action,
    CmdRunAction,
    AgentEchoAction,
    AgentFinishAction,
)
from opendevin.observation import (
    CmdOutputObservation,
    AgentMessageObservation,
)

from opendevin.llm.llm import LLM

SYSTEM_MESSAGE = """You are a helpful assistant. You will be provided access (as root) to a bash shell to complete user-provided tasks.
You will be able to execute commands in the bash shell, interact with the file system, install packages, and receive the output of your commands.

DO NOT provide code in ```triple backticks```. Instead, you should execute bash command on behalf of the user by wrapping them with <execute> and </execute>.
For example:

You can list the files in the current directory by executing the following command:
<execute>ls</execute>

You can also install packages using pip:
<execute> pip install numpy </execute>

You can also write a block of code to a file:
<execute>
echo "import math
print(math.pi)" > math.py
</execute>

When you are done, execute "exit" to close the shell and end the conversation.
"""

INVALID_INPUT_MESSAGE = (
    "I don't understand your input. \n"
    "If you want to execute command, please use <execute> YOUR_COMMAND_HERE </execute>.\n"
    "If you already completed the task, please exit the shell by generating: <execute> exit </execute>."
)


def parse_response(response) -> str:
    action = response.choices[0].message.content
    if "<execute>" in action and "</execute>" not in action:
        action += "</execute>"
    return action


class CodeActAgent(Agent):
    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """
        Initializes a new instance of the CodeActAgent class.

        Parameters:
        - instruction (str): The instruction for the agent to execute.
        - max_steps (int): The maximum number of steps to run the agent.
        """
        super().__init__(llm)
        self.messages: List[Mapping[str, str]] = []
        self.instruction: str = ""

    def step(self, state: State) -> Action:
        if len(self.messages) == 0:
            assert self.instruction, "Expecting instruction to be set"
            self.messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": self.instruction},
            ]
            print(colored("===USER:===\n" + self.instruction, "green"))
        updated_info = state.updated_info
        if updated_info:
            for prev_action, obs in updated_info:
                assert isinstance(prev_action, (CmdRunAction, AgentEchoAction)), "Expecting CmdRunAction or AgentEchoAction for Action"
                if isinstance(obs, AgentMessageObservation):  # warning message from itself
                    self.messages.append({"role": "user", "content": obs.content})
                    print(colored("===USER:===\n" + obs.content, "green"))
                elif isinstance(obs, CmdOutputObservation):
                    content = "OBSERVATION:\n" + obs.content
                    content += f"\n[Command {obs.command_id} finished with exit code {obs.exit_code}]]"
                    self.messages.append({"role": "user", "content": content})
                    print(colored("===ENV OBSERVATION:===\n" + content, "blue"))
                else:
                    raise NotImplementedError(f"Unknown observation type: {obs.__class__}")
        response = self.llm.completion(
            messages=self.messages,
            stop=["</execute>"],
            temperature=0.0,
            seed=42,
        )
        action_str: str = parse_response(response)
        self.messages.append({"role": "assistant", "content": action_str})
        print(colored("===ASSISTANT:===\n" + action_str, "yellow"))

        command = re.search(r"<execute>(.*)</execute>", action_str, re.DOTALL)
        if command is not None:
            # a command was found
            command_group = command.group(1)
            if command_group.strip() == "exit":
                print(colored("Exit received. Exiting...", "red"))
                return AgentFinishAction()
            return CmdRunAction(command = command_group)
            # # execute the code
            # # TODO: does exit_code get loaded into Message?
            # exit_code, observation = self.env.execute(command_group)
            # self._history.append(Message(Role.ASSISTANT, observation))
            # print(colored("===ENV OBSERVATION:===\n" + observation, "blue"))
        else:
            # we could provide a error message for the model to continue similar to
            # https://github.com/xingyaoww/mint-bench/blob/main/mint/envs/general_env.py#L18-L23
            # observation = INVALID_INPUT_MESSAGE
            # self._history.append(Message(Role.ASSISTANT, observation))
            # print(colored("===ENV OBSERVATION:===\n" + observation, "blue"))
            return AgentEchoAction(content=INVALID_INPUT_MESSAGE)  # warning message to itself


    def search_memory(self, query: str) -> List[str]:
        raise NotImplementedError("Implement this abstract method")

