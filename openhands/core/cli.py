import asyncio
import os
import subprocess
from typing import Type

from termcolor import colored

import agenthub  # noqa F401 (we import this to get the agents registered)
from openhands import __version__
from openhands.controller import AgentController
from openhands.controller.agent import Agent
from openhands.core.config import (
    get_parser,
    load_app_config,
)
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import (
    Action,
    ChangeAgentStateAction,
    CmdRunAction,
    MessageAction,
)
from openhands.events.event import Event
from openhands.events.observation import (
    AgentStateChangedObservation,
    CmdOutputObservation,
)
from openhands.llm.llm import LLM
from openhands.runtime import get_runtime_cls
from openhands.runtime.runtime import Runtime
from openhands.storage import get_file_store


def display_message(message: str):
    print(colored('🤖 ' + message + '\n', 'yellow'))


def display_command(command: str):
    print('❯ ' + colored(command + '\n', 'green'))


def display_command_output(output: str):
    lines = output.split('\n')
    for line in lines:
        if line.startswith('[Python Interpreter') or line.startswith('openhands@'):
            # TODO: clean this up once we clean up terminal output
            continue
        print(colored(line, 'blue'))
    print('\n')


def display_event(event: Event):
    if isinstance(event, Action):
        if hasattr(event, 'thought'):
            display_message(event.thought)
    if isinstance(event, MessageAction):
        if event.source != EventSource.USER:
            display_message(event.content)
    if isinstance(event, CmdRunAction):
        display_command(event.command)
    if isinstance(event, CmdOutputObservation):
        display_command_output(event.content)


async def amain():
    """Runs the OpenHands CLI"""

    parser = get_parser(add_eval=False)
    # Add the version argument
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'{__version__}',
        help='Show the version number and exit',
        default=None,
    )
    # Add the launch subcommand
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    launch_parser = subparsers.add_parser('launch', help='Launch OpenHands')
    launch_parser.add_argument('mode', choices=['cli', 'ui'], help='Launch mode')

    args = parser.parse_args()

    if args.version:
        print(f'OpenHands version: {__version__}')
        return

    if args.command == 'launch':
        directory = os.getcwd()
        print(f'Launching OpenHands in workspace: {directory}...')
        if args.mode == 'cli':
            await launch_cli(directory)
        elif args.mode == 'ui':
            launch_ui(directory)
    else:
        parser.print_help()


def main():
    asyncio.run(amain())


async def launch_cli(directory: str):
    os.chdir(directory)
    config = load_app_config()
    sid = 'cli'

    agent_cls: Type[Agent] = Agent.get_cls(config.default_agent)
    agent_config = config.get_agent_config(config.default_agent)
    llm_config = config.get_llm_config_from_agent(config.default_agent)
    agent = agent_cls(
        llm=LLM(config=llm_config),
        config=agent_config,
    )

    file_store = get_file_store(config.file_store, config.file_store_path)
    event_stream = EventStream(sid, file_store)

    runtime_cls = get_runtime_cls(config.runtime)
    runtime: Runtime = runtime_cls(  # noqa: F841
        config=config,
        event_stream=event_stream,
        sid=sid,
        plugins=agent_cls.sandbox_plugins,
    )

    controller = AgentController(
        agent=agent,
        max_iterations=config.max_iterations,
        max_budget_per_task=config.max_budget_per_task,
        agent_to_llm_config=config.get_agent_to_llm_config_map(),
        event_stream=event_stream,
    )

    if controller is not None:
        controller.agent_task = asyncio.create_task(controller.start_step_loop())

    async def prompt_for_next_task():
        next_message = input('How can I help? >> ')
        if next_message == 'exit':
            event_stream.add_event(
                ChangeAgentStateAction(AgentState.STOPPED), EventSource.USER
            )
            return
        action = MessageAction(content=next_message)
        event_stream.add_event(action, EventSource.USER)

    async def on_event(event: Event):
        display_event(event)
        if isinstance(event, AgentStateChangedObservation):
            if event.agent_state == AgentState.ERROR:
                print('An error occurred. Please try again.')
            if event.agent_state in [
                AgentState.AWAITING_USER_INPUT,
                AgentState.FINISHED,
                AgentState.ERROR,
            ]:
                await prompt_for_next_task()

    event_stream.subscribe(EventStreamSubscriber.MAIN, on_event)

    await prompt_for_next_task()

    while controller.state.agent_state not in [
        AgentState.STOPPED,
    ]:
        await asyncio.sleep(1)  # Give back control for a tick, so the agent can run

    print('Exiting...')
    await controller.close()


def launch_ui(directory: str):
    os.chdir(directory)
    os.environ['WORKSPACE_BASE'] = directory
    os.environ['WORKSPACE_MOUNT_PATH'] = directory
    subprocess.run(['make', 'run'], check=True)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(amain())
    finally:
        pass
