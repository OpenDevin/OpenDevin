import argparse
import asyncio
from typing import Type

import agenthub  # noqa F401 (we import this to get the agents registered)
from opendevin.agent import Agent
from opendevin.controller import AgentController

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agent with a specific task")
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        type=str,
        help="The working directory for the agent",
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        type=str,
        help="The task for the agent to perform",
    )
    parser.add_argument(
        "-c",
        "--agent-cls",
        default="LangchainsAgent",
        type=str,
        help="The agent class to use",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="gpt-4-0125-preview",
        type=str,
        help="The (litellm) model name to use",
    )
    parser.add_argument(
        "-i",
        "--max-iterations",
        default=10,
        type=int,
        help="The maximum number of iterations to run the agent",
    )
    args = parser.parse_args()

    print(
        f'Running agent {args.agent_cls} (model: {args.model_name}, directory: {args.directory}) with task: "{args.task}"'
    )

    AgentCls: Type[Agent] = Agent.get_cls(args.agent_cls)
    agent = AgentCls(model_name=args.model_name)
    controller = AgentController(
        agent, workdir=args.directory, max_iterations=args.max_iterations
    )
    asyncio.run(controller.start_loop(args.task))
