import os
import asyncio
from typing import Optional

from fastapi import WebSocketDisconnect

from opendevin.action import (
    Action,
    NullAction,
)
from opendevin.agent import Agent
from opendevin.controller import AgentController
from opendevin.llm.llm import LLM
from opendevin.observation import (
    Observation,
    UserMessageObservation
)

DEFAULT_WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", os.path.join(os.getcwd(), "workspace"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-0125-preview")

class Session:
    def __init__(self, websocket):
        self.websocket = websocket
        self.controller: Optional[AgentController] = None
        self.agent: Optional[Agent] = None
        self.agent_task = None
        asyncio.create_task(self.create_controller(), name="create controller") # FIXME: starting the docker container synchronously causes a websocket error...

    async def send_error(self, message):
        await self.send({"error": True, "message": message})

    async def send_message(self, message):
        await self.send({"message": message})

    async def send(self, data):
        if self.websocket is None:
            return
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            print("Error sending data to client", e)

    async def start_listening(self):
        try:
            while True:
                try:
                    data = await self.websocket.receive_json()
                except ValueError:
                    await self.send_error("Invalid JSON")
                    continue

                # TODO: we only need to implement user message for now
                # since even Devin does not support having the user taking other
                # actions (e.g., edit files) while the agent is running
                action = data.get("action", None)
                if action is None:
                    await self.send_error("Invalid event")
                elif self.controller is None:
                    await self.send_error("No agent started. Please wait a second...")
                elif action == "initialize":
                    await self.create_controller(data)
                elif action == "start":
                    await self.start_task(data)
                elif action == "chat":
                    self.controller.add_history(NullAction(), UserMessageObservation(data["message"]))
                else:
                    raise NotImplementedError("action not implemented")

        except WebSocketDisconnect as e:
            self.websocket = None
            if self.agent_task:
                self.agent_task.cancel()
            print("Client websocket disconnected", e)

    async def create_controller(self, start_event=None):
        directory = DEFAULT_WORKSPACE_DIR
        if start_event and "directory" in start_event["args"]:
            directory = start_event["args"]["directory"]
        agent_cls = "LangchainsAgent"
        if start_event and "agent_cls" in start_event["args"]:
            agent_cls = start_event["args"]["agent_cls"]
        model = LLM_MODEL
        if start_event and "model" in start_event["args"]:
            model = start_event["args"]["model"]
        if not os.path.exists(directory):
            print(f"Workspace directory {directory} does not exist. Creating it...")
            os.makedirs(directory)
        directory = os.path.relpath(directory, os.getcwd())
        llm = LLM(model)
        AgentCls = Agent.get_cls(agent_cls)
        self.agent = AgentCls(llm)
        self.controller = AgentController(self.agent, workdir=directory, callbacks=[self.on_agent_event])
        await self.send({"action": "initialize", "message": "Control loop started."})

    async def start_task(self, start_event):
        if "task" not in start_event["args"]:
            await self.send_error("No task specified")
            return
        await self.send_message("Starting new task...")
        task = start_event["args"]["task"]
        if self.controller is None:
            await self.send_error("No agent started. Please wait a second...")
            return
        self.agent_task = asyncio.create_task(self.controller.start_loop(task), name="agent loop")

    def on_agent_event(self, event: Observation | Action):
        event_dict = event.to_dict()
        asyncio.create_task(self.send(event_dict), name="send event in callback")
