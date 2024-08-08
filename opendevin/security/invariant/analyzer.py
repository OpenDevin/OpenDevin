import uuid
from typing import Any, Optional, List

import docker
import re

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action.action import (
    Action,
    ActionSecurityRisk,
)
from opendevin.events.event import Event, EventSource
from opendevin.events.observation import Observation
from opendevin.events.serialization.action import action_from_dict
from opendevin.events.stream import EventStream
from opendevin.runtime.utils import find_available_tcp_port
from opendevin.security.analyzer import SecurityAnalyzer
from opendevin.security.invariant.client import InvariantClient
from opendevin.security.invariant.parser import TraceElement, parse_element


class InvariantAnalyzer(SecurityAnalyzer):
    """Security analyzer based on Invariant."""

    trace: list[TraceElement]
    input: list[dict]
    container_name: str = 'opendevin-invariant-server'
    image_name: str = 'ghcr.io/invariantlabs-ai/server:opendevin'
    api_host: str = 'http://localhost'
    timeout: int = 180
    settings: dict = {}

    def __init__(
        self,
        event_stream: EventStream,
        policy: Optional[str] = None,
        sid: Optional[str] = None,
    ):
        """Initializes a new instance of the InvariantAnalzyer class."""
        super().__init__(event_stream)
        self.trace = []
        self.input = []
        self.settings = {}
        if sid is None:
            self.sid = str(uuid.uuid4())

        try:
            self.docker_client = docker.from_env()
        except Exception as ex:
            logger.exception(
                f'Error creating Invariant Security Analyzer container. Please check that Docker is running or disable the Security Analyzer in settings.',
                exc_info=False,
            )
            raise ex
        running_containers = self.docker_client.containers.list(
            filters={'name': self.container_name}
        )
        if not running_containers:
            all_containers = self.docker_client.containers.list(
                all=True, filters={'name': self.container_name}
            )
            if all_containers:
                self.container = all_containers[0]
                all_containers[0].start()
            else:
                self.api_port = find_available_tcp_port()
                self.container = self.docker_client.containers.run(
                    self.image_name,
                    name=self.container_name,
                    platform='linux/amd64',
                    ports={'8000/tcp': self.api_port},
                    detach=True,
                )
        else:
            self.container = running_containers[0]

        elapsed = 0
        while self.container.status != 'running':
            self.container = self.docker_client.containers.get(self.container_name)
            elapsed += 1
            logger.info(
                f'waiting for container to start: {elapsed}, container status: {self.container.status}'
            )
            if elapsed > self.timeout:
                break

        self.api_port = int(
            self.container.attrs['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort']
        )

        self.api_server = f'{self.api_host}:{self.api_port}'
        self.client = InvariantClient(self.api_server, self.sid)
        if policy is None:
            policy, _ = self.client.Policy.get_template()
            if policy is None:
                policy = ''
        self.monitor = self.client.Monitor.from_string(policy)

    async def close(self):
        self.container.stop()

    async def log_event(self, event: Event) -> None:
        if isinstance(event, Observation):
            element = parse_element(self.trace, event)
            self.trace.extend(element)
            self.input.extend([e.model_dump(exclude_none=True) for e in element])  # type: ignore [call-overload]
        else:
            logger.info('Invariant skipping element: event')

    def get_risk(self, results: List[str]) -> ActionSecurityRisk:
        mapping = {"high": ActionSecurityRisk.HIGH, "medium": ActionSecurityRisk.MEDIUM, "low": ActionSecurityRisk.LOW}
        regex = r'(?<=risk=)\w+'
        risks = []
        for result in results:
            m = re.search(regex, result)
            if m and m.group() in mapping:
                risks.append(mapping[m.group()])

        if risks:
            return max(risks)

        return ActionSecurityRisk.LOW

    async def act(self, event: Event) -> None:
        if await self.should_confirm(event):
            await self.confirm(event)

    async def should_confirm(self, event: Event) -> bool:
        risk = event.security_risk  # type: ignore [attr-defined]
        return risk is not None and risk < self.settings.get('RISK_SEVERITY', ActionSecurityRisk.MEDIUM) and hasattr(event, 'is_confirmed') and event.is_confirmed == "awaiting_confirmation"

    async def confirm(self, event: Event) -> None:
        new_event = action_from_dict({"action":"change_agent_state", "args":{"agent_state":"user_confirmed"}})
        if event.source:
            self.event_stream.add_event(new_event, event.source)
        else:
            self.event_stream.add_event(new_event, EventSource.AGENT)

    async def security_risk(self, event: Action) -> ActionSecurityRisk:
        logger.info('Calling security_risk on InvariantAnalyzer')
        new_elements = parse_element(self.trace, event)
        input = [e.model_dump(exclude_none=True) for e in new_elements]  # type: ignore [call-overload]
        self.trace.extend(new_elements)
        result, err = self.monitor.check(self.input, input)
        self.input.extend(input)
        risk = ActionSecurityRisk.UNKNOWN
        if err:
            logger.warning(f'Error checking policy: {err}')
            return risk

        risk = self.get_risk(result)

        return risk

    ### Handle API requests
    async def handle_api_request(self, request: Request) -> Any:
        path_parts = request.url.path.strip('/').split('/')
        endpoint = path_parts[-1]  # Get the last part of the path

        if request.method == 'GET':
            if endpoint == 'export-trace':
                return await self.export_trace(request)
            elif endpoint == 'policy':
                return await self.get_policy(request)
            elif endpoint == 'settings':
                return await self.get_settings(request)
        elif request.method == 'POST':
            if endpoint == 'policy':
                return await self.update_policy(request)
            elif endpoint == 'settings':
                return await self.update_settings(request)
        raise HTTPException(status_code=405, detail="Method Not Allowed")

    async def export_trace(self, request: Request) -> Any:
        return JSONResponse(content=self.input)

    async def get_policy(self, request: Request) -> Any:
        return JSONResponse(content={'policy': self.monitor.policy})

    async def update_policy(self, request: Request) -> Any:
        data = await request.json()
        policy = data.get('policy')
        new_monitor = self.client.Monitor.from_string(policy)
        self.monitor = new_monitor
        return JSONResponse(content={'policy': policy})

    async def get_settings(self, request: Request) -> Any:
        return JSONResponse(content=self.settings)

    async def update_settings(self, request: Request) -> Any:
        settings = await request.json()
        self.settings = settings
        return JSONResponse(content=self.settings)