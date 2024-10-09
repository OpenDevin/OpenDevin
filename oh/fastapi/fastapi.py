import asyncio
import logging
from typing import Annotated, List, Optional
from uuid import UUID
from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Path,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.websockets import WebSocketState

from oh.agent.agent_config import AgentConfig
from oh.agent.agent_info import AgentInfo
from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC
from oh.fastapi.conversation_info import ConversationInfo
from oh.fastapi.dynamic_types import DynamicTypes
from oh.fastapi.websocket_conversation_listener import WebsocketConversationListener
from oh.fastapi.websocket_conversation_broker_listener import (
    WebsocketConversationBrokerListener,
)
from oh.file.file_filter import FileFilter
from oh.file.file_info import FileInfo
from oh.conversation_broker.conversation_broker_abc import ConversationBrokerABC
from oh.storage.page import Page
from oh.util.async_util import wait_all

_LOGGER = logging.getLogger(__name__)
FILE_UPLOAD_TIMEOUT = 60  # TODO: make this configurable
GENERAL_TIMEOUT = 15


def add_open_hands_to_fastapi(api: FastAPI, conversation_broker: ConversationBrokerABC):
    """
    OpenHands external API consists of 3 main entities:

    Announcement - Some (Polymorphic) event that happened on the server
    Command - Some command to perform on the server
    Conversation - Context in which commands are performed and events are triggered

    To begin creating commands and receiving events, a conversation is required:

    POST   /conversation  - begin a conversation
    GET    /conversation  - list conversations
    GET    /conversation-count  - count conversations
    GET    /conversation/{conversation_id}  - get conversation info
    DELETE /conversation/{conversation_id}  - finish a conversation
    GET    /conversation/{conversation_id}/event  - list conversation events
    POST   /conversation/{conversation_id}/event  - trigger a conversation event
    GET    /conversation/{conversation_id}/event/{event_id}  - get a conversation event
    GET    /conversation/{conversation_id}/command  - list conversation commands
    GET    /conversation/{conversation_id}/command/{command_id}  - get a conversation command
    POST   /conversation/{conversation_id}/command  - create a conversation command
    DELETE /conversation/{conversation_id}/command/{command_id}  - cancel a conversation command

    POST   /conversation/{conversation_id}/dir/{path}  - create a new directory
    POST   /conversation/{conversation_id}/file/{path}  - create a new file (touch)
    POST   /conversation/{conversation_id}/upload/{parent_path}  - upload a set of files
    DELETE /conversation/{conversation_id}/file/{path}  - delete
    GET    /conversation/{conversation_id}/file-content/{path}
    GET    /conversation/{conversation_id}/file/{path}
    GET    /conversation/{conversation_id}/file-search
    GET    /conversation/{conversation_id}/file-count

    GET    /conversation/{conversation_id}/agent-info  - get agent info

    WS     /conversation/{conversation_id}  - connect to a conversation via websocket
    WS     /conversation/  - create a new conversation and connect to it via via websocket
    WS     /fire-hose  - fire hose of all events on the server
    """

    dynamic_types = DynamicTypes()

    @api.post("/conversation")
    async def create_conversation(
        agent_config: Annotated[Optional[AgentConfig], Body()] = None
    ) -> ConversationInfo:
        """Begin the process of creating a conversation"""
        return await conversation_broker.create_conversation(agent_config)

    @api.get("/conversation")
    async def search_conversations(
        page_id: Optional[str] = None,
    ) -> Page[ConversationInfo]:
        """Get a page of conversation info. Typically this is an admin only operation."""
        return await conversation_broker.search_conversations(page_id=page_id)

    @api.get("/conversation-count")
    async def count_conversations() -> Page[ConversationInfo]:
        """Count the number of conversations. Typically this is an admin only operation."""
        return await conversation_broker.count_conversations()

    @api.get("/conversation/{conversation_id}")
    async def get_conversation(conversation_id: UUID) -> Optional[ConversationInfo]:
        """Given an id, get conversation info. Return None if the conversation could not be found."""
        return await conversation_broker.get_conversation(conversation_id)

    @api.delete("/conversation/{conversation_id}")
    async def destroy_conversation(conversation_id: UUID) -> bool:
        """
        Begin the process of destroying a conversation. An attempt will be made to gracefully
        terminate any running commands within the conversation.
        """
        return await conversation_broker.destroy_conversation(conversation_id)

    @api.get("/conversation/{conversation_id}/event")
    async def search_events(
        conversation_id: UUID, page_id: Optional[str] = None
    ) -> Page[dynamic_types.get_event_info_class()]:  # type: ignore
        """Get events for a conversation."""
        conversation = await conversation_broker.get_conversation(conversation_id)
        page = await conversation.search_events(page_id=page_id)
        return page

    @api.get("/conversation/{conversation_id}/event-count")
    async def search_events(conversation_id: UUID) -> int:  # type: ignore
        """Get events for a conversation."""
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.count_events()
        return result

    @api.post("/conversation/{conversation_id}/event")
    async def trigger_event(
        conversation_id: UUID,
        detail: Annotated[dynamic_types.get_event_detail_type(), Body()],  # type: ignore
    ) -> dynamic_types.get_event_info_class():  # type: ignore
        """Trigger an event"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        event = await conversation.trigger_event(detail)
        return event

    @api.get("/conversation/{conversation_id}/event/{event_id}")
    async def get_event(conversation_id: UUID, event_id: UUID) -> Optional[dynamic_types.get_event_info_class()]:  # type: ignore
        """Get an event with the id given"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        event = await conversation.get_event(event_id)
        return event

    @api.get("/conversation/{conversation_id}/command")
    async def search_commands(
        conversation_id: UUID, page_id: Optional[str] = None
    ) -> Page[dynamic_types.get_command_info_class()]:  # type: ignore
        """Get commands for a conversation"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        page = await conversation.search_commands(page_id=page_id)
        return page

    @api.get("/conversation/{conversation_id}/command/{command_id}")
    async def get_command(conversation_id: UUID, command_id: UUID) -> Optional[dynamic_types.get_command_info_class()]:  # type: ignore
        """Get commands for a conversation"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        command = await conversation.get_command(command_id)
        return command

    @api.post("/conversation/{conversation_id}/command")
    async def create_command(
        conversation_id: UUID,
        create_command: Annotated[dynamic_types.get_create_command_class(), Body()],  # type: ignore
    ) -> dynamic_types.get_command_info_class():  # type: ignore
        """Given an id, get conversation info. Return None if the conversation could not be found."""
        conversation = await conversation_broker.get_conversation(conversation_id)
        command = await conversation.create_command(
            create_command.runnable, create_command.title, create_command.delay
        )
        return command

    @api.delete("/conversation/{conversation_id}/command/{command_id}")
    async def cancel_command(conversation_id: UUID, command_id: UUID) -> bool:
        """Given an id, get conversation info. Return None if the conversation could not be found."""
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.cancel_command(command_id)
        return result

    @api.post("/conversation/{conversation_id}/dir/{path}")
    async def create_dir(conversation_id: UUID, path: str) -> FileInfo:
        """
        Make the directory at the path given if it does not exist. Return info in the directory.
        Directories have the mime type `application/x-directory`
        """
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.create_dir(path)
        return result

    @api.post("/conversation/{conversation_id}/file/{path}")
    async def create_file(conversation_id: UUID, path: str) -> FileInfo:
        """Update the updated_at on the file at the path given. If no file exists, create an empty file"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.create_file(path)
        return result

    @api.post("/conversation/{conversation_id}/upload/{parent_path}")
    async def upload_files(
        conversation_id: UUID, parent_path: str, files: List[UploadFile]
    ) -> List[FileInfo]:
        """Upload files to the path given. Any existing file is overwritten"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        results = await wait_all(
            (
                conversation.save_file(f"{parent_path}/{file.filename}", file.file)
                for file in files
            ),
            FILE_UPLOAD_TIMEOUT,
        )
        return results

    @api.delete("/conversation/{conversation_id}/file/{path}")
    async def delete_file(conversation_id: UUID, path: str) -> bool:
        """Delete the file at the path given. Return True if the file existed and was deleted"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.delete_file(path)
        return result

    @api.get("/conversation/{conversation_id}/file-content/{path}")
    async def download_file(conversation_id: UUID, path: str) -> FileResponse:
        """Get the file at the path given. Directories are not downloadable. Return a download if the file was retrieved."""
        conversation = await conversation_broker.get_conversation(conversation_id)
        download = await conversation.load_file(path)
        if download is None:
            raise HTTPException(status_code=404)
        if download.download_url:
            return RedirectResponse(download.download_url)
        if download.path:
            return FileResponse(download.path, media_type=download.file_info.mime_type)
        if download.content_stream:
            return StreamingResponse(
                download.content_stream, media_type=download_file.mime_type
            )

    @api.get("/conversation/{conversation_id}/file/{path}")
    async def get_file_info(conversation_id: UUID, path: str) -> Optional[FileInfo]:
        """Get info on a file"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        result = await conversation.get_file_info(path)
        return result

    @api.get("/conversation/{conversation_id}/file-search")
    async def search_file_info(
        conversation_id: UUID,
        path_prefix: Optional[str] = None,
        path_delimiter: Optional[str] = "/",
        page_id: Optional[str] = None,
    ) -> Page[FileInfo]:
        """Search files available in the conversation"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        page = await conversation.search_file_info(
            FileFilter(path_prefix, path_delimiter), page_id
        )
        return page

    @api.get("/conversation/{conversation_id}/file-count")
    async def count_files(
        conversation_id: UUID,
        path_prefix: Optional[str] = None,
        path_delimiter: Optional[str] = None,
    ) -> int:
        """Count files available in the conversation"""
        conversation = await conversation_broker.get_conversation(conversation_id)
        page = await conversation.count_files(FileFilter(path_prefix, path_delimiter))
        return page

    @api.get("/conversation/{conversation_id}/file-count")
    async def get_agent_info(conversation_id: UUID) -> AgentInfo:
        conversation = await conversation_broker.get_conversation(conversation_id)
        return await conversation.get_agent_info()
        

    @api.get("/asyncapi.json")
    def get_async_schema() -> JSONResponse:
        return JSONResponse(
            {
                "asyncapi": "3.0.0",
                "info": {
                    "title": "OpenHands",
                    "version": "1.0.0",
                },
                "channels": {
                    "/": {
                        "address": "connect/{conversation_id}",
                        "parameters": {
                            "conversation_id": {
                                "type": "string",
                                "format": "uuid",
                                "description": "The UUID of the conversation.",
                            },
                        },
                        "messages": {
                            "Announcement": {
                                "name": "Announcement",
                                "payload": dynamic_types.get_event_info_type_adapter().json_schema(),
                            },
                        },
                        "operations": {
                            "CreateCommand": {
                                "name": "CreateCommand",
                                "payload": dynamic_types.get_runnable_type_adapter().json_schema(),
                            }
                        },
                    },
                },
            }
        )

    @api.websocket("/conversation")
    async def connect_and_create_conversation(
        websocket: WebSocket,
        agent_type: str,
        agent_llm: str,
        agent_key: str,
    ):
        await websocket.accept()
        agent_config = AgentConfig(agent_type, agent_llm, agent_key)
        conversation = await conversation_broker.create_conversation(agent_config)
        listener_id = await conversation.add_listener(
            WebsocketConversationListener(
                conversation.id, websocket, dynamic_types.get_event_info_type_adapter()
            )
        )
        try:
            while websocket.application_state == WebSocketState.CONNECTED:
                data = await websocket.receive_json()
                runnable = dynamic_types.get_runnable_type_adapter().validate_python(
                    data["runnable"]
                )
                await conversation.create_command(
                    runnable, data.get("title"), data.get("delay")
                )
        except WebSocketDisconnect as e:
            _LOGGER.debug("websocket_closed")
        finally:
            await conversation.remove_listener(listener_id)

    @api.websocket("/conversation/{conversation_id}")
    async def connect(
        conversation_id: Annotated[UUID, Path(title="The id of the conversation")],
        websocket: WebSocket,
    ):
        """Once a conversation is RUNNING, external agents can connect to it."""
        await asyncio.wait_for(websocket.accept(), GENERAL_TIMEOUT)
        conversation = await conversation_broker.get_conversation(conversation_id)
        listener_id = await conversation.add_listener(
            WebsocketConversationListener(
                conversation_id, websocket, dynamic_types.get_event_info_type_adapter()
            )
        )
        try:
            while websocket.application_state == WebSocketState.CONNECTED:
                data = await websocket.receive_json()
                runnable = dynamic_types.get_runnable_type_adapter().validate_python(
                    data["runnable"]
                )
                await conversation.create_command(
                    runnable, data.get("title"), data.get("delay")
                )
        except WebSocketDisconnect as e:
            _LOGGER.debug("websocket_closed")
        finally:
            await conversation.remove_listener(listener_id)

    @api.websocket("/fire-hose")
    async def fire_hose(websocket: WebSocket):
        """Listen for all events on all conversations. Typically this is an admin only operation."""
        await websocket.accept()
        listener = WebsocketConversationBrokerListener(
            websocket, dynamic_types.get_event_info_type_adapter()
        )
        page_id = None
        while True:
            page = await conversation_broker.search_conversations(page_id=page_id)
            conversations = await wait_all(
                conversation_broker.get_conversation(result.id)
                for result in page.results
            )
            await wait_all(
                listener.after_create_conversation(conversation)
                for conversation in conversations
            )
            page_id = page.next_page_id
            if not page_id:
                break
        listener_id = await conversation_broker.add_listener(listener)
        try:
            while websocket.application_state == WebSocketState.CONNECTED:
                # We don't send anything on the fire hose - just sleep
                await asyncio.sleep(1)
        except WebSocketDisconnect as e:
            _LOGGER.debug("websocket_closed")
        finally:
            await conversation_broker.remove_listener(listener_id)

    @api.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):

        exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
        # or logger.error(f'{exc}')
        _LOGGER.error(request, exc_str)
        content = {"status_code": 10422, "message": exc_str, "data": None}
        return JSONResponse(content=content, status_code=422)
