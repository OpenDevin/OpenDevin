import uuid
from pathlib import Path

import litellm
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles

import agenthub  # noqa F401 (we import this to get the agents registered)
from opendevin.controller.agent import Agent
from opendevin.core.config import config
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import AgentState  # Add this import
from opendevin.events.action import ChangeAgentStateAction, NullAction
from opendevin.events.observation import AgentStateChangedObservation, NullObservation
from opendevin.events.serialization import event_to_dict
from opendevin.llm import bedrock
from opendevin.server.auth import get_sid_from_token, sign_token
from opendevin.server.data_models.feedback import FeedbackDataModel, store_feedback
from opendevin.server.session import session_manager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3001'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

security_scheme = HTTPBearer()


@app.middleware('http')
async def attach_session(request: Request, call_next):
    """
    Middleware to attach session information to the request.

    This middleware checks for the Authorization header, validates the token,
    and attaches the corresponding session to the request state.

    Args:
        request (Request): The incoming request object.
        call_next (Callable): The next middleware or route handler in the chain.

    Returns:
        Response: The response from the next middleware or route handler.
    """
    if request.url.path.startswith('/api/options/') or not request.url.path.startswith(
        '/api/'
    ):
        response = await call_next(request)
        return response

    if not request.headers.get('Authorization'):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Missing Authorization header'},
        )

    auth_token = request.headers.get('Authorization')
    if 'Bearer' in auth_token:
        auth_token = auth_token.split('Bearer')[1].strip()

    request.state.sid = get_sid_from_token(auth_token)
    if request.state.sid == '':
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Invalid token'},
        )

    request.state.session = session_manager.get_session(request.state.sid)
    if request.state.session is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': 'Session not found'},
        )

    response = await call_next(request)
    return response


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving events from the client (i.e., the browser).
    WebSocket endpoint for real-time communication.
    Once connected, you can send various actions:
    This function handles the WebSocket connection, including authentication,
    - Initialize the agent:
    session management, and event streaming.
        ```json
        {"action": "initialize", "args": {"LLM_MODEL": "ollama/llama3", "AGENT": "CodeActAgent", "LANGUAGE": "en", "LLM_API_KEY": "ollama"}}
    Args:
        ```
        websocket (WebSocket): The WebSocket connection object.
    - Start a new development task:
        ```json
        {"action": "start", "args": {"task": "write a bash script that prints hello"}}
        ```
    - Send a message:
        ```json
        {"action": "message", "args": {"content": "Hello, how are you?"}}
        ```
    - Write contents to a file:
        ```json
        {"action": "write", "args": {"path": "./greetings.txt", "content": "Hello, OpenDevin?"}}
        ```
    - Read the contents of a file:
        ```json
        {"action": "read", "args": {"path": "./greetings.txt"}}
        ```
    - Run a command:
        ```json
        {"action": "run", "args": {"command": "ls -l", "background": false, "thought": ""}}
        ```
    - Run an IPython command:
        ```json
        {"action": "run_ipython", "args": {"command": "print('Hello, IPython!')"}}
        ```
    - Kill a background command:
        ```json
        {"action": "kill", "args": {"id": "command_id"}}
        ```
    - Open a web page:
        ```json
        {"action": "browse", "args": {"url": "https://arxiv.org/html/2402.01030v2"}}
        ```
    - Search long-term memory:
        ```json
        {"action": "recall", "args": {"query": "past projects"}}
        ```
    - Add a task to the root_task:
        ```json
        {"action": "add_task", "args": {"task": "Implement feature X"}}
        ```
    - Update a task in the root_task:
        ```json
        {"action": "modify_task", "args": {"id": "0", "state": "in_progress", "thought": ""}}
        ```
    - Change the agent's state:
        ```json
        {"action": "change_agent_state", "args": {"state": "paused"}}
        ```
    - Finish the task:
        ```json
        {"action": "finish", "args": {}}
        ```
    """
    await websocket.accept()

    session = None
    if websocket.query_params.get('token'):
        token = websocket.query_params.get('token')
        sid = get_sid_from_token(token)

        if sid == '':
            await websocket.send_json({'error': 'Invalid token', 'error_code': 401})
            await websocket.close()
            return
    else:
        sid = str(uuid.uuid4())
        token = sign_token({'sid': sid})

    session = session_manager.add_or_restart_session(sid, websocket)
    await websocket.send_json({'token': token, 'status': 'ok'})

    latest_event_id = -1
    if websocket.query_params.get('latest_event_id'):
        latest_event_id = int(websocket.query_params.get('latest_event_id'))
    for event in session.agent_session.event_stream.get_events(
        start_id=latest_event_id + 1
    ):
        if isinstance(
            event,
            (
                NullAction,
                NullObservation,
                ChangeAgentStateAction,
                AgentStateChangedObservation,
            ),
        ):
            continue
        await websocket.send_json(event_to_dict(event))

    await session.loop_recv()


@app.get('/api/options/models')
async def get_litellm_models():
    """
    Get all models supported by LiteLLM.

    This function combines models from litellm and Bedrock, removing any
    error-prone Bedrock models.

    To get the models:
    ```sh
    curl http://localhost:3000/api/litellm-models
    ```

    Returns:
        list: A sorted list of unique model names.
    """
    litellm_model_list = litellm.model_list + list(litellm.model_cost.keys())
    litellm_model_list_without_bedrock = bedrock.remove_error_modelId(
        litellm_model_list
    )
    bedrock_model_list = bedrock.list_foundation_models()
    model_list = litellm_model_list_without_bedrock + bedrock_model_list

    return list(sorted(set(model_list)))


@app.get('/api/options/agents')
async def get_agents():
    """
    Get all agents supported by LiteLLM.

    To get the agents:
    ```sh
    curl http://localhost:3000/api/agents
    ```

    Returns:
        list: A sorted list of agent names.
    """
    agents = sorted(Agent.list_agents())
    return agents


@app.get('/api/list-files')
def list_files(request: Request, path: str = '/'):
    """
    List files in the specified path.

    This function retrieves a list of files from the agent's runtime file store,
    excluding certain system and hidden files/directories.

    To list files:
    ```sh
    curl http://localhost:3000/api/list-files
    ```

    Args:
        request (Request): The incoming request object.
        path (str, optional): The path to list files from. Defaults to '/'.

    Returns:
        list: A list of file names in the specified path.

    Raises:
        HTTPException: If there's an error listing the files.
    """
    if not request.state.session.agent_session.runtime:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': 'Runtime not yet initialized'},
        )

    exclude_list = (
        '.git',
        '.DS_Store',
        '.svn',
        '.hg',
        '.idea',
        '.vscode',
        '.settings',
        '.pytest_cache',
        '__pycache__',
        'node_modules',
        'vendor',
        'build',
        'dist',
        'bin',
        'logs',
        'log',
        'tmp',
        'temp',
        'coverage',
        'venv',
        'env',
    )

    try:
        entries = request.state.session.agent_session.runtime.file_store.list(path)

        # Filter entries, excluding special folders
        if entries:
            return [
                entry
                for entry in entries
                if Path(entry).parts and Path(entry).parts[-1] not in exclude_list
            ]
        return []
    except Exception as e:
        logger.error(f'Error refreshing files: {e}', exc_info=False)
        error_msg = f'Error refreshing files: {e}'
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': error_msg},
        )


@app.get('/api/select-file')
def select_file(file: str, request: Request):
    """
    Retrieve the content of a specified file.

    To select a file:
    ```sh
    curl http://localhost:3000/api/select-file?file=<file_path>
    ```

    Args:
        file (str): The path of the file to be retrieved.
        request (Request): The incoming request object.

    Returns:
        dict: A dictionary containing the file content.

    Raises:
        HTTPException: If there's an error opening the file.
    """
    try:
        content = request.state.session.agent_session.runtime.file_store.read(file)
    except Exception as e:
        logger.error(f'Error opening file {file}: {e}', exc_info=False)
        error_msg = f'Error opening file: {e}'
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': error_msg},
        )
    return {'code': content}


@app.post('/api/upload-files')
async def upload_file(request: Request, files: list[UploadFile]):
    """
    Upload a list of files to the workspace.

    To upload a files:
    ```sh
    curl -X POST -F "file=@<file_path1>" -F "file=@<file_path2>" http://localhost:3000/api/upload-files
    ```

    Args:
        request (Request): The incoming request object.
        files (list[UploadFile]): A list of files to be uploaded.

    Returns:
        dict: A message indicating the success of the upload operation.

    Raises:
        HTTPException: If there's an error saving the files.
    """
    try:
        for file in files:
            file_contents = await file.read()
            request.state.session.agent_session.runtime.file_store.write(
                file.filename, file_contents
            )
    except Exception as e:
        logger.error(f'Error saving files: {e}', exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error saving file:s {e}'},
        )
    return {'message': 'Files uploaded successfully', 'file_count': len(files)}


@app.post('/api/submit-feedback')
async def submit_feedback(request: Request, feedback: FeedbackDataModel):
    """
    Submit user feedback.

    This function stores the provided feedback data.

    To submit feedback:
    ```sh
    curl -X POST -F "email=test@example.com" -F "token=abc" -F "feedback=positive" -F "permissions=private" -F "trajectory={}" http://localhost:3000/api/submit-feedback
    ```

    Args:
        request (Request): The incoming request object.
        feedback (FeedbackDataModel): The feedback data to be stored.

    Returns:
        dict: The stored feedback data.

    Raises:
        HTTPException: If there's an error submitting the feedback.
    """
    # Assuming the storage service is already configured in the backend
    # and there is a function to handle the storage.
    try:
        feedback_data = store_feedback(feedback)
        return JSONResponse(status_code=200, content=feedback_data)
    except Exception as e:
        logger.error(f'Error submitting feedback: {e}')
        return JSONResponse(
            status_code=500, content={'error': 'Failed to submit feedback'}
        )


@app.get('/api/root_task')
def get_root_task(request: Request):
    """
    Retrieve the root task of the current agent session.

    To get the root_task:
    ```sh
    curl -H "Authorization: Bearer <TOKEN>" http://localhost:3000/api/root_task
    ```

    Args:
        request (Request): The incoming request object.

    Returns:
        dict: The root task data if available.

    Raises:
        HTTPException: If the root task is not available.
    """
    controller = request.state.session.agent_session.controller
    if controller is not None:
        state = controller.get_state()
        if state:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=state.root_task.to_dict(),
            )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get('/api/defaults')
async def appconfig_defaults():
    """
    Retrieve the default configuration settings.

    To get the default configurations:
    ```sh
    curl http://localhost:3000/api/defaults
    ```

    Returns:
        dict: The default configuration settings.
    """
    return config.defaults_dict


@app.post('/api/save-file')
async def save_file(request: Request):
    """
    Save a file to the agent's runtime file store.

    This endpoint allows saving a file when the agent is in a paused, finished,
    or awaiting user input state. It checks the agent's state before proceeding
    with the file save operation.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        JSONResponse: A JSON response indicating the success of the operation.

    Raises:
        HTTPException:
            - 403 error if the agent is not in an allowed state for editing.
            - 400 error if the file path or content is missing.
            - 500 error if there's an unexpected error during the save operation.
    """
    try:
        # Get the agent's current state
        controller = request.state.session.agent_session.controller
        agent_state = controller.get_agent_state()

        # Check if the agent is in an allowed state for editing
        if agent_state not in [
            AgentState.INIT,
            AgentState.PAUSED,
            AgentState.FINISHED,
            AgentState.AWAITING_USER_INPUT,
        ]:
            raise HTTPException(
                status_code=403,
                detail='Code editing is only allowed when the agent is paused, finished, or awaiting user input',
            )

        # Extract file path and content from the request
        data = await request.json()
        file_path = data.get('filePath')
        content = data.get('content')

        # Validate the presence of required data
        if not file_path or content is None:
            raise HTTPException(status_code=400, detail='Missing filePath or content')

        # Save the file to the agent's runtime file store
        request.state.session.agent_session.runtime.file_store.write(file_path, content)

        # Return a success response
        return JSONResponse(
            status_code=200, content={'message': 'File saved successfully'}
        )
    except Exception as e:
        # Log the error and return a 500 response
        logger.error(f'Error saving file: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Error saving file: {e}')


app.mount('/', StaticFiles(directory='./frontend/dist', html=True), name='dist')
