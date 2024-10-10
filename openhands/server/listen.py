import asyncio
import io
import os
import re
import tempfile
import uuid
import warnings
from contextlib import asynccontextmanager

import requests
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

from openhands.security.options import SecurityAnalyzers
from openhands.server.data_models.feedback import FeedbackDataModel, store_feedback
from openhands.storage import get_file_store

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import openhands.agenthub  # noqa F401 (we import this to get the agents registered)
from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig, load_app_config
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    ChangeAgentStateAction,
    FileReadAction,
    FileWriteAction,
    NullAction,
)
from openhands.events.observation import (
    AgentStateChangedObservation,
    ErrorObservation,
    FileReadObservation,
    FileWriteObservation,
    NullObservation,
)
from openhands.events.serialization import event_to_dict
from openhands.llm import bedrock
from openhands.runtime.runtime import Runtime
from openhands.server.auth import get_sid_from_token, sign_token
from openhands.server.session import SessionManager

load_dotenv()

config = load_app_config()
file_store = get_file_store(config.file_store, config.file_store_path)
session_manager = SessionManager(config, file_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_manager
    async with session_manager:
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3001', 'http://127.0.0.1:3001'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

security_scheme = HTTPBearer()


def load_file_upload_config() -> tuple[int, bool, list[str]]:
    """Load file upload configuration from the config object.

    This function retrieves the file upload settings from the global config object.
    It handles the following settings:
    - Maximum file size for uploads
    - Whether to restrict file types
    - List of allowed file extensions

    It also performs sanity checks on the values to ensure they are valid and safe.

    Returns:
        tuple: A tuple containing:
            - max_file_size_mb (int): Maximum file size in MB. 0 means no limit.
            - restrict_file_types (bool): Whether file type restrictions are enabled.
            - allowed_extensions (set): Set of allowed file extensions.
    """
    # Retrieve values from config
    max_file_size_mb = config.file_uploads_max_file_size_mb
    restrict_file_types = config.file_uploads_restrict_file_types
    allowed_extensions = config.file_uploads_allowed_extensions

    # Sanity check for max_file_size_mb
    if not isinstance(max_file_size_mb, int) or max_file_size_mb < 0:
        logger.warning(
            f'Invalid max_file_size_mb: {max_file_size_mb}. Setting to 0 (no limit).'
        )
        max_file_size_mb = 0

    # Sanity check for allowed_extensions
    if not isinstance(allowed_extensions, (list, set)) or not allowed_extensions:
        logger.warning(
            f'Invalid allowed_extensions: {allowed_extensions}. Setting to [".*"].'
        )
        allowed_extensions = ['.*']
    else:
        # Ensure all extensions start with a dot and are lowercase
        allowed_extensions = [
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in allowed_extensions
        ]

    # If restrictions are disabled, allow all
    if not restrict_file_types:
        allowed_extensions = ['.*']

    logger.debug(
        f'File upload config: max_size={max_file_size_mb}MB, '
        f'restrict_types={restrict_file_types}, '
        f'allowed_extensions={allowed_extensions}'
    )

    return max_file_size_mb, restrict_file_types, allowed_extensions


# Load configuration
MAX_FILE_SIZE_MB, RESTRICT_FILE_TYPES, ALLOWED_EXTENSIONS = load_file_upload_config()


def is_extension_allowed(filename):
    """Check if the file extension is allowed based on the current configuration.

    This function supports wildcards and files without extensions.
    The check is case-insensitive for extensions.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    if not RESTRICT_FILE_TYPES:
        return True

    file_ext = os.path.splitext(filename)[1].lower()  # Convert to lowercase
    return (
        '.*' in ALLOWED_EXTENSIONS
        or file_ext in (ext.lower() for ext in ALLOWED_EXTENSIONS)
        or (file_ext == '' and '.' in ALLOWED_EXTENSIONS)
    )


@app.middleware('http')
async def attach_session(request: Request, call_next):
    """Middleware to attach session information to the request.

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

    # Bypass authentication for OPTIONS requests (preflight)
    if request.method == 'OPTIONS':
        response = await call_next(request)
        return response

    # For all other methods, validate the Authorization header
    if not request.headers.get('Authorization'):
        logger.warning('Missing Authorization header')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Missing Authorization header'},
        )

    auth_token = request.headers.get('Authorization')
    if 'Bearer' in auth_token:
        auth_token = auth_token.split('Bearer')[1].strip()

    request.state.sid = get_sid_from_token(auth_token, config.jwt_secret)
    if request.state.sid == '':
        logger.warning('Invalid token')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Invalid token'},
        )

    request.state.conversation = session_manager.attach_to_conversation(
        request.state.sid
    )
    if request.state.conversation is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': 'Session not found'},
        )

    response = await call_next(request)
    return response


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving events from the client (i.e., the browser).
    Once connected, the client can send various actions:
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
        {"action": "message", "args": {"content": "Hello, how are you?", "images_urls": ["base64_url1", "base64_url2"]}}
        ```
    - Write contents to a file:
        ```json
        {"action": "write", "args": {"path": "./greetings.txt", "content": "Hello, OpenHands?"}}
        ```
    - Read the contents of a file:
        ```json
        {"action": "read", "args": {"path": "./greetings.txt"}}
        ```
    - Run a command:
        ```json
        {"action": "run", "args": {"command": "ls -l", "thought": "", "is_confirmed": "confirmed"}}
        ```
    - Run an IPython command:
        ```json
        {"action": "run_ipython", "args": {"command": "print('Hello, IPython!')"}}
        ```
    - Open a web page:
        ```json
        {"action": "browse", "args": {"url": "https://arxiv.org/html/2402.01030v2"}}
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
    await asyncio.wait_for(websocket.accept(), 10)

    if websocket.query_params.get('token'):
        token = websocket.query_params.get('token')
        sid = get_sid_from_token(token, config.jwt_secret)

        if sid == '':
            await websocket.send_json({'error': 'Invalid token', 'error_code': 401})
            await websocket.close()
            return
    else:
        sid = str(uuid.uuid4())
        token = sign_token({'sid': sid}, config.jwt_secret)

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
async def get_litellm_models() -> list[str]:
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
    # TODO: for bedrock, this is using the default config
    llm_config: LLMConfig = config.get_llm_config()
    bedrock_model_list = []
    if (
        llm_config.aws_region_name
        and llm_config.aws_access_key_id
        and llm_config.aws_secret_access_key
    ):
        bedrock_model_list = bedrock.list_foundation_models(
            llm_config.aws_region_name,
            llm_config.aws_access_key_id,
            llm_config.aws_secret_access_key,
        )
    model_list = litellm_model_list_without_bedrock + bedrock_model_list
    for llm_config in config.llms.values():
        ollama_base_url = llm_config.ollama_base_url
        if llm_config.model.startswith('ollama'):
            if not ollama_base_url:
                ollama_base_url = llm_config.base_url
        if ollama_base_url:
            ollama_url = ollama_base_url.strip('/') + '/api/tags'
            try:
                ollama_models_list = requests.get(ollama_url, timeout=3).json()[
                    'models'
                ]
                for model in ollama_models_list:
                    model_list.append('ollama/' + model['name'])
                break
            except requests.exceptions.RequestException as e:
                logger.error(f'Error getting OLLAMA models: {e}', exc_info=True)

    return list(sorted(set(model_list)))


@app.get('/api/options/agents')
async def get_agents():
    """Get all agents supported by LiteLLM.

    To get the agents:
    ```sh
    curl http://localhost:3000/api/agents
    ```

    Returns:
        list: A sorted list of agent names.
    """
    agents = sorted(Agent.list_agents())
    return agents


@app.get('/api/options/security-analyzers')
async def get_security_analyzers():
    """Get all supported security analyzers.

    To get the security analyzers:
    ```sh
    curl http://localhost:3000/api/security-analyzers
    ```

    Returns:
        list: A sorted list of security analyzer names.
    """
    return sorted(SecurityAnalyzers.keys())


FILES_TO_IGNORE = [
    '.git/',
    '.DS_Store',
    'node_modules/',
    '__pycache__/',
]


@app.get('/api/list-files')
async def list_files(request: Request, path: str | None = None):
    """List files in the specified path.

    This function retrieves a list of files from the agent's runtime file store,
    excluding certain system and hidden files/directories.

    To list files:
    ```sh
    curl http://localhost:3000/api/list-files
    ```

    Args:
        request (Request): The incoming request object.
        path (str, optional): The path to list files from. Defaults to None.

    Returns:
        list: A list of file names in the specified path.

    Raises:
        HTTPException: If there's an error listing the files.
    """
    if not request.state.conversation.runtime:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={'error': 'Runtime not yet initialized'},
        )
    runtime: Runtime = request.state.conversation.runtime
    file_list = await asyncio.get_event_loop().run_in_executor(
        None, runtime.list_files, path
    )
    if path:
        file_list = [os.path.join(path, f) for f in file_list]

    file_list = [f for f in file_list if f not in FILES_TO_IGNORE]

    def filter_for_gitignore(file_list, base_path):
        gitignore_path = os.path.join(base_path, '.gitignore')
        try:
            read_action = FileReadAction(gitignore_path)
            observation = runtime.run_action(read_action)
            spec = PathSpec.from_lines(
                GitWildMatchPattern, observation.content.splitlines()
            )
        except Exception as e:
            print(e)
            return file_list
        file_list = [entry for entry in file_list if not spec.match_file(entry)]
        return file_list

    file_list = filter_for_gitignore(file_list, '')

    return file_list


@app.get('/api/select-file')
async def select_file(file: str, request: Request):
    """Retrieve the content of a specified file.

    To select a file:
    ```sh
    curl http://localhost:3000/api/select-file?file=<file_path>
    ```

    Args:
        file (str): The path of the file to be retrieved.
            Expect path to be absolute inside the runtime.
        request (Request): The incoming request object.

    Returns:
        dict: A dictionary containing the file content.

    Raises:
        HTTPException: If there's an error opening the file.
    """
    runtime: Runtime = request.state.conversation.runtime

    file = os.path.join(runtime.config.workspace_mount_path_in_sandbox, file)
    read_action = FileReadAction(file)
    observation = await runtime.async_run_action(read_action)

    if isinstance(observation, FileReadObservation):
        content = observation.content
        return {'code': content}
    elif isinstance(observation, ErrorObservation):
        logger.error(f'Error opening file {file}: {observation}', exc_info=False)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={'error': f'Error opening file: {observation}'},
        )


def sanitize_filename(filename):
    """Sanitize the filename to prevent directory traversal"""
    # Remove any directory components
    filename = os.path.basename(filename)
    # Remove any non-alphanumeric characters except for .-_
    filename = re.sub(r'[^\w\-_\.]', '', filename)
    # Limit the filename length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[: max_length - len(ext)] + ext
    return filename


@app.post('/api/upload-files')
async def upload_file(request: Request, files: list[UploadFile]):
    """Upload a list of files to the workspace.

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
        uploaded_files = []
        skipped_files = []
        for file in files:
            safe_filename = sanitize_filename(file.filename)
            file_contents = await file.read()

            if (
                MAX_FILE_SIZE_MB > 0
                and len(file_contents) > MAX_FILE_SIZE_MB * 1024 * 1024
            ):
                skipped_files.append(
                    {
                        'name': safe_filename,
                        'reason': f'Exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB',
                    }
                )
                continue

            if not is_extension_allowed(safe_filename):
                skipped_files.append(
                    {'name': safe_filename, 'reason': 'File type not allowed'}
                )
                continue

            # copy the file to the runtime
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file_path = os.path.join(tmp_dir, safe_filename)
                with open(tmp_file_path, 'wb') as tmp_file:
                    tmp_file.write(file_contents)
                    tmp_file.flush()

                runtime: Runtime = request.state.conversation.runtime
                runtime.copy_to(
                    tmp_file_path, runtime.config.workspace_mount_path_in_sandbox
                )
            uploaded_files.append(safe_filename)

        response_content = {
            'message': 'File upload process completed',
            'uploaded_files': uploaded_files,
            'skipped_files': skipped_files,
        }

        if not uploaded_files and skipped_files:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    **response_content,
                    'error': 'No files were uploaded successfully',
                },
            )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response_content)

    except Exception as e:
        logger.error(f'Error during file upload: {e}', exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                'error': f'Error during file upload: {str(e)}',
                'uploaded_files': [],
                'skipped_files': [],
            },
        )


@app.post('/api/submit-feedback')
async def submit_feedback(request: Request, feedback: FeedbackDataModel):
    """Submit user feedback.

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


@app.get('/api/defaults')
async def appconfig_defaults():
    """Retrieve the default configuration settings.

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
    """Save a file to the agent's runtime file store.

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
        # Extract file path and content from the request
        data = await request.json()
        file_path = data.get('filePath')
        content = data.get('content')

        # Validate the presence of required data
        if not file_path or content is None:
            raise HTTPException(status_code=400, detail='Missing filePath or content')

        # Save the file to the agent's runtime file store
        runtime: Runtime = request.state.conversation.runtime
        file_path = os.path.join(
            runtime.config.workspace_mount_path_in_sandbox, file_path
        )
        write_action = FileWriteAction(file_path, content)
        observation = await runtime.async_run_action(write_action)

        if isinstance(observation, FileWriteObservation):
            return JSONResponse(
                status_code=200, content={'message': 'File saved successfully'}
            )
        elif isinstance(observation, ErrorObservation):
            return JSONResponse(
                status_code=500,
                content={'error': f'Failed to save file: {observation}'},
            )
        else:
            return JSONResponse(
                status_code=500,
                content={'error': f'Unexpected observation: {observation}'},
            )
    except Exception as e:
        # Log the error and return a 500 response
        logger.error(f'Error saving file: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Error saving file: {e}')


@app.route('/api/security/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def security_api(request: Request):
    """Catch-all route for security analyzer API requests.

    Each request is handled directly to the security analyzer.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        Any: The response from the security analyzer.

    Raises:
        HTTPException: If the security analyzer is not initialized.
    """
    if not request.state.conversation.security_analyzer:
        raise HTTPException(status_code=404, detail='Security analyzer not initialized')

    return await request.state.conversation.security_analyzer.handle_api_request(
        request
    )


@app.get('/api/zip-directory')
async def zip_current_workspace(request: Request):
    try:
        logger.info('Zipping workspace')
        runtime: Runtime = request.state.conversation.runtime

        path = runtime.config.workspace_mount_path_in_sandbox
        zip_file_bytes = runtime.copy_from(path)
        zip_stream = io.BytesIO(zip_file_bytes)  # Wrap to behave like a file stream
        response = StreamingResponse(
            zip_stream,
            media_type='application/x-zip-compressed',
            headers={'Content-Disposition': 'attachment; filename=workspace.zip'},
        )

        return response
    except Exception as e:
        logger.error(f'Error zipping workspace: {e}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail='Failed to zip workspace',
        )


class AuthCode(BaseModel):
    code: str


@app.post('/github/callback')
def github_callback(auth_code: AuthCode):
    # Prepare data for the token exchange request
    data = {
        'client_id': os.getenv('GITHUB_CLIENT_ID'),
        'client_secret': os.getenv('GITHUB_CLIENT_SECRET'),
        'code': auth_code.code,
    }

    logger.info(f'Exchanging code for token: {data}')

    headers = {'Accept': 'application/json'}
    response = requests.post(
        'https://github.com/login/oauth/access_token', data=data, headers=headers
    )

    if response.status_code != 200:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'Failed to exchange code for token'},
        )

    token_response = response.json()

    if 'access_token' not in token_response:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={'error': 'No access token in response'},
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={'access_token': token_response['access_token']},
    )


app.mount('/', StaticFiles(directory='./frontend/build', html=True), name='dist')
