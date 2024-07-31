#!/usr/bin/env python3

import asyncio
import logging
import os
import re
from uuid import uuid4

import tornado
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tornado.escape import json_decode, json_encode, url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.ioloop import PeriodicCallback
from tornado.websocket import websocket_connect

logger = logging.getLogger(__name__)


def strip_ansi(o: str) -> str:
    """Removes ANSI escape sequences from `o`, as defined by ECMA-048 in
    http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf

    # https://github.com/ewen-lbh/python-strip-ansi/blob/master/strip_ansi/__init__.py

    >>> strip_ansi("\\033[33mLorem ipsum\\033[0m")
    'Lorem ipsum'

    >>> strip_ansi("Lorem \\033[38;25mIpsum\\033[0m sit\\namet.")
    'Lorem Ipsum sit\\namet.'

    >>> strip_ansi("")
    ''

    >>> strip_ansi("\\x1b[0m")
    ''

    >>> strip_ansi("Lorem")
    'Lorem'

    >>> strip_ansi('\\x1b[38;5;32mLorem ipsum\\x1b[0m')
    'Lorem ipsum'

    >>> strip_ansi('\\x1b[1m\\x1b[46m\\x1b[31mLorem dolor sit ipsum\\x1b[0m')
    'Lorem dolor sit ipsum'
    """
    # pattern = re.compile(r'/(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]/')
    pattern = re.compile(r'\x1B\[\d+(;\d+){0,2}m')
    stripped = pattern.sub('', o)
    return stripped


class JupyterKernel:
    def __init__(self, url_suffix, convid, lang='python'):
        self.base_url = f'http://{url_suffix}'
        self.base_ws_url = f'ws://{url_suffix}'
        self.lang = lang
        self.kernel_id = None
        self.ws = None
        self.convid = convid
        logger.debug(
            f'Jupyter kernel created for conversation {convid} at {url_suffix}'
        )

        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None
        self.initialized = False

    async def initialize(self):
        await self.execute(r'%colors nocolor')
        # pre-defined tools
        self.tools_to_run = [
            # TODO: You can add code for your pre-defined tools here
        ]
        if os.path.exists('/opendevin/plugins/agent_skills/agentskills.py'):
            self.tools_to_run.append('from agentskills import *')
        for tool in self.tools_to_run:
            res = await self.execute(tool)
            logger.debug(f'Tool [{tool}] initialized:\n{res}')
        self.initialized = True

    async def _send_heartbeat(self):
        if not self.ws:
            return
        try:
            self.ws.ping()
            # logging.info('Heartbeat sent...')
        except tornado.iostream.StreamClosedError:
            # logging.info('Heartbeat failed, reconnecting...')
            try:
                await self._connect()
            except ConnectionRefusedError:
                logger.debug(
                    'ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?'
                )

    async def _connect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 5
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        '{}/api/kernels'.format(self.base_url),
                        method='POST',
                        body=json_encode({'name': self.lang}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel['id']
                    break
                except Exception:
                    # kernels are not ready yet
                    n_tries -= 1
                    await asyncio.sleep(1)

            if n_tries == 0:
                raise ConnectionRefusedError('Failed to connect to kernel')

        ws_req = HTTPRequest(
            url='{}/api/kernels/{}/channels'.format(
                self.base_ws_url, url_escape(self.kernel_id)
            )
        )
        self.ws = await websocket_connect(ws_req)
        logger.debug('Connected to kernel websocket')

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(
            self._send_heartbeat, self.heartbeat_interval
        )
        self.heartbeat_callback.start()

    @retry(
        retry=retry_if_exception_type(ConnectionRefusedError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
    )
    async def execute(self, code, timeout=120):
        if not self.ws:
            await self._connect()

        msg_id = uuid4().hex
        assert self.ws is not None
        res = await self.ws.write_message(
            json_encode(
                {
                    'header': {
                        'username': '',
                        'version': '5.0',
                        'session': '',
                        'msg_id': msg_id,
                        'msg_type': 'execute_request',
                    },
                    'parent_header': {},
                    'channel': 'shell',
                    'content': {
                        'code': code,
                        'silent': False,
                        'store_history': False,
                        'user_expressions': {},
                        'allow_stdin': False,
                    },
                    'metadata': {},
                    'buffers': {},
                }
            )
        )
        logger.debug(f'Executed code in jupyter kernel:\n{res}')

        outputs = []

        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                assert self.ws is not None
                msg = await self.ws.read_message()
                msg = json_decode(msg)
                msg_type = msg['msg_type']
                parent_msg_id = msg['parent_header'].get('msg_id', None)

                if parent_msg_id != msg_id:
                    continue

                if os.environ.get('DEBUG'):
                    logger.debug(
                        f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}"
                    )

                if msg_type == 'error':
                    traceback = '\n'.join(msg['content']['traceback'])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == 'stream':
                    outputs.append(msg['content']['text'])
                elif msg_type in ['execute_result', 'display_data']:
                    outputs.append(msg['content']['data']['text/plain'])
                    if 'image/png' in msg['content']['data']:
                        # use markdone to display image (in case of large image)
                        outputs.append(
                            f"\n![image](data:image/png;base64,{msg['content']['data']['image/png']})\n"
                        )

                elif msg_type == 'execute_reply':
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f'{self.base_url}/api/kernels/{self.kernel_id}/interrupt',
                method='POST',
                body=json_encode({'kernel_id': self.kernel_id}),
            )
            logger.debug(f'Kernel interrupted: {interrupt_response}')

        try:
            execution_done = await asyncio.wait_for(wait_for_messages(), timeout)
        except asyncio.TimeoutError:
            await interrupt_kernel()
            return f'[Execution timed out ({timeout} seconds).]'

        if not outputs and execution_done:
            ret = '[Code executed successfully with no output]'
        else:
            ret = ''.join(outputs)

        # Remove ANSI
        ret = strip_ansi(ret)

        if os.environ.get('DEBUG'):
            logger.debug(f'OUTPUT:\n{ret}')
        return ret

    async def shutdown_async(self):
        if self.kernel_id:
            client = AsyncHTTPClient()
            await client.fetch(
                '{}/api/kernels/{}'.format(self.base_url, self.kernel_id),
                method='DELETE',
            )
            self.kernel_id = None
            if self.ws:
                self.ws.close()
                self.ws = None


class ExecuteHandler(tornado.web.RequestHandler):
    def initialize(self, jupyter_kernel):
        self.jupyter_kernel = jupyter_kernel

    async def post(self):
        data = json_decode(self.request.body)
        code = data.get('code')

        if not code:
            self.set_status(400)
            self.write('Missing code')
            return

        output = await self.jupyter_kernel.execute(code)

        self.write(output)


def make_app():
    jupyter_kernel = JupyterKernel(
        f"localhost:{os.environ.get('JUPYTER_GATEWAY_PORT')}",
        os.environ.get('JUPYTER_GATEWAY_KERNEL_ID'),
    )
    asyncio.get_event_loop().run_until_complete(jupyter_kernel.initialize())

    return tornado.web.Application(
        [
            (r'/execute', ExecuteHandler, {'jupyter_kernel': jupyter_kernel}),
        ]
    )


if __name__ == '__main__':
    app = make_app()
    app.listen(os.environ.get('JUPYTER_EXEC_SERVER_PORT'))
    tornado.ioloop.IOLoop.current().start()
