import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Literal, Mapping

from termcolor import colored

DISABLE_COLOR_PRINTING = False
DEBUG = os.getenv('DEBUG', 'False').lower() in ['true', '1', 'yes']

ColorType = Literal[
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'light_grey',
    'dark_grey',
    'light_red',
    'light_green',
    'light_yellow',
    'light_blue',
    'light_magenta',
    'light_cyan',
    'white',
]

LOG_COLORS: Mapping[str, ColorType] = {
    'ACTION': 'green',
    'USER_ACTION': 'light_red',
    'OBSERVATION': 'yellow',
    'USER_OBSERVATION': 'light_green',
    'DETAIL': 'cyan',
    'ERROR': 'red',
    'PLAN': 'light_magenta',
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg_type = record.__dict__.get('msg_type')
        event_source = record.__dict__.get('event_source')
        if event_source:
            new_msg_type = f'{event_source.upper()}_{msg_type}'
            if new_msg_type in LOG_COLORS:
                msg_type = new_msg_type
        if msg_type in LOG_COLORS and not DISABLE_COLOR_PRINTING:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type in ['ERROR'] or DEBUG:
                return f'{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}'
            return f'{time_str} - {msg_type_color}\n{msg}'
        elif msg_type == 'STEP':
            msg = '\n\n==============\n' + record.msg + '\n'
            return f'{msg}'
        return super().format(record)


console_formatter = ColoredFormatter(
    '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)

file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)
llm_formatter = logging.Formatter('%(message)s')


class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        # start with attributes
        sensitive_patterns = [
            'api_key',
            'aws_access_key_id',
            'aws_secret_access_key',
            'e2b_api_key',
            'github_token',
            'jwt_secret',
            'ssh_password',
            'password',
        ]

        # add env var names
        env_vars = [attr.upper() for attr in sensitive_patterns]
        sensitive_patterns.extend(env_vars)

        # and some special cases
        sensitive_patterns.append('JWT_SECRET')
        sensitive_patterns.append('LLM_API_KEY')
        sensitive_patterns.append('GITHUB_TOKEN')
        sensitive_patterns.append('SANDBOX_ENV_GITHUB_TOKEN')

        # this also formats the message with % args
        msg = record.getMessage()
        record.args = ()

        for attr in sensitive_patterns:
            pattern = rf"{attr}='?([\w-]+)'?"
            msg = re.sub(pattern, f"{attr}='******'", msg)

        # passed with msg
        record.msg = msg
        return True


def get_console_handler():
    """Returns a console handler for logging."""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    if DEBUG:
        console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    return console_handler


def get_file_handler(log_dir):
    """Returns a file handler for logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_name = f'opendevin_{timestamp}.log'
    file_handler = logging.FileHandler(os.path.join(log_dir, file_name))
    if DEBUG:
        file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    return file_handler


# Set up logging
logging.basicConfig(level=logging.ERROR)


def log_uncaught_exceptions(ex_cls, ex, tb):
    """Logs uncaught exceptions along with the traceback.

    Args:
        ex_cls (type): The type of the exception.
        ex (Exception): The exception instance.
        tb (traceback): The traceback object.

    Returns:
        None
    """
    logging.error(''.join(traceback.format_tb(tb)))
    logging.error('{0}: {1}'.format(ex_cls, ex))


sys.excepthook = log_uncaught_exceptions

opendevin_logger = logging.getLogger('opendevin')
opendevin_logger.setLevel(logging.INFO)
LOG_DIR = os.path.join(
    # parent dir of opendevin/core (i.e., root of the repo)
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'logs',
)
if DEBUG:
    opendevin_logger.setLevel(logging.DEBUG)
    # default log to project root
    opendevin_logger.info('DEBUG logging is enabled. Logging to %s', LOG_DIR)
opendevin_logger.addHandler(get_file_handler(LOG_DIR))
opendevin_logger.addHandler(get_console_handler())
opendevin_logger.addFilter(SensitiveDataFilter(opendevin_logger.name))
opendevin_logger.propagate = False
opendevin_logger.debug('Logging initialized')


# Exclude LiteLLM from logging output
logging.getLogger('LiteLLM').disabled = True
logging.getLogger('LiteLLM Router').disabled = True
logging.getLogger('LiteLLM Proxy').disabled = True


class LlmFileHandler(logging.FileHandler):
    """# LLM prompt and response logging"""

    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        """Initializes an instance of LlmFileHandler.

        Args:
            filename (str): The name of the log file.
            mode (str, optional): The file mode. Defaults to 'a'.
            encoding (str, optional): The file encoding. Defaults to None.
            delay (bool, optional): Whether to delay file opening. Defaults to False.
        """
        self.filename = filename
        self.message_counter = 1
        if DEBUG:
            self.session = datetime.now().strftime('%y-%m-%d_%H-%M')
        else:
            self.session = 'default'
        self.log_directory = os.path.join(LOG_DIR, 'llm', self.session)
        os.makedirs(self.log_directory, exist_ok=True)
        if not DEBUG:
            # Clear the log directory if not in debug mode
            for file in os.listdir(self.log_directory):
                file_path = os.path.join(self.log_directory, file)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    opendevin_logger.error(
                        'Failed to delete %s. Reason: %s', file_path, e
                    )
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        super().__init__(self.baseFilename, mode, encoding, delay)

    def emit(self, record):
        """Emits a log record.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        filename = f'{self.filename}_{self.message_counter:03}.log'
        self.baseFilename = os.path.join(self.log_directory, filename)
        self.stream = self._open()
        super().emit(record)
        self.stream.close()
        opendevin_logger.debug('Logging to %s', self.baseFilename)
        self.message_counter += 1


def _get_llm_file_handler(name, debug_level=logging.DEBUG):
    # The 'delay' parameter, when set to True, postpones the opening of the log file
    # until the first log message is emitted.
    llm_file_handler = LlmFileHandler(name, delay=True)
    llm_file_handler.setFormatter(llm_formatter)
    llm_file_handler.setLevel(debug_level)
    return llm_file_handler


def _setup_llm_logger(name, debug_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(debug_level)
    logger.addHandler(_get_llm_file_handler(name, debug_level))
    return logger


llm_prompt_logger = _setup_llm_logger('prompt', logging.DEBUG)
llm_response_logger = _setup_llm_logger('response', logging.DEBUG)
