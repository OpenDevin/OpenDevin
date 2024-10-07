import argparse
import os
import pathlib
import platform
from dataclasses import is_dataclass
from types import UnionType
from typing import Any, MutableMapping, get_args, get_origin

import toml
from dotenv import load_dotenv

from openhands.core import logger
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.app_config import AppConfig
from openhands.core.config.config_utils import (
    OH_DEFAULT_AGENT,
    OH_MAX_ITERATIONS,
    UndefinedString,
)
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.sandbox_config import SandboxConfig

load_dotenv()


def load_from_env(cfg: AppConfig, env_or_toml_dict: dict | MutableMapping[str, str]):
    """Reads the env-style vars and sets config attributes based on env vars or a config.toml dict.
    Compatibility with vars like LLM_BASE_URL, AGENT_MEMORY_ENABLED, SANDBOX_TIMEOUT and others.

    Args:
        cfg: The AppConfig object to set attributes on.
        env_or_toml_dict: The environment variables or a config.toml dict.
    """

    def get_optional_type(union_type: UnionType) -> Any:
        """Returns the non-None type from a Union."""
        types = get_args(union_type)
        return next((t for t in types if t is not type(None)), None)

    # helper function to set attributes based on env vars
    def set_attr_from_env(sub_config: Any, prefix=''):
        """Set attributes of a config dataclass based on environment variables."""
        for field_name, field_type in sub_config.__annotations__.items():
            # compute the expected env var name from the prefix and field name
            # e.g. LLM_BASE_URL
            env_var_name = (prefix + field_name).upper()

            if is_dataclass(field_type):
                # nested dataclass
                nested_sub_config = getattr(sub_config, field_name)
                set_attr_from_env(nested_sub_config, prefix=field_name + '_')
            elif env_var_name in env_or_toml_dict:
                # convert the env var to the correct type and set it
                value = env_or_toml_dict[env_var_name]

                # skip empty config values (fall back to default)
                if not value:
                    continue

                try:
                    # if it's an optional type, get the non-None type
                    if get_origin(field_type) is UnionType:
                        field_type = get_optional_type(field_type)

                    # Attempt to cast the env var to type hinted in the dataclass
                    if field_type is bool:
                        cast_value = str(value).lower() in ['true', '1']
                    else:
                        cast_value = field_type(value)
                    setattr(sub_config, field_name, cast_value)
                except (ValueError, TypeError):
                    logger.openhands_logger.error(
                        f'Error setting env var {env_var_name}={value}: check that the value is of the right type'
                    )

    # Start processing from the root of the config object
    set_attr_from_env(cfg)

    # load default LLM config from env
    default_llm_config = cfg.get_llm_config()
    set_attr_from_env(default_llm_config, 'LLM_')
    # load default agent config from env
    default_agent_config = cfg.get_agent_config()
    set_attr_from_env(default_agent_config, 'AGENT_')


def load_from_toml(cfg: AppConfig, toml_file: str = 'config.toml'):
    """Load the config from the toml file. Supports both styles of config vars.

    Args:
        cfg: The AppConfig object to update attributes of.
        toml_file: The path to the toml file. Defaults to 'config.toml'.
    """
    # try to read the config.toml file into the config object
    try:
        with open(toml_file, 'r', encoding='utf-8') as toml_contents:
            toml_config = toml.load(toml_contents)
    except FileNotFoundError:
        return
    except toml.TomlDecodeError as e:
        logger.openhands_logger.warning(
            f'Cannot parse config from toml, toml values have not been applied.\nError: {e}',
            exc_info=False,
        )
        return

    # if there was an exception or core is not in the toml, try to use the old-style toml
    if 'core' not in toml_config:
        # re-use the env loader to set the config from env-style vars
        load_from_env(cfg, toml_config)
        return

    core_config = toml_config['core']

    # load llm configs and agent configs
    for key, value in toml_config.items():
        if isinstance(value, dict):
            try:
                if key is not None and key.lower() == 'agent':
                    logger.openhands_logger.debug(
                        'Attempt to load default agent config from config toml'
                    )
                    non_dict_fields = {
                        k: v for k, v in value.items() if not isinstance(v, dict)
                    }
                    agent_config = AgentConfig(**non_dict_fields)
                    cfg.set_agent_config(agent_config, 'agent')
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, dict):
                            logger.openhands_logger.debug(
                                f'Attempt to load group {nested_key} from config toml as agent config'
                            )
                            agent_config = AgentConfig(**nested_value)
                            cfg.set_agent_config(agent_config, nested_key)
                elif key is not None and key.lower() == 'llm':
                    logger.openhands_logger.debug(
                        'Attempt to load default LLM config from config toml'
                    )
                    non_dict_fields = {
                        k: v for k, v in value.items() if not isinstance(v, dict)
                    }
                    llm_config = LLMConfig(**non_dict_fields)
                    cfg.set_llm_config(llm_config, 'llm')
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, dict):
                            logger.openhands_logger.debug(
                                f'Attempt to load group {nested_key} from config toml as llm config'
                            )
                            llm_config = LLMConfig(**nested_value)
                            cfg.set_llm_config(llm_config, nested_key)
                elif not key.startswith('sandbox') and key.lower() != 'core':
                    logger.openhands_logger.warning(
                        f'Unknown key in {toml_file}: "{key}"'
                    )
            except (TypeError, KeyError) as e:
                logger.openhands_logger.warning(
                    f'Cannot parse config from toml, toml values have not been applied.\n Error: {e}',
                    exc_info=False,
                )
        else:
            logger.openhands_logger.warning(f'Unknown key in {toml_file}: "{key}')

    try:
        # set sandbox config from the toml file
        sandbox_config = cfg.sandbox

        # migrate old sandbox configs from [core] section to sandbox config
        keys_to_migrate = [key for key in core_config if key.startswith('sandbox_')]
        for key in keys_to_migrate:
            new_key = key.replace('sandbox_', '')
            if new_key in sandbox_config.__annotations__:
                # read the key in sandbox and remove it from core
                setattr(sandbox_config, new_key, core_config.pop(key))
            else:
                logger.openhands_logger.warning(f'Unknown sandbox config: {key}')

        # the new style values override the old style values
        if 'sandbox' in toml_config:
            sandbox_config = SandboxConfig(**toml_config['sandbox'])

        # update the config object with the new values
        cfg.sandbox = sandbox_config
        for key, value in core_config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.openhands_logger.warning(f'Unknown core config key: {key}')
    except (TypeError, KeyError) as e:
        logger.openhands_logger.warning(
            f'Cannot parse config from toml, toml values have not been applied.\nError: {e}',
            exc_info=False,
        )


def finalize_config(cfg: AppConfig):
    """More tweaks to the config after it's been loaded."""
    cfg.workspace_base = os.path.abspath(cfg.workspace_base)
    # Set workspace_mount_path if not set by the user
    if cfg.workspace_mount_path is UndefinedString.UNDEFINED:
        cfg.workspace_mount_path = cfg.workspace_base

    if cfg.workspace_mount_rewrite:  # and not config.workspace_mount_path:
        # TODO why do we need to check if workspace_mount_path is None?
        base = cfg.workspace_base or os.getcwd()
        parts = cfg.workspace_mount_rewrite.split(':')
        cfg.workspace_mount_path = base.replace(parts[0], parts[1])

    for llm in cfg.llms.values():
        if llm.embedding_base_url is None:
            llm.embedding_base_url = llm.base_url

    if cfg.sandbox.use_host_network and platform.system() == 'Darwin':
        logger.openhands_logger.warning(
            'Please upgrade to Docker Desktop 4.29.0 or later to use host network mode on macOS. '
            'See https://github.com/docker/roadmap/issues/238#issuecomment-2044688144 for more information.'
        )

    # make sure cache dir exists
    if cfg.cache_dir:
        pathlib.Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)


# Utility function for command line --group argument
def get_llm_config_arg(
    llm_config_arg: str, toml_file: str = 'config.toml'
) -> LLMConfig | None:
    """Get a group of llm settings from the config file.

    A group in config.toml can look like this:

    ```
    [llm.gpt-3.5-for-eval]
    model = 'gpt-3.5-turbo'
    api_key = '...'
    temperature = 0.5
    num_retries = 8
    ...
    ```

    The user-defined group name, like "gpt-3.5-for-eval", is the argument to this function. The function will load the LLMConfig object
    with the settings of this group, from the config file, and set it as the LLMConfig object for the app.

    Note that the group must be under "llm" group, or in other words, the group name must start with "llm.".

    Args:
        llm_config_arg: The group of llm settings to get from the config.toml file.

    Returns:
        LLMConfig: The LLMConfig object with the settings from the config file.
    """
    # keep only the name, just in case
    llm_config_arg = llm_config_arg.strip('[]')

    # truncate the prefix, just in case
    if llm_config_arg.startswith('llm.'):
        llm_config_arg = llm_config_arg[4:]

    logger.openhands_logger.info(f'Loading llm config from {llm_config_arg}')

    # load the toml file
    try:
        with open(toml_file, 'r', encoding='utf-8') as toml_contents:
            toml_config = toml.load(toml_contents)
    except FileNotFoundError as e:
        logger.openhands_logger.error(f'Config file not found: {e}')
        return None
    except toml.TomlDecodeError as e:
        logger.openhands_logger.error(
            f'Cannot parse llm group from {llm_config_arg}. Exception: {e}'
        )
        return None

    # update the llm config with the specified section
    if 'llm' in toml_config and llm_config_arg in toml_config['llm']:
        return LLMConfig(**toml_config['llm'][llm_config_arg])
    logger.openhands_logger.debug(f'Loading from toml failed for {llm_config_arg}')
    return None


def add_eval_parser(group: argparse._ArgumentGroup):
    group.add_argument(
        '--eval-output-dir',
        default='evaluation/evaluation_outputs/outputs',
        type=str,
        help='The directory to save evaluation output',
    )
    group.add_argument(
        '--eval-n-limit',
        default=None,
        type=int,
        help='The number of instances to evaluate',
    )
    group.add_argument(
        '--eval-num-workers',
        default=4,
        type=int,
        help='The number of workers to use for evaluation',
    )
    group.add_argument(
        '--eval-note',
        default=None,
        type=str,
        help='The note to add to the evaluation directory',
    )
    group.add_argument(
        '--eval-ids',
        default=None,
        type=str,
        help='The comma-separated list (in quotes) of IDs of the instances to evaluate',
    )


def get_parser(add_eval: bool = True) -> argparse.ArgumentParser:
    """Get the parser for the command line arguments."""
    parser = argparse.ArgumentParser(description='Run an agent with a specific task')

    parser.add_argument(
        '--config-file',
        type=str,
        default='config.toml',
        help='Path to the config file (default: config.toml in the current directory)',
    )
    parser.add_argument(
        '-d',
        '--directory',
        type=str,
        help='The working directory for the agent',
    )
    parser.add_argument(
        '-t',
        '--task',
        type=str,
        default='',
        help='The task for the agent to perform',
    )
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help='Path to a file containing the task. Overrides -t if both are provided.',
    )
    parser.add_argument(
        '-c',
        '--agent-cls',
        default=OH_DEFAULT_AGENT,
        type=str,
        help='Name of the default agent to use',
    )
    parser.add_argument(
        '-i',
        '--max-iterations',
        default=OH_MAX_ITERATIONS,
        type=int,
        help='The maximum number of iterations to run the agent',
    )
    parser.add_argument(
        '-b',
        '--max-budget-per-task',
        type=float,
        help='The maximum budget allowed per task, beyond which the agent will stop.',
    )
    parser.add_argument(
        '-l',
        '--llm-config',
        default=None,
        type=str,
        help='Replace default LLM ([llm] section in config.toml) config with the specified LLM config, e.g. "llama3" for [llm.llama3] section in config.toml',
    )
    parser.add_argument(
        '-n',
        '--name',
        default='default',
        type=str,
        help='Name for the session',
    )

    if add_eval:
        eval_group = parser.add_argument_group('Evaluation options')
        add_eval_parser(eval_group)
    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = get_parser()
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


def load_app_config(
    set_logging_levels: bool = True, config_file: str = 'config.toml'
) -> AppConfig:
    """Load the configuration from the specified config file and environment variables.

    Args:
        set_logger_levels: Whether to set the global variables for logging levels.
        config_file: Path to the config file. Defaults to 'config.toml' in the current directory.
    """
    config = AppConfig()
    load_from_toml(config, config_file)
    load_from_env(config, os.environ)
    finalize_config(config)
    if set_logging_levels:
        logger.DEBUG = config.debug
        logger.DISABLE_COLOR_PRINTING = config.disable_color
    return config
