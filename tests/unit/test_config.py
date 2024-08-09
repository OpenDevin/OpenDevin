import os

import pytest

from opendevin.core.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    UndefinedString,
    finalize_config,
    get_llm_config_arg,
    load_from_env,
    load_from_toml,
)


@pytest.fixture
def setup_env():
    # Create old-style and new-style TOML files
    with open('old_style_config.toml', 'w') as f:
        f.write('[default]\nLLM_MODEL="GPT-4"\n')

    with open('new_style_config.toml', 'w') as f:
        f.write('[app]\nLLM_MODEL="GPT-3"\n')

    yield

    # Cleanup TOML files after the test
    os.remove('old_style_config.toml')
    os.remove('new_style_config.toml')


@pytest.fixture
def temp_toml_file(tmp_path):
    # Fixture to create a temporary directory and TOML file for testing
    tmp_toml_file = os.path.join(tmp_path, 'config.toml')
    yield tmp_toml_file


@pytest.fixture
def default_config(monkeypatch):
    # Fixture to provide a default AppConfig instance
    AppConfig.reset()
    yield AppConfig()


def test_compat_env_to_config(monkeypatch, setup_env):
    # Use `monkeypatch` to set environment variables for this specific test
    monkeypatch.setenv('WORKSPACE_BASE', '/repos/opendevin/workspace')
    monkeypatch.setenv('LLM_API_KEY', 'sk-proj-rgMV0...')
    monkeypatch.setenv('LLM_MODEL', 'gpt-4o')
    monkeypatch.setenv('AGENT_MEMORY_MAX_THREADS', '4')
    monkeypatch.setenv('AGENT_MEMORY_ENABLED', 'True')
    monkeypatch.setenv('DEFAULT_AGENT', 'CodeActAgent')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '10')

    config = AppConfig()
    load_from_env(config, os.environ)

    assert config.workspace_base == '/repos/opendevin/workspace'
    assert isinstance(config.get_llm_config(), LLMConfig)
    assert config.get_llm_config().api_key == 'sk-proj-rgMV0...'
    assert config.get_llm_config().model == 'gpt-4o'
    assert isinstance(config.get_agent_config(), AgentConfig)
    assert isinstance(config.get_agent_config().memory_max_threads, int)
    assert config.get_agent_config().memory_max_threads == 4
    assert config.get_agent_config().memory_enabled is True
    assert config.default_agent == 'CodeActAgent'
    assert config.sandbox.timeout == 10


def test_load_from_old_style_env(monkeypatch, default_config):
    # Test loading configuration from old-style environment variables using monkeypatch
    monkeypatch.setenv('LLM_API_KEY', 'test-api-key')
    monkeypatch.setenv('AGENT_MEMORY_ENABLED', 'True')
    monkeypatch.setenv('DEFAULT_AGENT', 'PlannerAgent')
    monkeypatch.setenv('WORKSPACE_BASE', '/opt/files/workspace')
    monkeypatch.setenv('SANDBOX_CONTAINER_IMAGE', 'custom_image')

    load_from_env(default_config, os.environ)

    assert default_config.get_llm_config().api_key == 'test-api-key'
    assert default_config.get_agent_config().memory_enabled is True
    assert default_config.default_agent == 'PlannerAgent'
    assert default_config.workspace_base == '/opt/files/workspace'
    assert (
        default_config.workspace_mount_path is UndefinedString.UNDEFINED
    )  # before finalize_config
    assert (
        default_config.workspace_mount_path_in_sandbox is not UndefinedString.UNDEFINED
    )
    assert default_config.sandbox.container_image == 'custom_image'


def test_load_from_new_style_toml(default_config, temp_toml_file):
    # Test loading configuration from a new-style TOML file
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[llm]
model = "test-model"
api_key = "toml-api-key"

[llm.cheap]
model = "some-cheap-model"
api_key = "cheap-model-api-key"

[agent]
memory_enabled = true

[agent.BrowsingAgent]
llm_config = "cheap"
memory_enabled = false

[sandbox]
timeout = 1

[core]
workspace_base = "/opt/files2/workspace"
default_agent = "TestAgent"
"""
        )

    load_from_toml(default_config, temp_toml_file)

    # default llm & agent configs
    assert default_config.default_agent == 'TestAgent'
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config().api_key == 'toml-api-key'
    assert default_config.get_agent_config().memory_enabled is True

    # undefined agent config inherits default ones
    assert (
        default_config.get_llm_config_from_agent('CodeActAgent')
        == default_config.get_llm_config()
    )
    assert default_config.get_agent_config('CodeActAgent').memory_enabled is True

    # defined agent config overrides default ones
    assert default_config.get_llm_config_from_agent(
        'BrowsingAgent'
    ) == default_config.get_llm_config('cheap')
    assert (
        default_config.get_llm_config_from_agent('BrowsingAgent').model
        == 'some-cheap-model'
    )
    assert default_config.get_agent_config('BrowsingAgent').memory_enabled is False

    assert default_config.workspace_base == '/opt/files2/workspace'
    assert default_config.sandbox.timeout == 1

    # before finalize_config, workspace_mount_path is UndefinedString.UNDEFINED if it was not set
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED
    assert (
        default_config.workspace_mount_path_in_sandbox is not UndefinedString.UNDEFINED
    )
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'

    finalize_config(default_config)

    # after finalize_config, workspace_mount_path is set to the absolute path of workspace_base
    # if it was undefined
    assert default_config.workspace_mount_path == '/opt/files2/workspace'


def test_compat_load_sandbox_from_toml(default_config: AppConfig, temp_toml_file: str):
    # test loading configuration from a new-style TOML file
    # uses a toml file with sandbox_vars instead of a sandbox section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[llm]
model = "test-model"

[agent]
memory_enabled = true

[core]
workspace_base = "/opt/files2/workspace"
sandbox_timeout = 500
sandbox_container_image = "node:14"
sandbox_user_id = 1001
default_agent = "TestAgent"
"""
        )

    load_from_toml(default_config, temp_toml_file)

    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config_from_agent().model == 'test-model'
    assert default_config.default_agent == 'TestAgent'
    assert default_config.get_agent_config().memory_enabled is True
    assert default_config.workspace_base == '/opt/files2/workspace'
    assert default_config.sandbox.timeout == 500
    assert default_config.sandbox.container_image == 'node:14'
    assert default_config.sandbox.user_id == 1001
    assert default_config.workspace_mount_path_in_sandbox == '/workspace'

    finalize_config(default_config)

    # app config doesn't have fields sandbox_*
    assert not hasattr(default_config, 'sandbox_timeout')
    assert not hasattr(default_config, 'sandbox_container_image')
    assert not hasattr(default_config, 'sandbox_user_id')

    # after finalize_config, workspace_mount_path is set to the absolute path of workspace_base
    # if it was undefined
    assert default_config.workspace_mount_path == '/opt/files2/workspace'


def test_env_overrides_compat_toml(monkeypatch, default_config, temp_toml_file):
    # test that environment variables override TOML values using monkeypatch
    # uses a toml file with sandbox_vars instead of a sandbox section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
api_key = "toml-api-key"

[core]
workspace_base = "/opt/files3/workspace"
disable_color = true
sandbox_timeout = 500
sandbox_user_id = 1001
""")

    monkeypatch.setenv('LLM_API_KEY', 'env-api-key')
    monkeypatch.setenv('WORKSPACE_BASE', 'UNDEFINED')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '1000')
    monkeypatch.setenv('SANDBOX_USER_ID', '1002')

    load_from_toml(default_config, temp_toml_file)

    # before finalize_config, workspace_mount_path is UndefinedString.UNDEFINED if it was not set
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED

    load_from_env(default_config, os.environ)

    assert os.environ.get('LLM_MODEL') is None
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config('llm').model == 'test-model'
    assert default_config.get_llm_config_from_agent().model == 'test-model'
    assert default_config.get_llm_config().api_key == 'env-api-key'

    # after we set workspace_base to 'UNDEFINED' in the environment,
    # workspace_base should be set to that
    # workspace_mount path is still UndefinedString.UNDEFINED
    assert default_config.workspace_base is not UndefinedString.UNDEFINED
    assert default_config.workspace_base == 'UNDEFINED'
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED
    assert default_config.workspace_mount_path == 'UNDEFINED'

    assert default_config.disable_color is True
    assert default_config.sandbox.timeout == 1000
    assert default_config.sandbox.user_id == 1002

    finalize_config(default_config)
    # after finalize_config, workspace_mount_path is set to absolute path of workspace_base if it was undefined
    assert default_config.workspace_mount_path == os.getcwd() + '/UNDEFINED'


def test_env_overrides_sandbox_toml(monkeypatch, default_config, temp_toml_file):
    # test that environment variables override TOML values using monkeypatch
    # uses a toml file with a sandbox section
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write("""
[llm]
model = "test-model"
api_key = "toml-api-key"

[core]
workspace_base = "/opt/files3/workspace"

[sandbox]
timeout = 500
user_id = 1001
""")

    monkeypatch.setenv('LLM_API_KEY', 'env-api-key')
    monkeypatch.setenv('WORKSPACE_BASE', 'UNDEFINED')
    monkeypatch.setenv('SANDBOX_TIMEOUT', '1000')
    monkeypatch.setenv('SANDBOX_USER_ID', '1002')

    load_from_toml(default_config, temp_toml_file)

    # before finalize_config, workspace_mount_path is UndefinedString.UNDEFINED if it was not set
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED

    # before load_from_env, values are set to the values from the toml file
    assert default_config.get_llm_config().api_key == 'toml-api-key'
    assert default_config.sandbox.timeout == 500
    assert default_config.sandbox.user_id == 1001

    load_from_env(default_config, os.environ)

    # values from env override values from toml
    assert os.environ.get('LLM_MODEL') is None
    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.get_llm_config().api_key == 'env-api-key'

    assert default_config.sandbox.timeout == 1000
    assert default_config.sandbox.user_id == 1002

    finalize_config(default_config)
    # after finalize_config, workspace_mount_path is set to absolute path of workspace_base if it was undefined
    assert default_config.workspace_mount_path == os.getcwd() + '/UNDEFINED'


def test_sandbox_config_from_toml(default_config, temp_toml_file):
    # Test loading configuration from a new-style TOML file
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write(
            """
[core]
workspace_base = "/opt/files/workspace"

[llm]
model = "test-model"

[sandbox]
timeout = 1
container_image = "custom_image"
user_id = 1001
"""
        )

    load_from_toml(default_config, temp_toml_file)
    load_from_env(default_config, os.environ)
    finalize_config(default_config)

    assert default_config.get_llm_config().model == 'test-model'
    assert default_config.sandbox.timeout == 1
    assert default_config.sandbox.container_image == 'custom_image'
    assert default_config.sandbox.user_id == 1001


def test_defaults_dict_after_updates(default_config):
    # Test that `defaults_dict` retains initial values after updates.
    initial_defaults = default_config.defaults_dict
    assert (
        initial_defaults['workspace_mount_path']['default'] is UndefinedString.UNDEFINED
    )
    assert initial_defaults['default_agent']['default'] == 'CodeActAgent'

    updated_config = AppConfig()
    updated_config.get_llm_config().api_key = 'updated-api-key'
    updated_config.get_llm_config('llm').api_key = 'updated-api-key'
    updated_config.get_llm_config_from_agent('agent').api_key = 'updated-api-key'
    updated_config.get_llm_config_from_agent('PlannerAgent').api_key = 'updated-api-key'
    updated_config.default_agent = 'PlannerAgent'

    defaults_after_updates = updated_config.defaults_dict
    assert defaults_after_updates['default_agent']['default'] == 'CodeActAgent'
    assert (
        defaults_after_updates['workspace_mount_path']['default']
        is UndefinedString.UNDEFINED
    )
    assert defaults_after_updates['sandbox']['timeout']['default'] == 120
    assert (
        defaults_after_updates['sandbox']['container_image']['default']
        == 'nikolaik/python-nodejs:python3.11-nodejs22'
    )
    assert defaults_after_updates == initial_defaults


def test_invalid_toml_format(monkeypatch, temp_toml_file, default_config):
    # Invalid TOML format doesn't break the configuration
    monkeypatch.setenv('LLM_MODEL', 'gpt-5-turbo-1106')
    monkeypatch.setenv('WORKSPACE_MOUNT_PATH', '/home/user/project')
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    with open(temp_toml_file, 'w', encoding='utf-8') as toml_file:
        toml_file.write('INVALID TOML CONTENT')

    load_from_toml(default_config)
    load_from_env(default_config, os.environ)
    default_config.jwt_secret = None  # prevent leak
    for llm in default_config.llms.values():
        llm.api_key = None  # prevent leak
    assert default_config.get_llm_config().model == 'gpt-5-turbo-1106'
    assert default_config.get_llm_config().custom_llm_provider is None
    assert default_config.workspace_mount_path == '/home/user/project'


def test_finalize_config(default_config):
    # Test finalize config
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED
    finalize_config(default_config)

    assert default_config.workspace_mount_path == os.path.abspath(
        default_config.workspace_base
    )


# tests for workspace, mount path, path in sandbox, cache dir
def test_workspace_mount_path_default(default_config):
    assert default_config.workspace_mount_path is UndefinedString.UNDEFINED
    finalize_config(default_config)
    assert default_config.workspace_mount_path == os.path.abspath(
        default_config.workspace_base
    )


def test_workspace_mount_rewrite(default_config, monkeypatch):
    default_config.workspace_base = '/home/user/project'
    default_config.workspace_mount_rewrite = '/home/user:/sandbox'
    monkeypatch.setattr('os.getcwd', lambda: '/current/working/directory')
    finalize_config(default_config)
    assert default_config.workspace_mount_path == '/sandbox/project'


def test_embedding_base_url_default(default_config):
    default_config.get_llm_config().base_url = 'https://api.exampleapi.com'
    finalize_config(default_config)
    assert (
        default_config.get_llm_config().embedding_base_url
        == 'https://api.exampleapi.com'
    )


def test_cache_dir_creation(default_config, tmpdir):
    default_config.cache_dir = str(tmpdir.join('test_cache'))
    finalize_config(default_config)
    assert os.path.exists(default_config.cache_dir)


def test_api_keys_repr_str():
    # Test LLMConfig
    llm_config = LLMConfig(
        api_key='my_api_key',
        aws_access_key_id='my_access_key',
        aws_secret_access_key='my_secret_key',
    )
    assert "api_key='******'" in repr(llm_config)
    assert "aws_access_key_id='******'" in repr(llm_config)
    assert "aws_secret_access_key='******'" in repr(llm_config)
    assert "api_key='******'" in str(llm_config)
    assert "aws_access_key_id='******'" in str(llm_config)
    assert "aws_secret_access_key='******'" in str(llm_config)

    # Check that no other attrs in LLMConfig have 'key' or 'token' in their name
    # This will fail when new attrs are added, and attract attention
    known_key_token_attrs_llm = [
        'api_key',
        'aws_access_key_id',
        'aws_secret_access_key',
        'input_cost_per_token',
        'output_cost_per_token',
    ]
    for attr_name in dir(LLMConfig):
        if (
            not attr_name.startswith('__')
            and attr_name not in known_key_token_attrs_llm
        ):
            assert (
                'key' not in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'key' in LLMConfig"
            assert (
                'token' not in attr_name.lower() or 'tokens' in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'token' in LLMConfig"

    # Test AgentConfig
    # No attrs in AgentConfig have 'key' or 'token' in their name
    agent_config = AgentConfig(memory_enabled=True, memory_max_threads=4)
    for attr_name in dir(AgentConfig):
        if not attr_name.startswith('__'):
            assert (
                'key' not in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'key' in AgentConfig"
            assert (
                'token' not in attr_name.lower() or 'tokens' in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'token' in AgentConfig"

    # Test AppConfig
    app_config = AppConfig(
        llms={'llm': llm_config},
        agents={'agent': agent_config},
        e2b_api_key='my_e2b_api_key',
        jwt_secret='my_jwt_secret',
    )
    assert "e2b_api_key='******'" in repr(app_config)
    assert "e2b_api_key='******'" in str(app_config)
    assert "jwt_secret='******'" in repr(app_config)
    assert "jwt_secret='******'" in str(app_config)

    # Check that no other attrs in AppConfig have 'key' or 'token' in their name
    # This will fail when new attrs are added, and attract attention
    known_key_token_attrs_app = ['e2b_api_key']
    for attr_name in dir(AppConfig):
        if (
            not attr_name.startswith('__')
            and attr_name not in known_key_token_attrs_app
        ):
            assert (
                'key' not in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'key' in AppConfig"
            assert (
                'token' not in attr_name.lower() or 'tokens' in attr_name.lower()
            ), f"Unexpected attribute '{attr_name}' contains 'token' in AppConfig"


def test_max_iterations_and_max_budget_per_task_from_toml(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 100
max_budget_per_task = 4.0
"""

    config = AppConfig()
    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    load_from_toml(config, temp_toml_file)

    assert config.max_iterations == 100
    assert config.max_budget_per_task == 4.0


def test_get_llm_config_arg(temp_toml_file):
    temp_toml = """
[core]
max_iterations = 100
max_budget_per_task = 4.0

[llm.gpt3]
model="gpt-3.5-turbo"
api_key="redacted"
embedding_model="openai"

[llm.gpt4o]
model="gpt-4o"
api_key="redacted"
embedding_model="openai"
"""

    with open(temp_toml_file, 'w') as f:
        f.write(temp_toml)

    llm_config = get_llm_config_arg('gpt3', temp_toml_file)
    assert llm_config.model == 'gpt-3.5-turbo'
    assert llm_config.embedding_model == 'openai'
