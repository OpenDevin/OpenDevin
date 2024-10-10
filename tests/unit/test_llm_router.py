import tempfile
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

from openhands.core.config import (
    AppConfig,
    LLMConfig,
    ModelConfig,
    RouterConfig,
    load_from_toml,
)
from openhands.llm.llm import LLM


@pytest.fixture(scope='module')
def llm_router_config():
    yield get_llm_router_config()


def get_llm_router_config():
    model_list = [
        ModelConfig(
            model_name='gpt-4o',
            litellm_params={
                'model': 'gpt-4o',
                'api_key': 'OPENAI_API_KEY',
                'max_retries': 3,
            },
        ),
        ModelConfig(
            model_name='claude-3-5-sonnet-20240620',
            litellm_params={
                'model': 'azure/claude-3-5-sonnet-20240620',
                'api_key': 'AZURE_API_KEY',
                'api_base': 'https://MODEL-URL.openai.azure.com//openai/',
                'api_version': '2023-05-15',
                'max_retries': 7,
                'timeout': 0.01,
                'stream_timeout': 0.000_001,
            },
        ),
    ]
    config = LLMConfig(
        model='gpt-4o',
        api_key='dummy_default_key',
        router_config=RouterConfig(
            model_list=model_list,
            routing_strategy='simple-shuffle',
            num_retries=3,
            cooldown_time=1.0,
            allowed_fails=5,
            cache_responses=False,
        ),
    )

    # Write the config to a file
    # with open('/mnt/d/github/workspace/routerconfig.toml', 'w') as f:
    #     toml.dump(asdict(config), f)

    return config


@pytest.fixture(scope='module')
def router_content_policy_config():
    model_list = [
        ModelConfig(
            model_name='content-policy-model',
            litellm_params={
                'model': 'openai/content-policy-model',
                'api_key': 'content-policy-api-key',
            },
        ),
        ModelConfig(
            model_name='fallback-model',
            litellm_params={
                'model': 'gpt-4o',
                'api_key': 'OPENAI_API_KEY',
            },
        ),
    ]
    return LLMConfig(
        model='content-policy-model',
        api_key='dummy_default_key',
        router_config=RouterConfig(
            model_list=model_list,
            routing_strategy='simple-shuffle',
            num_retries=3,
            cooldown_time=1.0,
            allowed_fails=5,
            cache_responses=False,
        ),
    )


@pytest.fixture(scope='module')
def router_content_policy_config_azure():
    model_list = [
        ModelConfig(
            model_name='gpt-4-vision-enhancements',
            litellm_params={
                'model': 'azure/gpt-4-vision',
                'api_key': 'AZURE_API_KEY',
                'max_tokens': 4096,
                'timeout': 300,
                'base_url': 'https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/',
            },
        )
    ]
    return LLMConfig(
        model='gpt-4-vision-enhanced',
        router_config=RouterConfig(
            model_list=model_list,
            routing_strategy='simple-shuffle',
            cooldown_time=1.0,
            num_retries=8,
            retry_after=15,
            allowed_fails=2,
            cache_responses=False,
        ),
    )


##############################################################################
##############################################################################


@patch('openhands.llm.llm.time.sleep')
@patch('openhands.llm.llm.litellm.Router')
def test_router_completion_rate_limit_wait_time(
    mock_router, mock_sleep, llm_router_config
):
    # Set up the mock router's completion method to simulate rate limiting
    mock_router_instance = mock_router.return_value
    mock_completion = mock_router_instance.completion

    # Simulate rate limiting on the first attempt and a successful completion on the second attempt
    mock_completion.side_effect = [
        RateLimitError(
            message='Rate limit exceeded',
            llm_provider='test_provider',
            model='test_model',
            response=None,
            litellm_debug_info=None,
            max_retries=2,
            num_retries=1,
        ),
        {
            'choices': [{'message': {'content': 'Retry successful'}}]
        },  # Successful response on the second call
    ]
    # Set num_retries to 1 to prevent retry logic from interfering
    llm_router_config.num_retries = 1

    llm = LLM(config=llm_router_config)

    # Expect a RateLimitError to be raised since num_retries is 1
    with pytest.raises(RateLimitError):
        llm.completion(
            messages=[{'role': 'user', 'content': 'Hello!'}],
            stream=False,
        )

    # Assertions
    assert mock_completion.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    'exception_type, extra_args',
    [
        (APIConnectionError, {}),
        (InternalServerError, {}),
        (ServiceUnavailableError, {}),
        (RateLimitError, {'response': None}),  # response is required for RateLimitError
    ],
)
@patch('openhands.llm.llm.litellm.Router')
def test_router_completion_retries(
    mock_router, llm_router_config, exception_type, extra_args
):
    """Since the Router class handles retries internally, when it exhausts all retries,
    it raises an exception. In this test, we're mocking the completion method to raise
    an exception on the first call and return a successful response on the second.
    However, the Router class is designed to retry only when the exception is
    RateLimitError. For other exceptions, it will not retry, unless there is a custom
    retry policy defined in the RouterConfig.
    """
    mock_router_instance = mock_router.return_value
    mock_completion = mock_router_instance.completion

    # Simulate the Router failing after all retries are exhausted
    mock_completion.side_effect = exception_type(
        message='Test error message',
        llm_provider='test_provider',
        model='test_model',
        litellm_debug_info=None,
        max_retries=2,
        num_retries=1,
        **extra_args,
    )

    llm = LLM(config=llm_router_config)

    # Expect the exception to be raised after retries are exhausted
    with pytest.raises(exception_type):
        llm.completion(messages=[{'role': 'user', 'content': 'Hello!'}], stream=False)

    # Assert that the completion function was called once
    assert mock_completion.call_count == 1


@patch('openhands.llm.llm.litellm.Router')
def test_router_completion_with_fallback(mock_router, llm_router_config):
    mock_router_instance = mock_router.return_value

    mock_router_instance.completion.return_value = {
        'choices': [{'message': {'content': 'Fallback response'}}],
        'model': 'fallback-model',
    }

    llm = LLM(llm_router_config)

    response = llm.completion(
        messages=[{'role': 'user', 'content': 'Hello!'}],
        fallbacks=['fallback-model'],
    )

    assert mock_router_instance.completion.call_count == 1

    call_kwargs = mock_router_instance.completion.call_args[1]
    assert call_kwargs['messages'] == [{'role': 'user', 'content': 'Hello!'}]
    assert call_kwargs['fallbacks'] == ['fallback-model']

    assert response['choices'][0]['message']['content'] == 'Fallback response'
    assert response['model'] == 'fallback-model'


# credits go to the litellm project for most of below tests


@patch('openhands.llm.llm.litellm.Router')
def test_llm_init_with_router(mock_router, llm_router_config):
    llm = LLM(llm_router_config)

    mock_router.assert_called_once()
    router_args = mock_router.call_args[1]
    assert len(router_args['model_list']) == len(llm_router_config.router_config.models)
    assert (
        router_args['routing_strategy']
        == llm_router_config.router_config.routing_strategy
    )
    assert router_args['num_retries'] == llm_router_config.router_config.num_retries
    assert router_args['cooldown_time'] == llm_router_config.router_config.cooldown_time
    assert router_args['allowed_fails'] == llm_router_config.router_config.allowed_fails
    assert (
        router_args['cache_responses']
        == llm_router_config.router_config.cache_responses
    )

    assert hasattr(llm, 'router')
    assert isinstance(llm.router, MagicMock)


@patch('openhands.llm.llm.litellm.Router')
def test_llm_completion_with_router(mock_router, llm_router_config):
    mock_router_instance = MagicMock()
    mock_router_instance.completion.return_value = {
        'choices': [{'message': {'content': 'Router response'}}]
    }
    mock_router.return_value = mock_router_instance

    llm = LLM(llm_router_config)
    response = llm.completion(messages=[{'role': 'user', 'content': 'Hello!'}])

    mock_router_instance.completion.assert_called_once_with(
        'gpt-4o', messages=[{'role': 'user', 'content': 'Hello!'}]
    )

    assert response['choices'][0]['message']['content'] == 'Router response'


@patch('openhands.llm.llm.litellm.Router')
def test_router_init_gpt_4_vision_enhancements(
    mock_router, router_content_policy_config_azure
):
    LLM(router_content_policy_config_azure)

    mock_router.assert_called_once()
    router_args = mock_router.call_args[1]
    config = router_content_policy_config_azure.router_config

    assert len(router_args['model_list']) == 1
    model_config = router_args['model_list'][0]

    assert model_config['model_name'] == 'gpt-4-vision-enhancements'

    litellm_params = model_config['litellm_params']
    assert litellm_params['model'] == 'azure/gpt-4-vision'
    assert litellm_params['api_key'] == 'AZURE_API_KEY'
    assert (
        litellm_params['base_url']
        == 'https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/'
    )
    assert litellm_params.get('api_version') is None
    assert litellm_params.get('custom_llm_provider') is None
    assert litellm_params.get('max_tokens') == 4096
    assert litellm_params['timeout'] == 300

    original_data_sources = config.models[0].litellm_params.get('dataSources')
    if original_data_sources:
        assert 'dataSources' in litellm_params
        assert len(litellm_params['dataSources']) == 1
        data_source = litellm_params['dataSources'][0]
        assert data_source['type'] == 'AzureComputerVision'
        assert data_source['parameters']['endpoint'] == 'AZURE_VISION_ENHANCE_ENDPOINT'
        assert data_source['parameters']['key'] == 'AZURE_VISION_ENHANCE_KEY'
    else:
        assert 'dataSources' not in litellm_params


@patch('openhands.llm.llm.litellm.Router')
def test_llm_init_with_toml_config(mock_router):
    with tempfile.NamedTemporaryFile(
        delete=False, mode='w', suffix='.toml'
    ) as temp_toml_file:
        temp_toml_file.write(TOML_EXAMPLE_TEXT)
        temp_toml_file_path = temp_toml_file.name

    app_config = AppConfig()
    load_from_toml(app_config, toml_file=temp_toml_file_path)

    router_config = app_config.llms['llm'].router_config
    llm = LLM(LLMConfig(model=router_config.default_model, router_config=router_config))

    mock_router.assert_called_once()

    router_args = mock_router.call_args[1]
    assert len(router_args['model_list']) == 5

    models = router_args['model_list']

    # GPT-4 Vision Enhancements
    assert models[0]['model_name'] == 'gpt-4-vision-enhancements'
    assert models[0]['litellm_params']['model'] == 'azure/gpt-4-vision'
    assert models[0]['litellm_params']['api_key'] == 'AZURE_API_KEY'
    assert (
        models[0]['litellm_params']['base_url']
        == 'https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/'
    )
    assert (
        models[0]['litellm_params']['dataSources'][0]['type'] == 'AzureComputerVision'
    )
    assert (
        models[0]['litellm_params']['dataSources'][0]['parameters']['endpoint']
        == 'AZURE_VISION_ENHANCE_ENDPOINT'
    )
    assert (
        models[0]['litellm_params']['dataSources'][0]['parameters']['key']
        == 'AZURE_VISION_ENHANCE_KEY'
    )

    # OpenRouter Claude 3.5 Sonnet
    assert models[1]['model_name'] == 'or-sonnet-3.5'
    assert (
        models[1]['litellm_params']['model'] == 'openrouter/anthropic/claude-3.5-sonnet'
    )
    assert models[1]['litellm_params']['api_key'] == 'sk-or-v1-xxx'
    assert models[1]['litellm_params']['max_tokens'] == 100
    assert models[1]['litellm_params']['tpm'] == 100000
    assert models[1]['litellm_params']['rpm'] == 10000
    assert models[1]['litellm_params']['timeout'] == 30

    # Anthropic Claude 3.5 Sonnet
    assert models[2]['model_name'] == 'ant-sonnet-3.5'
    assert (
        models[2]['litellm_params']['model'] == 'anthropic/claude-3-5-sonnet-20240620'
    )
    assert models[2]['litellm_params']['api_key'] == 'sk-ant-xxx'
    assert models[2]['litellm_params']['max_tokens'] == 100
    assert models[2]['litellm_params']['tpm'] == 100000
    assert models[2]['litellm_params']['rpm'] == 10000
    assert models[2]['litellm_params']['timeout'] == 30

    # OpenRouter GPT-4
    assert models[3]['model_name'] == 'gpt-4o'
    assert models[3]['litellm_params']['model'] == 'openrouter/openai/chatgpt-4o-latest'
    assert models[3]['litellm_params']['api_key'] == 'sk-or-v1-xxx'
    assert models[3]['litellm_params']['max_tokens'] == 100
    assert models[3]['litellm_params']['tpm'] == 100000
    assert models[3]['litellm_params']['rpm'] == 10000
    assert models[3]['litellm_params']['timeout'] == 30

    # OpenRouter Llama 3.2 90B
    assert models[4]['model_name'] == 'or-lama-3.2-90b'
    assert (
        models[4]['litellm_params']['model']
        == 'openrouter/meta-llama/llama-3.2-90b-vision-instruct'
    )
    assert models[4]['litellm_params']['api_base'] == 'https://openrouter.ai/api/v1'
    assert models[4]['litellm_params']['api_key'] == 'sk-or-v1-xxx'

    assert router_args['routing_strategy'] == 'usage-based-routing'
    assert router_args['num_retries'] == 3
    assert router_args['cooldown_time'] == 1.0
    assert router_args['allowed_fails'] == 5

    assert hasattr(llm, 'router')
    assert isinstance(llm.router, MagicMock)


@patch('openhands.llm.llm.litellm.Router')
def test_router_content_policy_fallbacks(mock_router, router_content_policy_config):
    mock_router_instance = MagicMock()
    mock_router_instance.completion.return_value = {
        'choices': [{'message': {'content': 'Content policy fallback response'}}]
    }
    mock_router.return_value = mock_router_instance

    llm = LLM(router_content_policy_config)

    response = llm.completion(
        messages=[{'role': 'user', 'content': 'Test content policy'}],
        fallbacks=['fallback-model'],
        mock_response='Content policy fallback response',
    )

    assert (
        response['choices'][0]['message']['content']
        == 'Content policy fallback response'
    )
    assert mock_router_instance.completion.call_count == 1


TOML_EXAMPLE_TEXT = """
[core]
default_agent = "CodeActAgent"

[router_config]
default_model = "or-sonnet-3.5"
routing_strategy = "usage-based-routing"
cooldown_time = 1.0
num_retries = 3
retry_after = 15
timeout = 120
allowed_fails = 5
cache_responses = false

[router_config.cache_kwargs]

[router_config.retry_policy]
exceptions_to_retry = [ "RateLimitError",]
max_retries = 8
retry_after = 15
retry_after_multiplier = 2

[[router_config.models]]
model_name = "gpt-4-vision-enhancements"
[router_config.models.litellm_params]
model = "azure/gpt-4-vision"
api_key = "AZURE_API_KEY"
base_url = "https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions/"
[[router_config.models.litellm_params.dataSources]]
type = "AzureComputerVision"
[router_config.models.litellm_params.dataSources.parameters]
endpoint = "AZURE_VISION_ENHANCE_ENDPOINT"
key = "AZURE_VISION_ENHANCE_KEY"

[[router_config.models]]
model_name = "or-sonnet-3.5"
[router_config.models.litellm_params]
model = "openrouter/anthropic/claude-3.5-sonnet"
api_key = "sk-or-v1-xxx"
max_retries = 3
max_tokens = 100
tpm = 100000
rpm = 10000
timeout = 30

[[router_config.models]]
model_name = "ant-sonnet-3.5"
[router_config.models.litellm_params]
model = "anthropic/claude-3-5-sonnet-20240620"
api_key = "sk-ant-xxx"
max_retries = 3
max_tokens = 100
tpm = 100000
rpm = 10000
timeout = 30

[[router_config.models]]
model_name = "gpt-4o"
[router_config.models.litellm_params]
model = "openrouter/openai/chatgpt-4o-latest"
api_key = "sk-or-v1-xxx"
max_retries = 3
max_tokens = 100
tpm = 100000
rpm = 10000
timeout = 30

[[router_config.models]]
model_name = "or-lama-3.2-90b"
[router_config.models.litellm_params]
model = "openrouter/meta-llama/llama-3.2-90b-vision-instruct"
api_base = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-xxx"
"""
