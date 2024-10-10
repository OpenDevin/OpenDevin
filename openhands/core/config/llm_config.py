import os
from dataclasses import dataclass, fields

from openhands.core.config.config_utils import get_field_info
from openhands.core.config.router_config import RouterConfig

LLM_SENSITIVE_FIELDS = ['api_key', 'aws_access_key_id', 'aws_secret_access_key']


@dataclass
class LLMConfig:
    """Configuration for the LLM model.

    Attributes:
        model: The model to use.
        api_key: The API key to use.
        base_url: The base URL for the API. This is necessary for local LLMs. It is also used for Azure embeddings.
        api_version: The version of the API.
        embedding_model: The embedding model to use.
        embedding_base_url: The base URL for the embedding API.
        embedding_deployment_name: The name of the deployment for the embedding API. This is used for Azure OpenAI.
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_region_name: The AWS region name.
        num_retries: The number of retries to attempt.
        retry_multiplier: The multiplier for the exponential backoff.
        retry_min_wait: The minimum time to wait between retries, in seconds. This is exponential backoff minimum. For models with very low limits, this can be set to 15-20.
        retry_max_wait: The maximum time to wait between retries, in seconds. This is exponential backoff maximum.
        timeout: The timeout for the API.
        max_message_chars: The approximate max number of characters in the content of an event included in the prompt to the LLM. Larger observations are truncated.
        temperature: The temperature for the API.
        top_p: The top p for the API.
        custom_llm_provider: The custom LLM provider to use. This is undocumented in openhands, and normally not used. It is documented on the litellm side.
        max_input_tokens: The maximum number of input tokens. Note that this is currently unused, and the value at runtime is actually the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).
        max_output_tokens: The maximum number of output tokens. This is sent to the LLM.
        input_cost_per_token: The cost per input token. This will be used in logs for the user to check.
        output_cost_per_token: The cost per output token. This will be used in logs for the user to check.
        ollama_base_url: The base URL for the OLLAMA API used to retrieve model names.
        drop_params: Drop any unmapped (unsupported) params without causing an exception.
        disable_vision: If model is vision capable, this option allows to disable image processing (useful for cost reduction).
        caching_prompt: Use the prompt caching feature if provided by the LLM and supported by the provider.
        log_completions: Whether to log LLM completions to the state.
        router_config: The router configuration with model list, routing strategy, etc.; gets assigned by load_from_toml.
    """

    model: str = 'gpt-4o'
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    embedding_model: str = 'local'
    embedding_base_url: str | None = None
    embedding_deployment_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str | None = None
    openrouter_site_url: str = 'https://docs.all-hands.dev/'
    openrouter_app_name: str = 'OpenHands'
    num_retries: int = 8
    retry_multiplier: float = 2
    retry_min_wait: int = 15
    retry_max_wait: int = 120
    timeout: int | None = None
    max_message_chars: int = 10_000  # maximum number of characters in an observation's content when sent to the llm
    temperature: float = 0.0
    top_p: float = 1.0
    custom_llm_provider: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    ollama_base_url: str | None = None
    drop_params: bool = True
    disable_vision: bool | None = None
    caching_prompt: bool = True
    log_completions: bool = False

    # Single router configuration, set up by load_from_toml
    router_config: RouterConfig | None = None

    def defaults_to_dict(self) -> dict:
        """Serialize fields to a dict for the frontend, including type hints, defaults, and whether it's optional."""
        result = {}
        for f in fields(self):
            result[f.name] = get_field_info(f)
        return result

    def to_litellm_params(self) -> dict:
        """Gather necessary attributes for litellm_params, excluding specific unsupported fields."""
        excluded_fields = {
            'log_completions',
            'caching_prompt',
            'disable_vision',
            'ollama_base_url',
            'openrouter_site_url',
            'openrouter_app_name',
            'embedding_model',
            'embedding_base_url',
            'embedding_deployment_name',
            'retry_multiplier',
            'retry_min_wait',
            'retry_max_wait',
            'max_message_chars',
        }
        # Collect all attributes except those in excluded_fields
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in excluded_fields
        }

    def __post_init__(self):
        """Post-initialization hook to assign OpenRouter-related variables to environment variables.
        This ensures that these values are accessible to litellm at runtime.
        """

        # Assign OpenRouter-specific variables to environment variables
        if self.openrouter_site_url:
            os.environ['OR_SITE_URL'] = self.openrouter_site_url
        if self.openrouter_app_name:
            os.environ['OR_APP_NAME'] = self.openrouter_app_name

    def __str__(self):
        attr_str = []
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, f.name)

            if attr_name in LLM_SENSITIVE_FIELDS:
                attr_value = '******' if attr_value else None

            attr_str.append(f'{attr_name}={repr(attr_value)}')

        return f"LLMConfig({', '.join(attr_str)})"

    def __repr__(self):
        return self.__str__()

    def to_safe_dict(self) -> dict:
        """Return a dict with the sensitive fields replaced with ******."""
        ret = self.__dict__.copy()
        for k, v in ret.items():
            if k in LLM_SENSITIVE_FIELDS:
                ret[k] = '******' if v else None
        return ret
