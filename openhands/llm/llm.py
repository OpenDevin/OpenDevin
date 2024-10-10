import copy
import time
import warnings
from functools import partial
from typing import Any

from openhands.core.config import LLMConfig

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
from litellm import ModelInfo, Router
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken, ModelResponse, Usage

from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.core.metrics import Metrics
from openhands.llm.debug_mixin import DebugMixin
from openhands.llm.retry_mixin import RetryMixin

__all__ = ['LLM']

# tuple of exceptions to retry on
LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIConnectionError,
    # FIXME: APIError is useful on 502 from a proxy for example,
    # but it also retries on other errors that are permanent
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

# cache prompt supporting models
# remove this when we gemini and deepseek are supported
CACHE_PROMPT_SUPPORTED_MODELS = [
    'claude-3-5-sonnet-20240620',
    'claude-3-haiku-20240307',
    'claude-3-opus-20240229',
    'anthropic/claude-3-opus-20240229',
    'anthropic/claude-3-haiku-20240307',
    'anthropic/claude-3-5-sonnet-20240620',
]


class LLM(RetryMixin, DebugMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
    ):
        """Initializes the LLM. If LLMConfig is passed, its values will be the fallback.

        Passing simple parameters always overrides config.

        Args:
            config: The LLM configuration.
            metrics: The metrics to use.
        """
        self.metrics: Metrics = metrics if metrics is not None else Metrics()
        self.cost_metric_supported: bool = True
        self.config: LLMConfig = copy.deepcopy(config)
        self.router: Router | None = None

        # list of LLM completions (for logging purposes). Each completion is a dict with the following keys:
        # - 'messages': list of messages
        # - 'response': response from the LLM
        self.llm_completions: list[dict[str, Any]] = []

        # litellm actually uses base Exception here for unknown model
        self.model_info: ModelInfo | None = None
        if self.config.model:
            try:
                if self.config.model.startswith('openrouter'):
                    self.model_info = litellm.get_model_info(self.config.model)
                else:
                    self.model_info = litellm.get_model_info(
                        self.config.model.split(':')[0]
                    )
            # noinspection PyBroadException
            except Exception as e:
                logger.warning(f'Could not get model info for {config.model}:\n{e}')

        # Set the max tokens in an LM-specific way if not set
        if self.config.max_input_tokens is None:
            if (
                self.model_info is not None
                and 'max_input_tokens' in self.model_info
                and isinstance(self.model_info['max_input_tokens'], int)
            ):
                self.config.max_input_tokens = self.model_info['max_input_tokens']
            else:
                # Safe fallback for any potentially viable model
                self.config.max_input_tokens = 4096

        if self.config.max_output_tokens is None:
            # Safe default for any potentially viable model
            self.config.max_output_tokens = 4096
            if self.model_info is not None:
                # max_output_tokens has precedence over max_tokens, if either exists.
                # litellm has models with both, one or none of these 2 parameters!
                if 'max_output_tokens' in self.model_info and isinstance(
                    self.model_info['max_output_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_output_tokens']
                elif 'max_tokens' in self.model_info and isinstance(
                    self.model_info['max_tokens'], int
                ):
                    self.config.max_output_tokens = self.model_info['max_tokens']

        if self.vision_is_active():
            logger.debug('LLM: model has vision enabled')
        if self.is_caching_prompt_active():
            logger.debug('LLM: caching prompt enabled')

        router_config: dict[str, Any] | None = None

        # Set up router configuration
        if self.config.router_config and self.config.router_config.model_list:
            model_list = [
                {
                    'model_name': model.model_name,
                    'litellm_params': model.litellm_params,
                }
                for model in self.config.router_config.model_list
            ]
            router_config = {
                'model_list': model_list,
                'routing_strategy': self.config.router_config.routing_strategy,
                'num_retries': self.config.router_config.num_retries,
                'cooldown_time': self.config.router_config.cooldown_time,
                'allowed_fails': self.config.router_config.allowed_fails,
                'cache_responses': self.config.router_config.cache_responses,
                'cache_kwargs': self.config.router_config.cache_kwargs,
                'retry_policy': self.config.router_config.retry_policy,
                'default_fallbacks': self.config.router_config.default_fallbacks,
            }
            self.router = litellm.Router(**router_config)
            completion_func = self.router.completion
        else:
            # If no router models are specified, use the default model config
            self._completion = partial(
                litellm_completion,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                api_version=self.config.api_version,
                custom_llm_provider=self.config.custom_llm_provider,
                max_tokens=self.config.max_output_tokens,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                drop_params=self.config.drop_params,
            )
            completion_unwrapped = self._completion

        def wrapper(*args, **kwargs):
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            messages: list[dict[str, Any]] | dict[str, Any] = []

            # some callers might send the model and messages directly
            # litellm allows positional args, like completion(model, messages, **kwargs)
            if len(args) > 1:
                # ignore the first argument if it's provided (it would be the model)
                # design wise: we don't allow overriding the configured values
                # implementation wise: the partial function set the model as a kwarg already
                # as well as other kwargs
                messages = args[1] if len(args) > 1 else args[0]
                kwargs['messages'] = messages

                # remove the first args, they're sent in kwargs
                args = args[2:]
            elif 'messages' in kwargs:
                messages = kwargs['messages']

            # ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            # log the entire LLM prompt
            self.log_prompt(messages)

            if self.is_caching_prompt_active():
                # Anthropic-specific prompt caching
                if 'claude-3' in self.config.model:
                    kwargs['extra_headers'] = {
                        'anthropic-beta': 'prompt-caching-2024-07-31',
                    }

            # Call the completion function (either router.completion or litellm_completion)
            # we don't support streaming here, thus we get a ModelResponse
            if self.router and self.config.router_config:
                default_model = self.config.router_config.model_list[0].model_name
                model = kwargs.pop('model', default_model)
                resp: ModelResponse = completion_func(model, **kwargs)
            else:
                resp = completion_unwrapped(*args, **kwargs)

            # log for evals or other scripts that need the raw completion
            if self.config.log_completions:
                self.llm_completions.append(
                    {
                        'messages': messages,
                        'response': resp,
                        'timestamp': time.time(),
                        'cost': self._completion_cost(resp),
                    }
                )

            message_back: str = resp['choices'][0]['message']['content']

            # log the LLM response
            self.log_response(message_back)

            # post-process the response
            self._post_completion(resp)

            return resp

        if not (self.config.router_config and self.config.router_config.model_list):
            wrapper = self.retry_decorator(
                num_retries=self.config.num_retries,
                retry_exceptions=LLM_RETRY_EXCEPTIONS,
                retry_min_wait=self.config.retry_min_wait,
                retry_max_wait=self.config.retry_max_wait,
                retry_multiplier=self.config.retry_multiplier,
            )(wrapper)

        self._completion = partial(wrapper)

    @property
    def completion(self):
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion

    def vision_is_active(self):
        return not self.config.disable_vision and self._supports_vision()

    def _supports_vision(self):
        """Acquire from litellm if model is vision capable.

        Returns:
            bool: True if model is vision capable. If model is not supported by litellm, it will return False.
        """
        # litellm.supports_vision currently returns False for 'openai/gpt-...' or 'anthropic/claude-...' (with prefixes)
        # but model_info will have the correct value for some reason.
        # we can go with it, but we will need to keep an eye if model_info is correct for Vertex or other providers
        # remove when litellm is updated to fix https://github.com/BerriAI/litellm/issues/5608
        return litellm.supports_vision(self.config.model) or (
            self.model_info is not None
            and self.model_info.get('supports_vision', False)
        )

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is supported and enabled for current model.

        Returns:
            boolean: True if prompt caching is supported and enabled for the given model.
        """
        return (
            self.config.caching_prompt is True
            and self.model_info is not None
            and self.model_info.get('supports_prompt_caching', False)
            and self.config.model in CACHE_PROMPT_SUPPORTED_MODELS
        )

    def _post_completion(self, response: ModelResponse) -> None:
        """Post-process the completion response.

        Logs the cost and usage stats of the completion call.
        """
        try:
            cur_cost = self._completion_cost(response)
        except Exception:
            cur_cost = 0

        stats = ''
        if self.cost_metric_supported:
            # keep track of the cost
            stats = 'Cost: %.2f USD | Accumulated Cost: %.2f USD\n' % (
                cur_cost,
                self.metrics.accumulated_cost,
            )

        usage: Usage | None = response.get('usage')

        if usage:
            # keep track of the input and output tokens
            input_tokens = usage.get('prompt_tokens')
            output_tokens = usage.get('completion_tokens')

            if input_tokens:
                stats += 'Input tokens: ' + str(input_tokens)

            if output_tokens:
                stats += (
                    (' | ' if input_tokens else '')
                    + 'Output tokens: '
                    + str(output_tokens)
                    + '\n'
                )

            # read the prompt caching status as received from the provider
            model_extra = usage.get('model_extra', {})

            cache_creation_input_tokens = model_extra.get('cache_creation_input_tokens')
            if cache_creation_input_tokens:
                stats += (
                    'Input tokens (cache write): '
                    + str(cache_creation_input_tokens)
                    + '\n'
                )

            cache_read_input_tokens = model_extra.get('cache_read_input_tokens')
            if cache_read_input_tokens:
                stats += (
                    'Input tokens (cache read): ' + str(cache_read_input_tokens) + '\n'
                )

        # log the stats
        if stats:
            logger.info(stats)

    def get_token_count(self, messages):
        """Get the number of tokens in a list of messages.

        Args:
            messages (list): A list of messages.

        Returns:
            int: The number of tokens.
        """
        try:
            return litellm.token_counter(model=self.config.model, messages=messages)
        except Exception:
            # TODO: this is to limit logspam in case token count is not supported
            return 0

    def _is_local(self):
        """Determines if the system is using a locally running LLM.

        Returns:
            boolean: True if executing a local model.
        """
        if self.config.base_url is not None:
            for substring in ['localhost', '127.0.0.1' '0.0.0.0']:
                if substring in self.config.base_url:
                    return True
        elif self.config.model is not None:
            if self.config.model.startswith('ollama'):
                return True
        return False

    def _completion_cost(self, response):
        """Calculate the cost of a completion response based on the model.  Local models are treated as free.
        Add the current cost into total cost in metrics.

        Args:
            response: A response from a model invocation.

        Returns:
            number: The cost of the response.
        """
        if not self.cost_metric_supported:
            return 0.0

        extra_kwargs = {}
        if (
            self.config.input_cost_per_token is not None
            and self.config.output_cost_per_token is not None
        ):
            cost_per_token = CostPerToken(
                input_cost_per_token=self.config.input_cost_per_token,
                output_cost_per_token=self.config.output_cost_per_token,
            )
            logger.info(f'Using custom cost per token: {cost_per_token}')
            extra_kwargs['custom_cost_per_token'] = cost_per_token

        if not self._is_local():
            try:
                cost = litellm_completion_cost(
                    completion_response=response, **extra_kwargs
                )
                self.metrics.add_cost(cost)
                return cost
            except Exception:
                self.cost_metric_supported = False
                logger.warning('Cost calculation not supported for this model.')
        return 0.0

    def __str__(self):
        if self.config.api_version:
            return f'LLM(model={self.config.model}, api_version={self.config.api_version}, base_url={self.config.base_url})'
        elif self.config.base_url:
            return f'LLM(model={self.config.model}, base_url={self.config.base_url})'
        return f'LLM(model={self.config.model})'

    def __repr__(self):
        return str(self)

    def reset(self):
        self.metrics = Metrics()
        self.llm_completions = []

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]

        # set flags to know how to serialize the messages
        for message in messages:
            message.cache_enabled = self.is_caching_prompt_active()
            message.vision_enabled = self.vision_is_active()

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]
