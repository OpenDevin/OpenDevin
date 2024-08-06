import asyncio
import copy
import warnings
from functools import partial

from opendevin.core.config import LLMConfig

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    ContentPolicyViolationError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from opendevin.core.exceptions import UserCancelledError
from opendevin.core.logger import llm_prompt_logger, llm_response_logger
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.metrics import Metrics

__all__ = ['LLM']

message_separator = '\n\n----------\n\n'


class LLM:
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
            config: The LLM configuration
        """
        self.config = copy.deepcopy(config)
        self.metrics = metrics if metrics is not None else Metrics()
        self.cost_metric_supported = True

        # Set up config attributes with default values to prevent AttributeError
        LLMConfig.set_missing_attributes(self.config)

        # litellm actually uses base Exception here for unknown model
        self.model_info = None
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
                # Max input tokens for gpt3.5, so this is a safe fallback for any potentially viable model
                self.config.max_input_tokens = 4096

        if self.config.max_output_tokens is None:
            if (
                self.model_info is not None
                and 'max_output_tokens' in self.model_info
                and isinstance(self.model_info['max_output_tokens'], int)
            ):
                self.config.max_output_tokens = self.model_info['max_output_tokens']
            else:
                # Max output tokens for gpt3.5, so this is a safe fallback for any potentially viable model
                self.config.max_output_tokens = 1024

        if self.config.drop_params:
            litellm.drop_params = self.config.drop_params

        # Set up config attributes with default values to prevent AttributeError
        self.config.api_version = getattr(self.config, 'api_version', None)
        self.config.custom_llm_provider = getattr(
            self.config, 'custom_llm_provider', None
        )
        self.config.max_output_tokens = getattr(self.config, 'max_output_tokens', None)
        self.config.timeout = getattr(self.config, 'timeout', None)
        self.config.temperature = getattr(self.config, 'temperature', 0.0)
        self.config.top_p = getattr(self.config, 'top_p', 0.9)

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
        )

        completion_unwrapped = self._completion

        def attempt_on_error(retry_state):
            logger.error(
                f'{retry_state.outcome.exception()}. Attempt #{retry_state.attempt_number} | You can customize these settings in the configuration.',
                exc_info=False,
            )
            return None

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=attempt_on_error,
        )
        def wrapper(*args, **kwargs):
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            # log the prompt
            debug_message = ''
            for message in messages:
                content = message['content']

                if isinstance(content, list):
                    for element in content:
                        if isinstance(element, dict):
                            if 'text' in element:
                                content_str = element['text'].strip()
                            elif (
                                'image_url' in element and 'url' in element['image_url']
                            ):
                                content_str = element['image_url']['url']
                            else:
                                content_str = str(element)
                        else:
                            content_str = str(element)

                        debug_message += message_separator + content_str
                else:
                    content_str = str(content)

                debug_message += message_separator + content_str

            llm_prompt_logger.debug(debug_message)

            # skip if messages is empty (thus debug_message is empty)
            if debug_message:
                resp = completion_unwrapped(*args, **kwargs)
            else:
                resp = {'choices': [{'message': {'content': ''}}]}

            # log the response
            message_back = resp['choices'][0]['message']['content']
            llm_response_logger.debug(message_back)

            # post-process to log costs
            self._post_completion(resp)

            return resp

        self._completion = wrapper  # type: ignore

        # Async version
        self._async_completion = partial(
            self._call_acompletion,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=True,
        )

        async_completion_unwrapped = self._async_completion

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=attempt_on_error,
        )
        async def async_completion_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            # log the prompt
            debug_message = ''
            for message in messages:
                content = message['content']

                if isinstance(content, list):
                    for element in content:
                        if isinstance(element, dict):
                            if 'text' in element:
                                content_str = element['text']
                            elif (
                                'image_url' in element and 'url' in element['image_url']
                            ):
                                content_str = element['image_url']['url']
                            else:
                                content_str = str(element)
                        else:
                            content_str = str(element)

                        debug_message += message_separator + content_str
                else:
                    content_str = str(content)

                debug_message += message_separator + content_str

            llm_prompt_logger.debug(debug_message)

            async def check_stopped():
                while True:
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError('LLM request cancelled by user')
                    await asyncio.sleep(0.1)

            stop_check_task = asyncio.create_task(check_stopped())

            try:
                # Directly call and await litellm_acompletion
                resp = await async_completion_unwrapped(*args, **kwargs)

                # skip if messages is empty (thus debug_message is empty)
                if debug_message:
                    message_back = resp['choices'][0]['message']['content']
                    llm_response_logger.debug(message_back)
                else:
                    resp = {'choices': [{'message': {'content': ''}}]}
                self._post_completion(resp)

                # We do not support streaming in this method, thus return resp
                return resp

            except UserCancelledError:
                logger.info('LLM request cancelled by user.')
                raise
            except OpenAIError as e:
                logger.error(f'OpenAIError occurred:\n{e}')
                raise
            except (
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
                InternalServerError,
            ) as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                await asyncio.sleep(0.1)
                stop_check_task.cancel()
                try:
                    await stop_check_task
                except asyncio.CancelledError:
                    pass

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.num_retries),
            wait=wait_random_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (
                    RateLimitError,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    ContentPolicyViolationError,
                )
            ),
            after=attempt_on_error,
        )
        async def async_acompletion_stream_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion with streaming function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1]

            # log the prompt
            debug_message = ''
            for message in messages:
                debug_message += message_separator + message['content']
            llm_prompt_logger.debug(debug_message)

            try:
                # Directly call and await litellm_acompletion
                resp = await async_completion_unwrapped(*args, **kwargs)

                # For streaming we iterate over the chunks
                async for chunk in resp:
                    # Check for cancellation before yielding the chunk
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError(
                            'LLM request cancelled due to CANCELLED state'
                        )
                    # with streaming, it is "delta", not "message"!
                    message_back = chunk['choices'][0]['delta']['content']
                    llm_response_logger.debug(message_back)
                    self._post_completion(chunk)

                    yield chunk

            except UserCancelledError:
                logger.info('LLM request cancelled by user.')
                raise
            except OpenAIError as e:
                logger.error(f'OpenAIError occurred:\n{e}')
                raise
            except (
                RateLimitError,
                APIConnectionError,
                ServiceUnavailableError,
                InternalServerError,
            ) as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                if kwargs.get('stream', False):
                    await asyncio.sleep(0.1)

        self._async_completion = async_completion_wrapper  # type: ignore
        self._async_streaming_completion = async_acompletion_stream_wrapper  # type: ignore

    async def _call_acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **kwargs)

    @property
    def completion(self):
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        return self._completion

    @property
    def async_completion(self):
        """Decorator for the async litellm acompletion function.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        return self._async_completion

    @property
    def async_streaming_completion(self):
        """Decorator for the async litellm acompletion function with streaming.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        return self._async_streaming_completion

    def supports_vision(self):
        return litellm.supports_vision(self.config.model)

    def _post_completion(self, response: str) -> None:
        """Post-process the completion response."""
        try:
            cur_cost = self.completion_cost(response)
        except Exception:
            cur_cost = 0
        if self.cost_metric_supported:
            logger.info(
                'Cost: %.2f USD | Accumulated Cost: %.2f USD',
                cur_cost,
                self.metrics.accumulated_cost,
            )

    def get_token_count(self, messages):
        """Get the number of tokens in a list of messages.

        Args:
            messages (list): A list of messages.

        Returns:
            int: The number of tokens.
        """
        return litellm.token_counter(model=self.config.model, messages=messages)

    def is_local(self):
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

    def completion_cost(self, response):
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

        if not self.is_local():
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
