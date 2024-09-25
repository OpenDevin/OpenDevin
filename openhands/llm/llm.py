import asyncio
import copy
import os
import time
import warnings
from functools import partial
from typing import Any

from openhands.core.config import LLMConfig
from openhands.runtime.utils.shutdown_listener import should_continue

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    ContentPolicyViolationError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openhands.core.exceptions import (
    LLMResponseError,
    OperationCancelled,
    UserCancelledError,
)
from openhands.core.logger import get_llm_loggers
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.core.metrics import Metrics
from openhands.runtime.utils.shutdown_listener import should_exit

__all__ = ['LLM']

message_separator = '\n\n----------\n\n'

cache_prompting_supported_models = [
    'claude-3-5-sonnet-20240620',
    'claude-3-haiku-20240307',
]

llm_prompt_logger, llm_response_logger = get_llm_loggers().values()


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
        self.metrics = metrics if metrics is not None else Metrics()
        self.cost_metric_supported = True
        self.config = copy.deepcopy(config)

        os.environ['OR_SITE_URL'] = self.config.openrouter_site_url
        os.environ['OR_APP_NAME'] = self.config.openrouter_app_name

        # list of LLM completions (for logging purposes). Each completion is a dict with the following keys:
        # - 'messages': list of messages
        # - 'response': response from the LLM
        self.llm_completions: list[dict[str, Any]] = []

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

        # Tuple of exceptions to retry on
        self.retry_exceptions = (
            APIConnectionError,
            ContentPolicyViolationError,
            InternalServerError,
            OpenAIError,
            RateLimitError,
        )

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

        # This only seems to work with Google as the provider, not with OpenRouter!
        gemini_safety_settings = (
            [
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE',
                },
            ]
            if self.config.model.lower().startswith('gemini')
            else None
        )

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
            **(
                {'safety_settings': gemini_safety_settings}
                if gemini_safety_settings is not None
                else {}
            ),
        )

        if self.vision_is_active():
            logger.debug('LLM: model has vision enabled')

        completion_unwrapped = self._completion

        def log_retry_attempt(retry_state):
            """With before_sleep, this is called before `custom_completion_wait` and
            ONLY if the retry is triggered by an exception."""
            if should_exit():
                raise OperationCancelled(
                    'Operation cancelled.'
                )  # exits the @retry loop
            exception = retry_state.outcome.exception()
            logger.error(
                f'{exception}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
                exc_info=False,
            )

        def custom_completion_wait(retry_state):
            """Custom wait function for litellm completion."""
            if not retry_state:
                return 0
            exception = retry_state.outcome.exception() if retry_state.outcome else None
            if exception is None:
                return 0

            min_wait_time = self.config.retry_min_wait
            max_wait_time = self.config.retry_max_wait

            # for rate limit errors, wait 1 minute by default, max 4 minutes between retries
            exception_type = type(exception).__name__
            logger.error(f'\nexception_type: {exception_type}\n')

            if exception_type == 'RateLimitError':
                min_wait_time = 60
                max_wait_time = 240
            elif exception_type == 'BadRequestError' and exception.response:
                # this should give us the burried, actual error message from
                # the LLM model.
                logger.error(f'\n\nBadRequestError: {exception.response}\n\n')

            # Return the wait time using exponential backoff
            exponential_wait = wait_exponential(
                multiplier=self.config.retry_multiplier,
                min=min_wait_time,
                max=max_wait_time,
            )

            # Call the exponential wait function with retry_state to get the actual wait time
            return exponential_wait(retry_state)

        @retry(
            before_sleep=log_retry_attempt,
            stop=stop_after_attempt(self.config.num_retries),
            reraise=True,
            retry=(
                retry_if_exception_type(self.retry_exceptions)
                & retry_if_not_exception_type(OperationCancelled)
            ),
            wait=custom_completion_wait,
        )
        def wrapper(*args, **kwargs):
            """Wrapper for the litellm completion function. Logs the input and output of the completion function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1] if len(args) > 1 else []

            # this serves to prevent empty messages and logging the messages
            debug_message = self._get_debug_message(messages)

            if self.is_caching_prompt_active():
                # Anthropic-specific prompt caching
                if 'claude-3' in self.config.model:
                    kwargs['extra_headers'] = {
                        'anthropic-beta': 'prompt-caching-2024-07-31',
                    }

            # skip if messages is empty (thus debug_message is empty)
            if debug_message:
                llm_prompt_logger.debug(debug_message)
                resp = completion_unwrapped(*args, **kwargs)
            else:
                logger.debug('No completion messages!')
                resp = {'choices': [{'message': {'content': ''}}]}

            if self.config.log_completions:
                self.llm_completions.append(
                    {
                        'messages': messages,
                        'response': resp,
                        'timestamp': time.time(),
                        'cost': self.completion_cost(resp),
                    }
                )

            # log the response
            message_back = resp['choices'][0]['message']['content']
            if message_back:
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
            drop_params=self.config.drop_params,
            **(
                {'safety_settings': gemini_safety_settings}
                if gemini_safety_settings is not None
                else {}
            ),
        )

        async_completion_unwrapped = self._async_completion

        @retry(
            before_sleep=log_retry_attempt,
            stop=stop_after_attempt(self.config.num_retries),
            reraise=True,
            retry=(
                retry_if_exception_type(self.retry_exceptions)
                & retry_if_not_exception_type(OperationCancelled)
            ),
            wait=custom_completion_wait,
        )
        async def async_completion_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1] if len(args) > 1 else []

            # this serves to prevent empty messages and logging the messages
            debug_message = self._get_debug_message(messages)

            async def check_stopped():
                while should_continue():
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
                if debug_message:
                    llm_prompt_logger.debug(debug_message)
                    resp = await async_completion_unwrapped(*args, **kwargs)
                else:
                    logger.debug('No completion messages!')
                    resp = {'choices': [{'message': {'content': ''}}]}

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
            except (
                APIConnectionError,
                ContentPolicyViolationError,
                InternalServerError,
                NotFoundError,
                OpenAIError,
                RateLimitError,
                ServiceUnavailableError,
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
            before_sleep=log_retry_attempt,
            stop=stop_after_attempt(self.config.num_retries),
            reraise=True,
            retry=(
                retry_if_exception_type(self.retry_exceptions)
                & retry_if_not_exception_type(OperationCancelled)
            ),
            wait=custom_completion_wait,
        )
        async def async_acompletion_stream_wrapper(*args, **kwargs):
            """Async wrapper for the litellm acompletion with streaming function."""
            # some callers might just send the messages directly
            if 'messages' in kwargs:
                messages = kwargs['messages']
            else:
                messages = args[1] if len(args) > 1 else []

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
            except (
                APIConnectionError,
                ContentPolicyViolationError,
                InternalServerError,
                NotFoundError,
                OpenAIError,
                RateLimitError,
                ServiceUnavailableError,
            ) as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                if kwargs.get('stream', False):
                    await asyncio.sleep(0.1)

        self._async_completion = async_completion_wrapper  # type: ignore
        self._async_streaming_completion = async_acompletion_stream_wrapper  # type: ignore

    def _get_debug_message(self, messages):
        if not messages:
            return ''

        messages = messages if isinstance(messages, list) else [messages]
        return message_separator.join(
            self._format_message_content(msg) for msg in messages if msg['content']
        )

    def _format_message_content(self, message):
        content = message['content']
        if isinstance(content, list):
            return self._format_list_content(content)
        return str(content)

    def _format_list_content(self, content_list):
        return '\n'.join(
            self._format_content_element(element) for element in content_list
        )

    def _format_content_element(self, element):
        if isinstance(element, dict):
            if 'text' in element:
                return element['text']
            if (
                self.vision_is_active()
                and 'image_url' in element
                and 'url' in element['image_url']
            ):
                return element['image_url']['url']
        return str(element)

    async def _call_acompletion(self, *args, **kwargs):
        """This is a wrapper for the litellm acompletion function which
        makes it mockable for testing.
        """
        return await litellm.acompletion(*args, **kwargs)

    @property
    def completion(self):
        """Decorator for the litellm completion function.

        Check the complete documentation at https://litellm.vercel.app/docs/completion
        """
        try:
            return self._completion
        except Exception as e:
            raise LLMResponseError(e)

    @property
    def async_completion(self):
        """Decorator for the async litellm acompletion function.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        try:
            return self._async_completion
        except Exception as e:
            raise LLMResponseError(e)

    @property
    def async_streaming_completion(self):
        """Decorator for the async litellm acompletion function with streaming.

        Check the complete documentation at https://litellm.vercel.app/docs/providers/ollama#example-usage---streaming--acompletion
        """
        try:
            return self._async_streaming_completion
        except Exception as e:
            raise LLMResponseError(e)

    def vision_is_active(self):
        return not self.config.disable_vision and self._supports_vision()

    def _supports_vision(self):
        """Acquire from litellm if model is vision capable.

        Returns:
            bool: True if model is vision capable. If model is not supported by litellm, it will return False.
        """
        try:
            return litellm.supports_vision(self.config.model)
        except Exception:
            return False

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is enabled and supported for current model.

        Returns:
            boolean: True if prompt caching is active for the given model.
        """
        return self.config.caching_prompt is True and any(
            model in self.config.model for model in cache_prompting_supported_models
        )

    def _post_completion(self, response) -> None:
        """Post-process the completion response."""
        try:
            cur_cost = self.completion_cost(response)
        except Exception:
            cur_cost = 0

        stats = ''
        if self.cost_metric_supported:
            stats = 'Cost: %.2f USD | Accumulated Cost: %.2f USD\n' % (
                cur_cost,
                self.metrics.accumulated_cost,
            )

        usage = response.get('usage')

        if usage:
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
            logger.debug(f'Using custom cost per token: {cost_per_token}')
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
        self.llm_completions = []

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            return [messages.model_dump()]
        return [message.model_dump() for message in messages]
