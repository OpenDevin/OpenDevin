from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openhands.core.exceptions import OperationCancelled
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.utils.shutdown_listener import should_exit


class RetryMixin:
    """Mixin class for retry logic."""

    def retry_decorator(self, **kwargs):
        """
        Create a retry decorator with customizable parameters.

        Args:
            **kwargs: Keyword arguments to override default retry behavior.
                      Keys: num_retries, retry_exceptions, retry_min_wait, retry_max_wait, retry_multiplier

        Returns:
            A retry decorator with the specified or default parameters.
        """
        num_retries = kwargs.get('num_retries')
        retry_exceptions = kwargs.get('retry_exceptions')
        retry_min_wait = kwargs.get('retry_min_wait')
        retry_max_wait = kwargs.get('retry_max_wait')
        retry_multiplier = kwargs.get('retry_multiplier')

        return retry(
            before_sleep=self.log_retry_attempt,
            stop=stop_after_attempt(num_retries),
            reraise=True,
            retry=(retry_if_exception_type(retry_exceptions)),
            wait=wait_exponential(
                multiplier=retry_multiplier,
                min=retry_min_wait,
                max=retry_max_wait,
            ),
        )

    def log_retry_attempt(self, retry_state):
        """Log retry attempts."""
        if should_exit():
            raise OperationCancelled('Operation cancelled.')  # exits the @retry loop
        exception = retry_state.outcome.exception()
        logger.error(
            f'{exception}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
            exc_info=False,
        )
