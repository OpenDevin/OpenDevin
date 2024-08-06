import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TypeVar  # noqa: F401

logger = logging.getLogger(__name__)

T = TypeVar('T')


def async_to_sync(func):
    # type: (Callable[..., Any]) -> Callable[..., Any]
    """
    A decorator that allows an asynchronous function to be called in a synchronous context.

    This decorator wraps an asynchronous function and ensures that it can be executed
    synchronously by running the event loop until the coroutine completes. If the current
    event loop is closed or does not exist, a new event loop is created.

    Args:
        func: The function to be wrapped.

    Returns:
        A synchronous callable that executes the function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # type: (*Any, **Any) -> Any

        async def run_async():
            result = await func(*args, **kwargs)
            return result

        def run_sync():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError('Event loop is closed')
            except RuntimeError:
                logger.debug('Creating a new event loop')
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(run_async(), loop).result()
            else:
                return loop.run_until_complete(run_async())

        # Check if we're in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return run_async()
            else:
                return run_sync()
        except RuntimeError:
            logger.debug('No event loop, running sync')
            return run_sync()

    return wrapper
