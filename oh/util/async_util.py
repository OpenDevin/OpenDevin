import asyncio
from typing import Callable, Coroutine, Iterable

GENERAL_TIMEOUT: int = 15


async def async_thread(fn: Callable, *args, **kwargs):
    """
    Shorthand for running a function in the default background thread pool executor
    and awaiting the result
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    return result


async def wait_all(
    iterable: Iterable[Coroutine], timeout: int = GENERAL_TIMEOUT
) -> Iterable:
    """
    Shorthand for waiting for all the coroutines in the iterable given. Creates
    a task for each coroutine
    """
    coroutines = [asyncio.create_task(c) for c in iterable]
    if not coroutines:
        return
    done, pending = await asyncio.wait(coroutines, timeout=timeout)
    if pending:
        raise asyncio.Timeout()
    return (d.result() for d in done)
