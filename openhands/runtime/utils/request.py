from typing import Any, Callable, Type

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    HTTPError,
    Timeout,
)
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)
from urllib3.exceptions import IncompleteRead

from openhands.utils.tenacity_stop import stop_if_should_exit


def is_server_error(exception):
    return (
        isinstance(exception, requests.HTTPError)
        and exception.response.status_code >= 500
    )


def is_404_error(exception):
    return (
        isinstance(exception, requests.HTTPError)
        and exception.response.status_code == 404
    )


DEFAULT_RETRY_EXCEPTIONS = [
    ConnectionError,
    Timeout,
    HTTPError,
    IncompleteRead,
    ChunkedEncodingError,
]


def send_request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    retry_exceptions: list[Type[Exception]] | None = None,
    retry_fns: list[Callable[[Exception], bool]] | None = None,
    **kwargs: Any,
) -> requests.Response:
    exceptions_to_catch = retry_exceptions or DEFAULT_RETRY_EXCEPTIONS
    retry_condition = retry_if_exception_type(
        tuple(exceptions_to_catch)
    ) | retry_if_exception(is_server_error)
    if retry_fns is not None:
        for fn in retry_fns:
            retry_condition |= retry_if_exception(fn)
    # wait a few more seconds to get the timeout error from client side
    kwargs['timeout'] = timeout + 10

    @retry(
        stop=stop_after_delay(timeout) | stop_if_should_exit(),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_condition,
        reraise=True,
    )
    def _send_request_with_retry():
        response = session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    return _send_request_with_retry()
