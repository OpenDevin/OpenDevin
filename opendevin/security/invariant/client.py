import time
from typing import Any, Optional, Tuple, Union, List, Dict

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout


class InvariantClient:
    timeout: int = 120

    def __init__(self, server_url: str, session_id: Optional[str] = None):
        self.server = server_url
        self.session_id, err = self._create_session(session_id)
        if err:
            raise RuntimeError(f'Failed to create session: {err}')
        self.Policy = self._Policy(self)
        self.Monitor = self._Monitor(self)

    def _create_session(
        self, session_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Exception]]:
        elapsed = 0
        while elapsed < self.timeout:
            try:
                if session_id:
                    response = requests.get(
                        f'{self.server}/session/new?session_id={session_id}', timeout=60
                    )
                else:
                    response = requests.get(f'{self.server}/session/new', timeout=60)
                response.raise_for_status()
                return response.json().get('id'), None
            except (ConnectionError, Timeout):
                elapsed += 1
                time.sleep(1)
            except HTTPError as http_err:
                return None, http_err
            except Exception as err:
                return None, err
        return None, ConnectionError('Connection timed out')

    def close_session(self) -> Union[None, Exception]:
        try:
            response = requests.delete(
                f'{self.server}/session/?session_id={self.session_id}', timeout=60
            )
            response.raise_for_status()
        except (ConnectionError, Timeout, HTTPError) as err:
            return err
        return None

    class _Policy:
        def __init__(self, invariant):
            self.server = invariant.server
            self.session_id = invariant.session_id

        def _create_policy(self, rule: str) -> Tuple[Optional[str], Optional[Exception]]:
            try:
                response = requests.post(
                    f'{self.server}/policy/new?session_id={self.session_id}',
                    json={'rule': rule},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json().get('policy_id'), None
            except (ConnectionError, Timeout, HTTPError) as err:
                return None, err

        def get_template(self) -> Tuple[Optional[str], Optional[Exception]]:
            try:
                response = requests.get(
                    f'{self.server}/policy/template',
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, Timeout, HTTPError) as err:
                return None, err

        def from_string(self, rule: str):
            policy_id, err = self._create_policy(rule)
            if err:
                raise err
            self.policy_id = policy_id
            return self

        def analyze(self, trace: List[Dict]) -> Union[Any, Exception]:
            try:
                response = requests.post(
                    f'{self.server}/policy/{self.policy_id}/analyze?session_id={self.session_id}',
                    json={'trace': trace},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, Timeout, HTTPError) as err:
                return None, err

    class _Monitor:
        def __init__(self, invariant):
            self.server = invariant.server
            self.session_id = invariant.session_id
            self.policy = ''

        def _create_monitor(self, rule: str) -> Tuple[Optional[str], Optional[Exception]]:
            try:
                response = requests.post(
                    f'{self.server}/monitor/new?session_id={self.session_id}',
                    json={'rule': rule},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json().get('monitor_id'), None
            except (ConnectionError, Timeout, HTTPError) as err:
                return None, err

        def from_string(self, rule: str):
            monitor_id, err = self._create_monitor(rule)
            if err:
                raise err
            self.monitor_id = monitor_id
            self.policy = rule
            return self

        def check(self, past_events: List[Dict], pending_events: List[Dict]) -> Union[Any, Exception]:
            try:
                response = requests.post(
                    f'{self.server}/monitor/{self.monitor_id}/check?session_id={self.session_id}',
                    json={"past_events": past_events, "pending_events": pending_events},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, Timeout, HTTPError) as err:
                return None, err
