from dataclasses import dataclass

import requests

from opendevin.observation import BrowserOutputObservation

from .base import ExecutableAction


@dataclass
class BrowseURLAction(ExecutableAction):
    url: str

    def run(self, *args, **kwargs) -> BrowserOutputObservation:
        response = requests.get(self.url)
        return BrowserOutputObservation(content=response.text, url=self.url)

    @property
    def message(self) -> str:
        return f"Browsing URL: {self.url}"
