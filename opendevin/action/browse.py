import requests

from dataclasses import dataclass
from opendevin.observation import BrowserOutputObservation

from .base import ExecutableAction

@dataclass
class BrowseURLAction(ExecutableAction):
    url: str
    action: str = "browse"

    def run(self, *args, **kwargs) -> BrowserOutputObservation:
        try:
            response = requests.get(self.url)
            return BrowserOutputObservation(
                content=response.text,
                status_code=response.status_code,
                url=self.url
            )
        except requests.exceptions.RequestException as e:
            return BrowserOutputObservation(
                content=str(e),
                error=True,
                url=self.url
            )

    @property
    def message(self) -> str:
        return f"Browsing URL: {self.url}"