from dataclasses import dataclass

from opendevin.core.schema import ActionType

from .action import Action, ActionSecurityRisk

@dataclass
class MessageAction(Action):
    content: str
    images_urls: list | None = None
    wait_for_response: bool = False
    action: str = ActionType.MESSAGE
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        ret = f'**MessageAction** (source={self.source})\n'
        ret += f'CONTENT: {self.content}'
        if self.images_urls:
            for url in self.images_urls:
                ret += f'\nIMAGE_URL: {url}'
        return ret
