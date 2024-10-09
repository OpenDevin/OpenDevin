from dataclasses import dataclass

from oh.conversation.conversation_abc import ConversationABC
from oh.storage.item_filter_abc import ItemFilterABC


@dataclass
class ConversationFilter(ItemFilterABC[ConversationABC]):
    # This is mostly future proofing for now

    def filter(self, item: ConversationABC) -> bool:
        return True
