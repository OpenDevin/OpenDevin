from __future__ import annotations
import asyncio
from dataclasses import dataclass
import random
from uuid import UUID
from oh.agent.agent_abc import AgentABC
from oh.conversation import conversation_abc
from oh.announcement.detail.text_reply import TextReply
from oh.command.command_status import CommandStatus


@dataclass
class MockAgent(AgentABC):
    """Mocks an agent - when prompted, it produces a number of TextReply events at delayed intervals"""

    min_number_replies: int = 1
    max_number_replies: int = 100
    min_delay: int = 1
    max_delay: int = 15
    cancellable: bool = True

    async def prompt(self, text: str, conversation: conversation_abc.ConversationABC):
        await conversation.trigger_event(
            TextReply(f'"{text}"??? That\'s a good one! Let me think about that!')
        )

        remaining_replies = random.randint(
            self.min_number_replies, self.max_number_replies
        )
        while remaining_replies:

            # Trigger a random reply
            await conversation.trigger_event(TextReply(random.choice(_REPLIES)))
            remaining_replies -= 1
            delay = random.randint(self.min_delay, self.max_delay)

            # Sleep for a random interval
            await asyncio.sleep(delay)

    async def stop():
        pass


_REPLIES = [
    "Whoever established the high road and how high it should be should be fired.",
    "Have you ever noticed that anybody driving slower than you is an idiot, and anyone going faster than you is a maniac?",
    "If I'm not back in five minutes, just wait longer.",
    "I like my money where I can see it: hanging in my closet.",
    "The suspense is terrible. I hope it'll last.",
    "Why do they call it rush hour when nothing moves?",
    "If you can't be kind, at least be vague.",
    "There is only one thing in the world worse than being talked about, and that is not being talked about.",
    "Always forgive your enemies; nothing annoys them so much.",
    "In real life, I assure you, there is no such thing as algebra.",
    "Instant gratification takes too long.",
    "Accept who you are. Unless you're a serial killer.",
    "Whoever said that money can't buy happiness, simply didn't know where to go shopping.",
    "So be wise, because the world needs more wisdom, and if you cannot be wise, pretend to be someone who is wise, and then just behave like they would.",
    "I'm not good at the advice. Can I interest you in a sarcastic comment?",
    "I'm sick of following my dreams, man. I'm just going to ask where they're going and hook up with 'em later.",
    "I'd love to stand here and talk with you...but I'm not going to.",
    "All you need is love. But a little chocolate now and then doesn't hurt.",
    "People say that money is not the key to happiness, but I always figured if you have enough money, you can have a key made.",
    "I'm not offended by blonde jokes because I know I'm not dumb…and I also know that I'm not blonde.",
    "It is useless to try to hold a person to anything he says while he's madly in love, drunk, or running for office.",
    "I remember it like it was yesterday. Of course, I don't really remember yesterday all that well.",
    "The trouble with having an open mind, of course, is that people will insist on coming along and trying to put things in it.",
    "To call you stupid would be an insult to stupid people! I've known sheep that could outwit you. I've worn dresses with higher IQs.",
    "Those people who think they know everything are a great annoyance to those of us who do.",
    "The reason I talk to myself is because I'm the only one whose answers I accept.",
    "I'm not superstitious…but I am a little stitious.",
    "Here's something to think about: How come you never see a headline like 'Psychic Wins Lottery'?",
]
