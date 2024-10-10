from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.action.agent import (
    AgentDelegateAction,
    AgentFinishAction,
)
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.observation import Observation
from openhands.events.stream import EventStream, EventStreamSubscriber


class ConversationMemory:
    """A list of events in the immediate memory of the agent.

    This class provides methods to retrieve and filter the events in the history of the running agent from the event stream.
    """

    _event_stream: EventStream
    delegates: dict[tuple[int, int], tuple[str, str]]

    def __init__(self, event_stream: EventStream):
        self._event_stream = event_stream
        self._event_stream.subscribe(
            EventStreamSubscriber.MEMORY, self.on_event, append=True
        )
        self.delegates = {}
        self.start_id = self._event_stream.get_latest_event_id()

    def get_events(
        self, reverse: bool = False, include_delegates: bool = False
    ) -> list[Event]:
        """Retrieve and return events for agent's use as a list of Event objects."""

        return list(
            self._event_stream.get_events(
                reverse=reverse, include_delegates=include_delegates
            )
        )

    def get_last_events(self, n: int) -> list[Event]:
        """Return the last n events from the event stream."""
        # dummy agent is using this
        # it should work, but it's not great to store temporary lists now just for a test
        end_id = self._event_stream.get_latest_event_id()

        # FIXME this ignores that there are events that won't be returned, like NullObservations
        start_id = max(self.start_id, end_id - n + 1)

        return list(
            event
            for event in self._event_stream.get_events(start_id=start_id, end_id=end_id)
        )

    async def on_event(self, event: Event):
        if not isinstance(event, AgentDelegateObservation):
            return

        logger.debug('AgentDelegateObservation received')

        # figure out what this delegate's actions were
        # from the last AgentDelegateAction to this AgentDelegateObservation
        # and save their ids as start and end ids
        # in order to use later to exclude them from parent stream
        # or summarize them
        delegate_end = event.id
        delegate_start = -1
        delegate_agent: str = ''
        delegate_task: str = ''
        for prev_event in self._event_stream.get_events(
            end_id=event.id - 1, reverse=True
        ):
            if isinstance(prev_event, AgentDelegateAction):
                delegate_start = prev_event.id
                delegate_agent = prev_event.agent
                delegate_task = prev_event.inputs.get('task', '')
                break

        if delegate_start == -1:
            logger.error(
                f'No AgentDelegateAction found for AgentDelegateObservation with id={delegate_end}'
            )
            return

        self.delegates[(delegate_start, delegate_end)] = (delegate_agent, delegate_task)
        logger.debug(
            f'Delegate {delegate_agent} with task {delegate_task} ran from id={delegate_start} to id={delegate_end}'
        )

    def reset(self):
        self.delegates = {}

        # wipe history of previous interactions
        # alternatively, we can re-initialize a new event stream, then we need to notify everyone who is subscribed to this event stream
        self.start_id = self._event_stream.get_latest_event_id()

    def get_current_user_intent(self):
        """Returns the latest user message and image(if provided) that appears after a FinishAction, or the first (the task) if nothing was finished yet."""
        last_user_message = None
        last_user_message_image_urls: list[str] | None = []
        for event in self._event_stream.get_events(reverse=True):
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                last_user_message = event.content
                last_user_message_image_urls = event.images_urls
            elif isinstance(event, AgentFinishAction):
                if last_user_message is not None:
                    return last_user_message

        return last_user_message, last_user_message_image_urls

    def get_last_action(self, end_id: int = -1) -> Action | None:
        """Return the last action from the event stream, filtered to exclude unwanted events."""

        end_id = (
            end_id if end_id != -1 else self._event_stream.get_latest_event_id() - 1
        )

        last_action = next(
            (
                event
                for event in self._event_stream.get_events(
                    start_id=self.start_id, end_id=end_id, reverse=True
                )
                if isinstance(event, Action)
            ),
            None,
        )

        return last_action

    def get_last_observation(self, end_id: int = -1) -> Observation | None:
        """Return the last observation from the event stream, filtered to exclude unwanted events."""

        end_id = (
            end_id if end_id != -1 else self._event_stream.get_latest_event_id() - 1
        )

        last_observation = next(
            (
                event
                for event in self._event_stream.get_events(
                    start_id=self.start_id, end_id=end_id, reverse=True
                )
                if isinstance(event, Observation)
            ),
            None,
        )

        return last_observation

    def get_last_user_message(self) -> str:
        """Return the content of the last user message from the event stream."""
        last_user_message = next(
            (
                event.content
                for event in self._event_stream.get_events(
                    start_id=self.start_id, reverse=True
                )
                if isinstance(event, MessageAction) and event.source == EventSource.USER
            ),
            None,
        )

        return last_user_message if last_user_message is not None else ''

    def get_last_agent_message(self) -> str:
        """Return the content of the last agent message from the event stream."""
        last_agent_message = next(
            (
                event.content
                for event in self._event_stream.get_events(
                    start_id=self.start_id, reverse=True
                )
                if isinstance(event, MessageAction)
                and event.source == EventSource.AGENT
            ),
            None,
        )

        return last_agent_message if last_agent_message is not None else ''
