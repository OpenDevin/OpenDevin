from typing import Callable

from opendevin.core.logger import opendevin_logger as logger
from opendevin.llm.llm import LLM

from . import get_summarize_prompt, parse_summary_response

MAX_TOKEN_COUNT_PADDING = (
    512  # estimation of tokens to add to the prompt for the max token count
)


class MemoryCondenser:
    """
    Condenses the prompt with a call to the LLM.
    """

    def __init__(
        self,
        action_prompt: Callable[..., str],
        summarize_prompt: Callable[[list[dict]], str] = get_summarize_prompt,
    ):
        """
        Initialize the MemoryCondenser with the action and summarize prompts.

        action_prompt is a callable that returns the prompt that is about to be sent to the LLM.
        The prompt callable will be called with recent events as arguments.
        summarize_prompt is a callable that returns a specific prompt that tells the LLM to summarize the recent events.
        The prompt callable will be called with events as argument.
        If not provided, the default summarize prompt will be used.

        Parameters:
        - action_prompt: The function to generate an action prompt. The function should accept recent events as arguments.
        - summarize_prompt: The function to generate a summarize prompt. The function should accept srecent events as arguments.
        """
        self.action_prompt = action_prompt
        self.summarize_prompt = summarize_prompt

    def condense(
        self,
        llm: LLM,
        recent_events: list[dict],
    ) -> tuple[list[dict], bool]:
        """
        Attempts to condense the recent events of the monologue by using the llm. Returns the condensed recent events if successful, or False if not.

        It includes default events in the prompt for context, but does not alter them.
        Condenses the events using a summary prompt.

        Parameters:
        - llm: LLM to be used for summarization.
        - recent_events: List of recent events that may be condensed.

        Returns:
        - The condensed recent events if successful, or False if condensation failed.
        """
        try:
            logger.debug('Condensing recent events')

            # attempt to condense the recent events
            new_recent_events = self._attempt_condense(llm, recent_events)

            if not new_recent_events or len(new_recent_events) == 0:
                logger.debug('Condensation failed: new_recent_events is empty')
                return [], False

            # re-generate the action prompt with the condensed events
            new_action_prompt = self.action_prompt(new_recent_events)

            # hope for the best
            return new_recent_events, True

        except Exception as e:
            logger.error('Condensation failed: %s', str(e), exc_info=False)
            return [], False

    def _attempt_condense(
        self,
        llm: LLM,
        recent_events: list[dict],
    ) -> list[dict] | None:
        """
        Attempts to condense the recent events by splitting them in half and summarizing the first half.

        Parameters:
        - llm: The llm to use for summarization.
        - default_events: The list of default events to include in the prompt.
        - recent_events: The list of recent events to include in the prompt.

        Returns:
        - The condensed recent events if successful, None otherwise.
        """

        # Split events
        midpoint = len(recent_events) // 2
        first_half = recent_events[:midpoint].copy()
        second_half = recent_events[midpoint:].copy()

        # attempt to condense the first half of the recent events
        summarize_prompt = self.summarize_prompt(first_half)

        # send the summarize prompt to the LLM
        messages = [{'content': summarize_prompt, 'role': 'user'}]
        response = llm.completion(messages=messages)
        response_content = response['choices'][0]['message']['content']

        # the new list of recent events will be source events or summarize actions
        condensed_events = parse_summary_response(response_content)

        # new recent events list
        if (
            not condensed_events
            or not isinstance(condensed_events, list)
            or len(condensed_events) == 0
        ):
            return None

        condensed_events.extend(second_half)
        return condensed_events

    def _needs_condense(self, **kwargs):
        """
        Checks if the prompt needs to be condensed based on the token count against the limits of the llm passed in the call.

        Parameters:
        - llm: The llm to use for checking the token count.
        - action_prompt: The prompt to check for token count. If not provided, it will attempt to generate it using the available arguments.
        - default_events: The list of default events to include in the prompt.
        - recent_events: The list of recent events to include in the prompt.

        Returns:
        - True if the prompt needs to be condensed, False otherwise.
        """
        llm = kwargs.get('llm')
        action_prompt = kwargs.get('action_prompt')

        if not llm:
            logger.warning("Missing argument 'llm', cannot check token count.")
            return False

        if not action_prompt:
            # Attempt to generate the action_prompt using the available arguments
            recent_events = kwargs.get('recent_events', [])

            action_prompt = self.action_prompt(recent_events)

        token_count = llm.get_token_count([{'content': action_prompt, 'role': 'user'}])
        return token_count >= self.get_token_limit(llm)

    def get_token_limit(self, llm: LLM) -> int:
        """
        Returns the token limit to use for the llm passed in the call.

        Parameters:
        - llm: The llm to get the token limit from.

        Returns:
        - The token limit of the llm.
        """
        return llm.max_input_tokens - MAX_TOKEN_COUNT_PADDING
