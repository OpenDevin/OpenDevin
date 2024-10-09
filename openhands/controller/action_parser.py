from abc import ABC, abstractmethod

from openhands.events.action import Action


class ActionParseError(Exception):
    """Exception raised when the response from the LLM cannot be parsed into an action."""

    def __init__(self, error: str):
        self.error = error

    def __str__(self):
        return self.error


class ResponseParser(ABC):
    """This abstract base class is a general interface for an response parser dedicated to
    parsing the action from the response from the LLM.
    """

    def __init__(
        self,
    ):
        # Need pay attention to the item order in self.action_parsers
        self.action_parsers = []

    @abstractmethod
    def parse(self, response: str) -> Action:
        """Parses the action from the response from the LLM.

        Parameters:
        - response (str): The response from the LLM.

        Returns:
        - action (Action): The action parsed from the response.
        """
        pass

    @abstractmethod
    def parse_response(self, response) -> str:
        """Parses the action from the response from the LLM.

        Parameters:
        - response (str): The response from the LLM.

        Returns:
        - action_str (str): The action str parsed from the response.
        """
        pass

    @abstractmethod
    def parse_action(self, action_str: str) -> Action:
        """Parses the action from the response from the LLM.

        Parameters:
        - action_str (str): The response from the LLM.

        Returns:
        - action (Action): The action parsed from the response.
        """
        pass


class ActionParser(ABC):
    """This abstract base class is a general interface for an action parser dedicated to
    parsing the action from the action str from the LLM.
    """

    @abstractmethod
    def check_condition(self, action_str: str) -> bool:
        """Check if the action string can be parsed by this parser."""
        pass

    @abstractmethod
    def parse(self, action_str: str) -> Action:
        """Parses the action from the action string from the LLM response."""
        pass
