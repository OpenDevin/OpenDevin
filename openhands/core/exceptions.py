class AgentNoInstructionError(Exception):
    def __init__(self, message='Instruction must be provided'):
        super().__init__(message)


class AgentEventTypeError(Exception):
    def __init__(self, message='Event must be a dictionary'):
        super().__init__(message)


class AgentAlreadyRegisteredError(Exception):
    def __init__(self, name=None):
        if name is not None:
            message = f"Agent class already registered under '{name}'"
        else:
            message = 'Agent class already registered'
        super().__init__(message)


class AgentNotRegisteredError(Exception):
    def __init__(self, name=None):
        if name is not None:
            message = f"No agent class registered under '{name}'"
        else:
            message = 'No agent class registered'
        super().__init__(message)


class TaskInvalidStateError(Exception):
    def __init__(self, state=None):
        if state is not None:
            message = f'Invalid state {state}'
        else:
            message = 'Invalid state'
        super().__init__(message)


class BrowserInitException(Exception):
    def __init__(self, message='Failed to initialize browser environment'):
        super().__init__(message)


class BrowserUnavailableException(Exception):
    def __init__(
        self,
        message='Browser environment is not available, please check if has been initialized',
    ):
        super().__init__(message)


# This exception gets sent back to the LLM
# It might be malformed JSON
class LLMMalformedActionError(Exception):
    def __init__(self, message='Malformed response'):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


# This exception gets sent back to the LLM
# For some reason, the agent did not return an action
class LLMNoActionError(Exception):
    def __init__(self, message='Agent must return an action'):
        super().__init__(message)


# This exception gets sent back to the LLM
# The LLM output did not include an action, or the action was not the expected type
class LLMResponseError(Exception):
    def __init__(self, message='Failed to retrieve action from LLM response'):
        super().__init__(message)


class UserCancelledError(Exception):
    def __init__(self, message='User cancelled the request'):
        super().__init__(message)


class MicroAgentValidationError(Exception):
    def __init__(self, message='Micro agent validation failed'):
        super().__init__(message)


class OperationCancelled(Exception):
    """Exception raised when an operation is cancelled (e.g. by a keyboard interrupt)."""

    def __init__(self, message='Operation was cancelled'):
        super().__init__(message)
