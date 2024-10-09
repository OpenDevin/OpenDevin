from pydantic import BaseModel, Field

__all__ = ['ActionType']


class ActionTypeSchema(BaseModel):
    INIT: str = Field(default='initialize')
    """Initializes the agent. Only sent by client.
    """

    MESSAGE: str = Field(default='message')
    """Represents a message.
    """

    START: str = Field(default='start')
    """Starts a new development task OR send chat from the user. Only sent by the client.
    """

    READ: str = Field(default='read')
    """Reads the content of a file.
    """

    WRITE: str = Field(default='write')
    """Writes the content to a file.
    """

    EDIT: str = Field(default='edit')
    """Edits a file by providing a draft.
    """

    RUN: str = Field(default='run')
    """Runs a command.
    """

    RUN_IPYTHON: str = Field(default='run_ipython')
    """Runs a IPython cell.
    """

    BROWSE: str = Field(default='browse')
    """Opens a web page.
    """

    BROWSE_INTERACTIVE: str = Field(default='browse_interactive')
    """Interact with the browser instance.
    """

    DELEGATE: str = Field(default='delegate')
    """Delegates a task to another agent.
    """

    FINISH: str = Field(default='finish')
    """If you're absolutely certain that you've completed your task and have tested your work,
    use the finish action to stop working.
    """

    REJECT: str = Field(default='reject')
    """If you're absolutely certain that you cannot complete the task with given requirements,
    use the reject action to stop working.
    """

    NULL: str = Field(default='null')

    SUMMARIZE: str = Field(default='summarize')

    ADD_TASK: str = Field(default='add_task')

    MODIFY_TASK: str = Field(default='modify_task')

    PAUSE: str = Field(default='pause')
    """Pauses the task.
    """

    RESUME: str = Field(default='resume')
    """Resumes the task.
    """

    STOP: str = Field(default='stop')
    """Stops the task. Must send a start action to restart a new task.
    """

    CHANGE_AGENT_STATE: str = Field(default='change_agent_state')

    PUSH: str = Field(default='push')
    """Push a branch to github."""

    SEND_PR: str = Field(default='send_pr')
    """Send a PR to github."""


ActionType = ActionTypeSchema()
