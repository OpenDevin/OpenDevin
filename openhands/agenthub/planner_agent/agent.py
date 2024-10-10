from openhands.agenthub.planner_agent.prompt import HISTORY_SIZE, get_prompt
from openhands.agenthub.planner_agent.response_parser import PlannerResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.message import ImageContent, Message, TextContent
from openhands.core.utils import json
from openhands.events.action import Action, AgentFinishAction
from openhands.events.serialization.event import event_to_memory
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory


class PlannerAgent(Agent):
    VERSION = '1.0'
    """
    The planner agent utilizes a special prompting strategy to create long term plans for solving problems.
    The agent is given its previous action-observation pairs, current task, and hint based on last action taken at every step.
    """
    response_parser = PlannerResponseParser()

    def __init__(self, llm: LLM, config: AgentConfig, memory: ConversationMemory):
        """Initialize the Planner Agent with an LLM

        Parameters:
        - llm: The llm to be used by this agent
        - config: The agent config
        - memory: The conversation memory
        """
        super().__init__(llm, config, memory)

    def step(self, state: State) -> Action:
        """Checks to see if current step is completed, returns AgentFinishAction if True.
        Otherwise, creates a plan prompt and sends to model for inference, returning the result as the next action.

        Parameters:
        - state (State): The current state given the previous actions and observations

        Returns:
        - AgentFinishAction: If the last state was 'completed', 'verified', or 'abandoned'
        - Action: The next action to take based on llm response
        """
        if state.root_task.state in [
            'completed',
            'verified',
            'abandoned',
        ]:
            return AgentFinishAction()

        # the goal (user-defined task)
        task, image_urls = self.memory.get_current_user_intent()

        # we will need the last action for the hint
        last_action = self.memory.get_last_action()

        # get history as a string to insert into the prompt
        history_str = self._get_history_str()

        # format the prompt
        prompt = get_prompt(
            state, task, last_action, history_str, self.llm.config.max_message_chars
        )

        # create the message
        content = [TextContent(text=prompt)]
        if self.llm.vision_is_active() and image_urls:
            content.append(ImageContent(image_urls=image_urls))
        message = Message(role='user', content=content)

        # send to the LLM and return the action
        resp = self.llm.completion(messages=self.llm.format_messages_for_llm(message))
        return self.response_parser.parse(resp)

    def _get_history_str(self) -> str:
        """
        Get the history string from the event stream.
        """

        # the history
        history_dicts = []

        # retrieve the latest HISTORY_SIZE events
        for event_count, event in enumerate(self.memory.get_events(reverse=True)):
            if event_count >= HISTORY_SIZE:
                break
            history_dicts.append(
                event_to_memory(event, self.llm.config.max_message_chars)
            )

        # history_dicts is in reverse order, lets fix it
        history_dicts.reverse()

        # and get it as a JSON string
        history_str = json.dumps(history_dicts, indent=2)

        return history_str
