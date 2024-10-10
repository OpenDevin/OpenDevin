import asyncio
import copy
import traceback
from typing import Type

from openhands.controller.agent import Agent
from openhands.controller.state.state import State, TrafficControlState
from openhands.controller.stuck import StuckDetector
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import (
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    AddTaskAction,
    AgentDelegateAction,
    AgentFinishAction,
    AgentRejectAction,
    ChangeAgentStateAction,
    CmdRunAction,
    IPythonRunCellAction,
    MessageAction,
    ModifyTaskAction,
    NullAction,
)
from openhands.events.event import Event
from openhands.events.observation import (
    AgentDelegateObservation,
    AgentStateChangedObservation,
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.runtime.utils.shutdown_listener import should_continue

# note: RESUME is only available on web GUI
TRAFFIC_CONTROL_REMINDER = (
    "Please click on resume button if you'd like to continue, or start a new task."
)


class AgentController:
    id: str
    agent: Agent
    max_iterations: int
    event_stream: EventStream
    state: State
    confirmation_mode: bool
    agent_to_llm_config: dict[str, LLMConfig]
    agent_configs: dict[str, AgentConfig]
    agent_task: asyncio.Future | None = None
    parent: 'AgentController | None' = None
    delegate: 'AgentController | None' = None
    _pending_action: Action | None = None

    def __init__(
        self,
        agent: Agent,
        event_stream: EventStream,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        sid: str = 'default',
        confirmation_mode: bool = False,
        initial_state: State | None = None,
        is_delegate: bool = False,
        headless_mode: bool = True,
    ):
        """Initializes a new instance of the AgentController class.

        Args:
            agent: The agent instance to control.
            event_stream: The event stream to publish events to.
            max_iterations: The maximum number of iterations the agent can run.
            max_budget_per_task: The maximum budget (in USD) allowed per task, beyond which the agent will stop.
            agent_to_llm_config: A dictionary mapping agent names to LLM configurations in the case that
                we delegate to a different agent.
            agent_configs: A dictionary mapping agent names to agent configurations in the case that
                we delegate to a different agent.
            sid: The session ID of the agent.
            initial_state: The initial state of the controller.
            is_delegate: Whether this controller is a delegate.
            headless_mode: Whether the agent is run in headless mode.
        """
        self._step_lock = asyncio.Lock()
        self.id = sid
        self.agent = agent
        self.headless_mode = headless_mode

        # subscribe to the event stream
        self.event_stream = event_stream
        self.event_stream.subscribe(
            EventStreamSubscriber.AGENT_CONTROLLER, self.on_event, append=is_delegate
        )

        # state from the previous session, state from a parent agent, or a fresh state
        self.set_initial_state(
            state=initial_state,
            max_iterations=max_iterations,
            confirmation_mode=confirmation_mode,
        )
        self.max_budget_per_task = max_budget_per_task
        self.agent_to_llm_config = agent_to_llm_config if agent_to_llm_config else {}
        self.agent_configs = agent_configs if agent_configs else {}
        self._initial_max_iterations = max_iterations
        self._initial_max_budget_per_task = max_budget_per_task

        # stuck helper
        self._stuck_detector = StuckDetector(self.state)

    async def close(self):
        """Closes the agent controller, canceling any ongoing tasks and unsubscribing from the event stream."""
        await self.set_agent_state_to(AgentState.STOPPED)
        self.event_stream.unsubscribe(EventStreamSubscriber.AGENT_CONTROLLER)

    def update_state_before_step(self):
        self.state.iteration += 1
        self.state.local_iteration += 1

    async def update_state_after_step(self):
        # update metrics especially for cost
        self.state.local_metrics = self.agent.llm.metrics
        if 'llm_completions' not in self.state.extra_data:
            self.state.extra_data['llm_completions'] = []
        self.state.extra_data['llm_completions'].extend(self.agent.llm.llm_completions)
        self.agent.llm.llm_completions.clear()

    async def report_error(self, message: str, exception: Exception | None = None):
        """Reports an error to the user and sends the exception to the LLM next step, in the hope it can self-correct.

        This method should be called for a particular type of errors, which have:
        - a user-friendly message, which will be shown in the chat box. This should not be a raw exception message.
        - an ErrorObservation that can be sent to the LLM by the agent, with the exception message, so it can self-correct next time.
        """
        self.state.last_error = message
        if exception:
            self.state.last_error += f': {exception}'
        self.event_stream.add_event(ErrorObservation(message), EventSource.AGENT)

    async def start_step_loop(self):
        """The main loop for the agent's step-by-step execution."""

        logger.info(f'[Agent Controller {self.id}] Starting step loop...')
        while should_continue():
            try:
                await self._step()
            except asyncio.CancelledError:
                logger.info('AgentController task was cancelled')
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f'Error while running the agent: {e}')
                logger.error(traceback.format_exc())
                await self.report_error(
                    'There was an unexpected error while running the agent', exception=e
                )
                await self.set_agent_state_to(AgentState.ERROR)
                break

            await asyncio.sleep(0.1)

    async def on_event(self, event: Event):
        """Callback from the event stream. Notifies the controller of incoming events.

        Args:
            event (Event): The incoming event to process.
        """
        if hasattr(event, 'hidden') and event.hidden:
            return
        if isinstance(event, Action):
            await self._handle_action(event)
        elif isinstance(event, Observation):
            await self._handle_observation(event)

    async def _handle_action(self, action: Action):
        """Handles actions from the event stream.

        Args:
            action (Action): The action to handle.
        """
        if isinstance(action, ChangeAgentStateAction):
            await self.set_agent_state_to(action.agent_state)  # type: ignore
        elif isinstance(action, MessageAction):
            await self._handle_message_action(action)
        elif isinstance(action, AgentDelegateAction):
            await self.start_delegate(action)
        elif isinstance(action, AddTaskAction):
            self.state.root_task.add_subtask(
                action.parent, action.goal, action.subtasks
            )
        elif isinstance(action, ModifyTaskAction):
            self.state.root_task.set_subtask_state(action.task_id, action.state)
        elif isinstance(action, AgentFinishAction):
            self.state.outputs = action.outputs
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.FINISHED)
        elif isinstance(action, AgentRejectAction):
            self.state.outputs = action.outputs
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.REJECTED)

    async def _handle_observation(self, observation: Observation):
        """Handles observation from the event stream.

        Args:
            observation (observation): The observation to handle.
        """
        if (
            self._pending_action
            and hasattr(self._pending_action, 'is_confirmed')
            and self._pending_action.is_confirmed
            == ActionConfirmationStatus.AWAITING_CONFIRMATION
        ):
            return

        # Make sure we print the observation in the same way as the LLM sees it
        observation_to_print = copy.deepcopy(observation)
        if len(observation_to_print.content) > self.agent.llm.config.max_message_chars:
            observation_to_print.content = truncate_content(
                observation_to_print.content, self.agent.llm.config.max_message_chars
            )
        logger.info(observation_to_print, extra={'msg_type': 'OBSERVATION'})
        if self._pending_action and self._pending_action.id == observation.cause:
            self._pending_action = None
            if self.state.agent_state == AgentState.USER_CONFIRMED:
                await self.set_agent_state_to(AgentState.RUNNING)
            if self.state.agent_state == AgentState.USER_REJECTED:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return

        if isinstance(observation, CmdOutputObservation):
            return
        elif isinstance(observation, AgentDelegateObservation):
            self.state.history.on_event(observation)
        elif isinstance(observation, ErrorObservation):
            if self.state.agent_state == AgentState.ERROR:
                self.state.metrics.merge(self.state.local_metrics)

    async def _handle_message_action(self, action: MessageAction):
        """Handles message actions from the event stream.

        Args:
            action (MessageAction): The message action to handle.
        """
        if action.source == EventSource.USER:
            logger.info(
                action, extra={'msg_type': 'ACTION', 'event_source': EventSource.USER}
            )
            if self.get_agent_state() != AgentState.RUNNING:
                await self.set_agent_state_to(AgentState.RUNNING)
        elif action.source == EventSource.AGENT and action.wait_for_response:
            await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

    def reset_task(self):
        """Resets the agent's task."""

        self.almost_stuck = 0
        self.agent.reset()

    async def set_agent_state_to(self, new_state: AgentState):
        """Updates the agent's state and handles side effects. Can emit events to the event stream.

        Args:
            new_state (AgentState): The new state to set for the agent.
        """
        logger.debug(
            f'[Agent Controller {self.id}] Setting agent({self.agent.name}) state from {self.state.agent_state} to {new_state}'
        )

        if new_state == self.state.agent_state:
            return

        if new_state == AgentState.STOPPED or new_state == AgentState.ERROR:
            self.reset_task()
        elif (
            new_state == AgentState.RUNNING
            and self.state.agent_state == AgentState.PAUSED
            and self.state.traffic_control_state == TrafficControlState.THROTTLING
        ):
            # user intends to interrupt traffic control and let the task resume temporarily
            self.state.traffic_control_state = TrafficControlState.PAUSED
            # User has chosen to deliberately continue - lets double the max iterations
            if (
                self.state.iteration is not None
                and self.state.max_iterations is not None
                and self._initial_max_iterations is not None
            ):
                if self.state.iteration >= self.state.max_iterations:
                    self.state.max_iterations += self._initial_max_iterations

            if (
                self.state.metrics.accumulated_cost is not None
                and self.max_budget_per_task is not None
                and self._initial_max_budget_per_task is not None
            ):
                if self.state.metrics.accumulated_cost >= self.max_budget_per_task:
                    self.max_budget_per_task += self._initial_max_budget_per_task
        elif self._pending_action is not None and (
            new_state == AgentState.USER_CONFIRMED
            or new_state == AgentState.USER_REJECTED
        ):
            if hasattr(self._pending_action, 'thought'):
                self._pending_action.thought = ''  # type: ignore[union-attr]
            if new_state == AgentState.USER_CONFIRMED:
                self._pending_action.is_confirmed = ActionConfirmationStatus.CONFIRMED  # type: ignore[attr-defined]
            else:
                self._pending_action.is_confirmed = ActionConfirmationStatus.REJECTED  # type: ignore[attr-defined]
            self.event_stream.add_event(self._pending_action, EventSource.AGENT)

        self.state.agent_state = new_state
        self.event_stream.add_event(
            AgentStateChangedObservation('', self.state.agent_state), EventSource.AGENT
        )

        if new_state == AgentState.INIT and self.state.resume_state:
            await self.set_agent_state_to(self.state.resume_state)
            self.state.resume_state = None

    def get_agent_state(self):
        """Returns the current state of the agent.

        Returns:
            AgentState: The current state of the agent.
        """
        return self.state.agent_state

    async def start_delegate(self, action: AgentDelegateAction):
        """Start a delegate agent to handle a subtask.

        OpenHands is a multi-agentic system. A `task` is a conversation between
        OpenHands (the whole system) and the user, which might involve one or more inputs
        from the user. It starts with an initial input (typically a task statement) from
        the user, and ends with either an `AgentFinishAction` initiated by the agent, a
        stop initiated by the user, or an error.

        A `subtask` is a conversation between an agent and the user, or another agent. If a `task`
        is conducted by a single agent, then it's also a `subtask`. Otherwise, a `task` consists of
        multiple `subtasks`, each executed by one agent.

        Args:
            action (AgentDelegateAction): The action containing information about the delegate agent to start.
        """
        agent_cls: Type[Agent] = Agent.get_cls(action.agent)
        agent_config = self.agent_configs.get(action.agent, self.agent.config)
        llm_config = self.agent_to_llm_config.get(action.agent, self.agent.llm.config)
        llm = LLM(config=llm_config)
        delegate_agent = agent_cls(llm=llm, config=agent_config)
        state = State(
            inputs=action.inputs or {},
            local_iteration=0,
            iteration=self.state.iteration,
            max_iterations=self.state.max_iterations,
            delegate_level=self.state.delegate_level + 1,
            # global metrics should be shared between parent and child
            metrics=self.state.metrics,
        )
        logger.info(
            f'[Agent Controller {self.id}]: start delegate, creating agent {delegate_agent.name} using LLM {llm}'
        )
        self.delegate = AgentController(
            sid=self.id + '-delegate',
            agent=delegate_agent,
            event_stream=self.event_stream,
            max_iterations=self.state.max_iterations,
            max_budget_per_task=self.max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            agent_configs=self.agent_configs,
            initial_state=state,
            is_delegate=True,
            headless_mode=self.headless_mode,
        )
        await self.delegate.set_agent_state_to(AgentState.RUNNING)

    async def _step(self) -> None:
        """Executes a single step of the parent or delegate agent. Detects stuck agents and limits on the number of iterations and the task budget."""
        if self.get_agent_state() != AgentState.RUNNING:
            await asyncio.sleep(1)
            return

        if self._pending_action:
            await asyncio.sleep(1)
            return

        if self.delegate is not None:
            assert self.delegate != self
            if self.delegate.get_agent_state() == AgentState.PAUSED:
                await asyncio.sleep(1)
            else:
                await self._delegate_step()
            return

        logger.info(
            f'{self.agent.name} LEVEL {self.state.delegate_level} LOCAL STEP {self.state.local_iteration} GLOBAL STEP {self.state.iteration}',
            extra={'msg_type': 'STEP'},
        )

        # check if agent hit the resources limit
        stop_step = False
        if self.state.iteration >= self.state.max_iterations:
            stop_step = await self._handle_traffic_control(
                'iteration', self.state.iteration, self.state.max_iterations
            )
        if self.max_budget_per_task is not None:
            current_cost = self.state.metrics.accumulated_cost
            if current_cost > self.max_budget_per_task:
                stop_step = await self._handle_traffic_control(
                    'budget', current_cost, self.max_budget_per_task
                )
        if stop_step:
            return

        self.update_state_before_step()
        action: Action = NullAction()
        try:
            action = self.agent.step(self.state)
            if action is None:
                raise LLMNoActionError('No action was returned')
        except (LLMMalformedActionError, LLMNoActionError, LLMResponseError) as e:
            # report to the user
            # and send the underlying exception to the LLM for self-correction
            await self.report_error(str(e))
            return

        if action.runnable:
            if self.state.confirmation_mode and (
                type(action) is CmdRunAction or type(action) is IPythonRunCellAction
            ):
                action.is_confirmed = ActionConfirmationStatus.AWAITING_CONFIRMATION
            self._pending_action = action

        if not isinstance(action, NullAction):
            if (
                hasattr(action, 'is_confirmed')
                and action.is_confirmed
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                await self.set_agent_state_to(AgentState.AWAITING_USER_CONFIRMATION)
            self.event_stream.add_event(action, EventSource.AGENT)

        await self.update_state_after_step()
        logger.info(action, extra={'msg_type': 'ACTION'})

        if self._is_stuck():
            await self.report_error('Agent got stuck in a loop')
            await self.set_agent_state_to(AgentState.ERROR)

    async def _delegate_step(self):
        """Executes a single step of the delegate agent."""
        logger.debug(f'[Agent Controller {self.id}] Delegate not none, awaiting...')
        await self.delegate._step()  # type: ignore[union-attr]
        logger.debug(f'[Agent Controller {self.id}] Delegate step done')
        assert self.delegate is not None
        delegate_state = self.delegate.get_agent_state()
        logger.debug(f'[Agent Controller {self.id}] Delegate state: {delegate_state}')
        if delegate_state == AgentState.ERROR:
            # update iteration that shall be shared across agents
            self.state.iteration = self.delegate.state.iteration

            # close the delegate upon error
            await self.delegate.close()
            self.delegate = None
            self.delegateAction = None

            await self.report_error('Delegator agent encountered an error')
        elif delegate_state in (AgentState.FINISHED, AgentState.REJECTED):
            logger.info(
                f'[Agent Controller {self.id}] Delegate agent has finished execution'
            )
            # retrieve delegate result
            outputs = self.delegate.state.outputs if self.delegate.state else {}

            # update iteration that shall be shared across agents
            self.state.iteration = self.delegate.state.iteration

            # close delegate controller: we must close the delegate controller before adding new events
            await self.delegate.close()

            # update delegate result observation
            # TODO: replace this with AI-generated summary (#2395)
            formatted_output = ', '.join(
                f'{key}: {value}' for key, value in outputs.items()
            )
            content = (
                f'{self.delegate.agent.name} finishes task with {formatted_output}'
            )
            obs: Observation = AgentDelegateObservation(
                outputs=outputs, content=content
            )

            # clean up delegate status
            self.delegate = None
            self.delegateAction = None
            self.event_stream.add_event(obs, EventSource.AGENT)
        return

    async def _handle_traffic_control(
        self, limit_type: str, current_value: float, max_value: float
    ):
        """Handles agent state after hitting the traffic control limit.

        Args:
            limit_type (str): The type of limit that was hit.
            current_value (float): The current value of the limit.
            max_value (float): The maximum value of the limit.
        """
        stop_step = False
        if self.state.traffic_control_state == TrafficControlState.PAUSED:
            logger.info('Hitting traffic control, temporarily resume upon user request')
            self.state.traffic_control_state = TrafficControlState.NORMAL
        else:
            self.state.traffic_control_state = TrafficControlState.THROTTLING
            if self.headless_mode:
                # set to ERROR state if running in headless mode
                # since user cannot resume on the web interface
                await self.report_error(
                    f'Agent reached maximum {limit_type} in headless mode, task stopped. '
                    f'Current {limit_type}: {current_value:.2f}, max {limit_type}: {max_value:.2f}'
                )
                await self.set_agent_state_to(AgentState.ERROR)
            else:
                await self.report_error(
                    f'Agent reached maximum {limit_type}, task paused. '
                    f'Current {limit_type}: {current_value:.2f}, max {limit_type}: {max_value:.2f}. '
                    f'{TRAFFIC_CONTROL_REMINDER}'
                )
                await self.set_agent_state_to(AgentState.PAUSED)
            stop_step = True
        return stop_step

    def get_state(self):
        """Returns the current running state object.

        Returns:
            State: The current state object.
        """
        return self.state

    def set_initial_state(
        self,
        state: State | None,
        max_iterations: int,
        confirmation_mode: bool = False,
    ):
        """Sets the initial state for the agent, either from the previous session, or from a parent agent, or by creating a new one.

        Args:
            state: The state to initialize with, or None to create a new state.
            max_iterations: The maximum number of iterations allowed for the task.
            confirmation_mode: Whether to enable confirmation mode.
        """
        # state from the previous session, state from a parent agent, or a new state
        # note that this is called twice when restoring a previous session, first with state=None
        if state is None:
            self.state = State(
                inputs={},
                max_iterations=max_iterations,
                confirmation_mode=confirmation_mode,
            )
        else:
            self.state = state

        # when restored from a previous session, the State object will have history, start_id, and end_id
        # connect it to the event stream
        self.state.history.set_event_stream(self.event_stream)

        # if start_id was not set in State, we're starting fresh, at the top of the stream
        start_id = self.state.start_id
        if start_id == -1:
            start_id = self.event_stream.get_latest_event_id() + 1
        else:
            logger.debug(f'AgentController {self.id} restoring from event {start_id}')

        # make sure history is in sync
        self.state.start_id = start_id
        self.state.history.start_id = start_id

        # if there was an end_id saved in State, set it in history
        # currently not used, later useful for delegates
        if self.state.end_id > -1:
            self.state.history.end_id = self.state.end_id

    def _is_stuck(self):
        """Checks if the agent or its delegate is stuck in a loop.

        Returns:
            bool: True if the agent is stuck, False otherwise.
        """
        # check if delegate stuck
        if self.delegate and self.delegate._is_stuck():
            return True

        return self._stuck_detector.is_stuck()

    def __repr__(self):
        return (
            f'AgentController(id={self.id}, agent={self.agent!r}, '
            f'event_stream={self.event_stream!r}, '
            f'state={self.state!r}, agent_task={self.agent_task!r}, '
            f'delegate={self.delegate!r}, _pending_action={self._pending_action!r})'
        )
