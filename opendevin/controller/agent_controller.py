import asyncio
import traceback
from typing import Optional, Tuple, Type

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State, TrafficControlState
from opendevin.controller.stuck import StuckDetector
from opendevin.core.config import LLMConfig
from opendevin.core.exceptions import (
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
    UserCancelledError,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import AgentState
from opendevin.events import EventSource, EventStream, EventStreamSubscriber
from opendevin.events.action import (
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
from opendevin.events.event import Event
from opendevin.events.observation import (
    AgentDelegateObservation,
    AgentStateChangedObservation,
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from opendevin.llm.llm import LLM

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
    agent_task: Optional[asyncio.Task] = None
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
            sid: The session ID of the agent.
            initial_state: The initial state of the controller.
            is_delegate: Whether this controller is a delegate.
            headless_mode: Whether the agent is run in headless mode.
        """
        self._is_closing = False
        self._close_event = asyncio.Event()
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

        # stuck helper
        self._stuck_detector = StuckDetector(self.state)

        if not is_delegate:
            self.agent_task = asyncio.create_task(self._start_step_loop())

    async def close(self):
        self._is_closing = True
        if self.agent_task is not None:
            self.agent_task.cancel()
        await self.set_agent_state_to(AgentState.STOPPED)
        self.event_stream.unsubscribe(EventStreamSubscriber.AGENT_CONTROLLER)
        self._close_event.set()

    def update_state_before_step(self):
        self.state.iteration += 1
        self.state.local_iteration += 1

    async def update_state_after_step(self):
        # update metrics especially for cost
        self.state.local_metrics = self.agent.llm.metrics

    async def report_error(self, message: str, exception: Exception | None = None):
        """This error will be reported to the user and sent to the LLM next step, in the hope it can self-correct.

        This method should be called for a particular type of errors, which have:
        - a user-friendly message, which will be shown in the chat box. This should not be a raw exception message.
        - an ErrorObservation that can be sent to the LLM by the agent, with the exception message, so it can self-correct next time.
        """
        self.state.last_error = message
        if exception:
            self.state.last_error += f': {exception}'
        await self.event_stream.add_event(ErrorObservation(message), EventSource.AGENT)

    async def _start_step_loop(self):
        logger.info(f'[Agent Controller `{self.id}`] Starting step loop...')
        try:
            while not self._is_closing:
                try:
                    await self._step()
                except asyncio.CancelledError:
                    logger.info(f'AgentController step was cancelled for `{self.id}`')
                    break
                except UserCancelledError:
                    logger.info('AgentController step was cancelled by user')
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f'Error while running the agent: {e}')
                    logger.error(traceback.format_exc())
                    await self.report_error(
                        'There was an unexpected error while running the agent',
                        exception=e,
                    )
                    await self.set_agent_state_to(AgentState.ERROR)
                    break

                await asyncio.sleep(0.1)
        finally:
            self._close_event.set()

    async def on_event(self, event: Event):
        # Only parse agent_state for ChangeAgentStateAction
        new_state = None
        if isinstance(event, ChangeAgentStateAction):
            success, parsed_state = self._parse_agent_state(event.agent_state)
            if success:
                new_state = parsed_state
            else:
                logger.error(f'Invalid agent state received: {event.agent_state}')
                return  # Exit early if the state is invalid

        # Cancelled is coming from the UI's Stop button
        if self.get_agent_state() == AgentState.CANCELLED:
            if isinstance(event, Observation) and not isinstance(
                event, AgentStateChangedObservation
            ):
                logger.info('Task cancelled, setting to finished.')
                await self.set_agent_state_to(AgentState.FINISHED)
                return
            elif isinstance(event, ChangeAgentStateAction):
                # Allow ChangeAgentStateAction to be processed even when stopped
                pass
            else:
                # Ignore all other events when cancelled
                logger.debug(f'Ignoring event in CANCELLED state: {event}')
                return

        if isinstance(event, ChangeAgentStateAction):
            if new_state is not None and new_state != self.state.agent_state:
                await self.set_agent_state_to(new_state)
        elif isinstance(event, MessageAction):
            if event.source == EventSource.USER:
                logger.info(
                    event,
                    extra={'msg_type': 'ACTION', 'event_source': EventSource.USER},
                )
                if self.get_agent_state() != AgentState.RUNNING:
                    await self.set_agent_state_to(AgentState.RUNNING)
            elif event.source == EventSource.AGENT and event.wait_for_response:
                await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
        elif isinstance(event, AgentDelegateAction):
            await self.start_delegate(event)
        elif isinstance(event, AddTaskAction):
            self.state.root_task.add_subtask(event.parent, event.goal, event.subtasks)
        elif isinstance(event, ModifyTaskAction):
            self.state.root_task.set_subtask_state(event.task_id, event.state)
        elif isinstance(event, AgentFinishAction):
            self.state.outputs = event.outputs  # type: ignore[attr-defined]
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.FINISHED)
        elif isinstance(event, AgentRejectAction):
            self.state.outputs = event.outputs  # type: ignore[attr-defined]
            self.state.metrics.merge(self.state.local_metrics)
            await self.set_agent_state_to(AgentState.REJECTED)
        elif isinstance(event, Observation):
            if (
                self._pending_action
                and hasattr(self._pending_action, 'is_confirmed')
                and self._pending_action.is_confirmed
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                return
            if self._pending_action and self._pending_action.id == event.cause:
                self._pending_action = None
                if self.state.agent_state == AgentState.USER_CONFIRMED:
                    await self.set_agent_state_to(AgentState.RUNNING)
                if self.state.agent_state == AgentState.USER_REJECTED:
                    await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
                logger.info(event, extra={'msg_type': 'OBSERVATION'})
            elif isinstance(event, CmdOutputObservation):
                logger.info(event, extra={'msg_type': 'OBSERVATION'})
            elif isinstance(event, AgentDelegateObservation):
                self.state.history.on_event(event)
                logger.info(event, extra={'msg_type': 'OBSERVATION'})
            elif isinstance(event, ErrorObservation):
                logger.info(event, extra={'msg_type': 'OBSERVATION'})

    def reset_task(self):
        self.almost_stuck = 0
        self.agent.reset()

    async def set_agent_state_to(self, new_state: AgentState):
        if new_state == self.state.agent_state:
            return

        logger.debug(
            f'[Agent Controller `{self.id}`] Setting agent({self.agent.name}) state from {self.state.agent_state} to {new_state}'
        )

        if (
            self.state.agent_state == AgentState.PAUSED
            and new_state == AgentState.RUNNING
            and self.state.traffic_control_state == TrafficControlState.THROTTLING
        ):
            # user intends to interrupt traffic control and let the task resume temporarily
            self.state.traffic_control_state = TrafficControlState.PAUSED

        self.state.agent_state = new_state
        if new_state in (AgentState.STOPPED, AgentState.ERROR):
            self.reset_task()

        if self._pending_action is not None and new_state in (
            AgentState.CANCELLED,
            AgentState.USER_CONFIRMED,
            AgentState.USER_REJECTED,
        ):
            if hasattr(self._pending_action, 'thought'):
                self._pending_action.thought = ''  # type: ignore[union-attr]
            if new_state == AgentState.CANCELLED:
                # await self.set_agent_state_to(AgentState.PAUSED)
                pass
            elif new_state == AgentState.USER_CONFIRMED:
                self._pending_action.is_confirmed = ActionConfirmationStatus.CONFIRMED  # type: ignore[attr-defined]
            else:
                self._pending_action.is_confirmed = ActionConfirmationStatus.REJECTED  # type: ignore[attr-defined]
            await self.event_stream.add_event(self._pending_action, EventSource.AGENT)

        await self.event_stream.add_event(
            AgentStateChangedObservation('', self.state.agent_state), EventSource.AGENT
        )

        if new_state == AgentState.INIT and self.state.resume_state:
            await self.set_agent_state_to(self.state.resume_state)
            self.state.resume_state = None

    def get_agent_state(self):
        """Returns the current state of the agent task."""
        return self.state.agent_state

    async def start_delegate(self, action: AgentDelegateAction):
        agent_cls: Type[Agent] = Agent.get_cls(action.agent)
        llm_config = self.agent_to_llm_config.get(action.agent, self.agent.llm.config)
        llm = LLM(config=llm_config)
        delegate_agent = agent_cls(llm=llm)
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
            f'[Agent Controller `{self.id}`]: start delegate, creating agent {delegate_agent.name} using {llm}'
        )
        self.delegate = AgentController(
            sid=self.id + '-delegate',
            agent=delegate_agent,
            event_stream=self.event_stream,
            max_iterations=self.state.max_iterations,
            max_budget_per_task=self.max_budget_per_task,
            agent_to_llm_config=self.agent_to_llm_config,
            initial_state=state,
            is_delegate=True,
        )
        await self.delegate.set_agent_state_to(AgentState.RUNNING)

    async def _step(self) -> None:
        if self.get_agent_state() != AgentState.RUNNING:
            await asyncio.sleep(1)
            return

        # Check for cancellation before proceeding
        if self.get_agent_state() == AgentState.CANCELLED:
            logger.info('Step cancelled due to agent state.')
            return

        if self._pending_action:
            logger.debug(
                f'[Agent Controller `{self.id}`] waiting for pending action: {self._pending_action}'
            )
            await asyncio.sleep(1)
            return

        if self.delegate is not None:
            logger.debug(
                f'[Agent Controller `{self.id}`] Delegate not none, awaiting...'
            )
            assert self.delegate != self
            await self.delegate._step()
            logger.debug(f'[Agent Controller `{self.id}`] Delegate step done')
            assert self.delegate is not None
            delegate_state = self.delegate.get_agent_state()
            logger.debug(
                f'[Agent Controller {self.id}] Delegate state: {delegate_state}'
            )
            if delegate_state == AgentState.ERROR:
                # close the delegate upon error
                await self.delegate.close()
                self.delegate = None
                await self.report_error('Delegator agent encountered an error')
                return

            delegate_done = delegate_state in (AgentState.FINISHED, AgentState.REJECTED)
            if delegate_done:
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
                await self.event_stream.add_event(obs, EventSource.AGENT)
            return

        logger.info(
            f'{self.agent.name} LEVEL {self.state.delegate_level} LOCAL STEP {self.state.local_iteration} GLOBAL STEP {self.state.iteration}',
            extra={'msg_type': 'STEP'},
        )

        if self.state.iteration >= self.state.max_iterations:
            if self.state.traffic_control_state == TrafficControlState.PAUSED:
                logger.info(
                    'Hitting traffic control, temporarily resume upon user request'
                )
                self.state.traffic_control_state = TrafficControlState.NORMAL
            else:
                self.state.traffic_control_state = TrafficControlState.THROTTLING
                if self.headless_mode:
                    # set to ERROR state if running in headless mode
                    # since user cannot resume on the web interface
                    await self.report_error(
                        'Agent reached maximum number of iterations in headless mode, task stopped.'
                    )
                    await self.set_agent_state_to(AgentState.ERROR)
                else:
                    await self.report_error(
                        f'Agent reached maximum number of iterations, task paused. {TRAFFIC_CONTROL_REMINDER}'
                    )
                    await self.set_agent_state_to(AgentState.PAUSED)
                return
        elif self.max_budget_per_task is not None:
            current_cost = self.state.metrics.accumulated_cost
            if current_cost > self.max_budget_per_task:
                if self.state.traffic_control_state == TrafficControlState.PAUSED:
                    logger.info(
                        'Hitting traffic control, temporarily resume upon user request'
                    )
                    self.state.traffic_control_state = TrafficControlState.NORMAL
                else:
                    self.state.traffic_control_state = TrafficControlState.THROTTLING
                    if self.headless_mode:
                        # set to ERROR state if running in headless mode
                        # there is no way to resume
                        await self.report_error(
                            f'Task budget exceeded. Current cost: {current_cost:.2f}, max budget: {self.max_budget_per_task:.2f}, task stopped.'
                        )
                        await self.set_agent_state_to(AgentState.ERROR)
                    else:
                        await self.report_error(
                            f'Task budget exceeded. Current cost: {current_cost:.2f}, Max budget: {self.max_budget_per_task:.2f}, task paused. {TRAFFIC_CONTROL_REMINDER}'
                        )
                        await self.set_agent_state_to(AgentState.PAUSED)
                    return

        self.update_state_before_step()
        action: Action = NullAction()
        try:
            action = self.agent.step(self.state)

            # Check for cancellation after getting the action
            if self.get_agent_state() == AgentState.CANCELLED:
                logger.info('Action cancelled due to agent state.')
                return

            if action is None:
                raise LLMNoActionError('No action was returned')

            if asyncio.iscoroutine(action):
                action = await action
            logger.info(action, extra={'msg_type': 'ACTION'})

        except UserCancelledError:
            logger.info('LLM request was cancelled.')
            await self.set_agent_state_to(AgentState.CANCELLED)
            return
        except (LLMMalformedActionError, LLMNoActionError, LLMResponseError) as e:
            # report to the user and send the underlying exception to the LLM for self-correction
            await self.report_error(f'{e}')
            return

        if hasattr(action, 'runnable') and action.runnable:
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
            await self.event_stream.add_event(action, EventSource.AGENT)

        await self.update_state_after_step()

        if self._is_stuck():
            await self.report_error('Agent got stuck in a loop')
            await self.set_agent_state_to(AgentState.ERROR)

    def get_state(self):
        return self.state

    def set_initial_state(
        self,
        state: State | None,
        max_iterations: int,
        confirmation_mode: bool = False,
    ):
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
            logger.debug(f'AgentController `{self.id}` restoring from event {start_id}')

        # make sure history is in sync
        self.state.start_id = start_id
        self.state.history.start_id = start_id

        # if there was an end_id saved in State, set it in history
        # currently not used, later useful for delegates
        if self.state.end_id > -1:
            self.state.history.end_id = self.state.end_id

    def _is_stuck(self):
        """Checks if the agent is stuck.

        Returns:
            bool: True if the agent is stuck, False otherwise.
        """
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

    async def stop(self) -> None:
        if self.state.agent_state not in (
            AgentState.FINISHED,
            # AgentState.REJECTED,
            AgentState.ERROR,
        ):
            await self.set_agent_state_to(AgentState.STOPPED)
        else:
            # If already in a terminal state, just ensure we're not running
            self._is_closing = True
            if self.agent_task:
                self.agent_task.cancel()

    def _parse_agent_state(self, state_str: str) -> Tuple[bool, AgentState | None]:
        """Parse a string into an AgentState enum value.

        Args:
            state_str (str): The string representation of the agent state.

        Returns:
            Tuple[bool, AgentState | None]: A tuple containing:
                - A boolean indicating whether the parsing was successful.
                - The AgentState enum value if successful, None otherwise.
        """
        try:
            new_state = AgentState(state_str)
            return True, new_state
        except ValueError:
            logger.error(f'Invalid agent state received: {state_str}')
            return False, None

    async def check_if_cancelled(self):
        """Check if the agent is cancelled. Serves as callback for the LLM.

        Returns:
            bool: True if the agent is stopped, False otherwise.
        """
        return self.get_agent_state() == AgentState.CANCELLED
