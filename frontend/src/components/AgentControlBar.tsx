import { Tooltip } from "@nextui-org/react";
import React, { useEffect } from "react";
import { useSelector } from "react-redux";
import ArrowIcon from "#/assets/arrow";
import PauseIcon from "#/assets/pause";
import PlayIcon from "#/assets/play";
import StopIcon from "#/assets/stop";
import { changeAgentState } from "#/services/agentStateService";
import store, { RootState } from "#/store";
import AgentState from "#/types/AgentState";
import { clearMessages } from "#/state/chatSlice";
import Session from "#/services/session";

const IgnoreTaskStateMap: { [k: string]: AgentState[] } = {
  [AgentState.PAUSED]: [
    AgentState.INIT,
    AgentState.PAUSED,
    AgentState.STOPPED,
    AgentState.FINISHED,
    AgentState.REJECTED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.CANCELLED,
    AgentState.AWAITING_USER_CONFIRMATION,
  ],
  [AgentState.RUNNING]: [
    AgentState.INIT,
    AgentState.RUNNING,
    AgentState.STOPPED,
    AgentState.FINISHED,
    AgentState.REJECTED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.CANCELLED,
    AgentState.AWAITING_USER_CONFIRMATION,
  ],
  [AgentState.STOPPED]: [AgentState.INIT, AgentState.STOPPED],
  [AgentState.USER_CONFIRMED]: [AgentState.RUNNING],
  [AgentState.USER_REJECTED]: [AgentState.RUNNING],
  [AgentState.AWAITING_USER_CONFIRMATION]: [],
};

interface ButtonProps {
  isDisabled: boolean;
  content: string;
  action: AgentState;
  handleAction: (action: AgentState) => void;
  large?: boolean;
}

function ActionButton({
  isDisabled = false,
  content,
  action,
  handleAction,
  children,
  large = false,
}: React.PropsWithChildren<ButtonProps>): React.ReactNode {
  return (
    <Tooltip content={content} closeDelay={100}>
      <button
        onClick={() => handleAction(action)}
        disabled={isDisabled}
        className={`
          relative overflow-visible cursor-default hover:cursor-pointer group
          disabled:cursor-not-allowed disabled:opacity-60
          ${large ? "rounded-full bg-neutral-800 p-3" : ""}
          transition-all duration-300 ease-in-out
        `}
        type="button"
      >
        <span className="relative z-10 group-hover:filter group-hover:drop-shadow-[0_0_5px_rgba(255,64,0,0.4)]">
          {children}
        </span>
        <span className="absolute -inset-[5px] border-2 border-red-400/40 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 ease-in-out" />
      </button>
    </Tooltip>
  );
}

function AgentControlBar() {
  const { curAgentState } = useSelector((state: RootState) => state.agent);
  const [desiredState, setDesiredState] = React.useState(AgentState.INIT);
  const [isLoading, setIsLoading] = React.useState(false);

  const handleAction = (action: AgentState) => {
    if (IgnoreTaskStateMap[action].includes(curAgentState)) {
      return;
    }

    if (action === AgentState.STOPPED) {
      store.dispatch(clearMessages());
    } else {
      setIsLoading(true);
    }

    setDesiredState(action);
    changeAgentState(action);
  };

  const handleCancelAction = () => {
    Session.cancelCurrentAction();
  };

  useEffect(() => {
    if (curAgentState === desiredState) {
      if (curAgentState === AgentState.STOPPED) {
        store.dispatch(clearMessages());
      }
      setIsLoading(false);
    } else if (curAgentState === AgentState.RUNNING) {
      setDesiredState(AgentState.RUNNING);
    }
    // We only want to run this effect when curAgentState changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curAgentState]);

  return (
    <div className="flex justify-between items-center gap-20">
      <div className="flex items-center gap-3">
        {curAgentState === AgentState.PAUSED ? (
          <ActionButton
            isDisabled={
              isLoading ||
              IgnoreTaskStateMap[AgentState.RUNNING].includes(curAgentState)
            }
            content="Resume the agent task"
            action={AgentState.RUNNING}
            handleAction={handleAction}
            large
          >
            <PlayIcon />
          </ActionButton>
        ) : (
          <ActionButton
            isDisabled={
              isLoading ||
              IgnoreTaskStateMap[AgentState.PAUSED].includes(curAgentState)
            }
            content="Pause the current task"
            action={AgentState.PAUSED}
            handleAction={handleAction}
            large
          >
            <PauseIcon />
          </ActionButton>
        )}
        <ActionButton
          isDisabled={isLoading}
          content="Start a new task"
          action={AgentState.STOPPED}
          handleAction={handleAction}
        >
          <ArrowIcon />
        </ActionButton>
        <ActionButton
          isDisabled={isLoading || curAgentState !== AgentState.RUNNING}
          content="Cancel current task"
          action={AgentState.CANCELLED}
          handleAction={handleCancelAction}
        >
          <StopIcon />
        </ActionButton>
      </div>
    </div>
  );
}

export default AgentControlBar;
