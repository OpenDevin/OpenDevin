import ActionType from "#/types/ActionType";
import AgentState from "#/types/AgentState";

export const generateAgentStateChangeEvent = (state: AgentState) =>
  JSON.stringify({
    action: ActionType.CHANGE_AGENT_STATE,
    args: { agent_state: state },
  });
