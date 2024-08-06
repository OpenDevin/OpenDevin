enum AgentState {
  LOADING = "loading",
  INIT = "init",
  RUNNING = "running",
  AWAITING_USER_INPUT = "awaiting_user_input",
  PAUSED = "paused",
  STOPPED = "stopped",
  FINISHED = "finished",
  REJECTED = "rejected",
  ERROR = "error",
  CANCELLED = "cancelled",
  AWAITING_USER_CONFIRMATION = "awaiting_user_confirmation",
  USER_CONFIRMED = "user_confirmed",
  USER_REJECTED = "user_rejected",
}

export default AgentState;
