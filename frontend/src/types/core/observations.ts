import AgentState from "../AgentState";
import { OpenHandsObservationEvent } from "./base";

export interface AgentStateChangeObservation
  extends OpenHandsObservationEvent<"agent_state_changed"> {
  source: "agent";
  extras: {
    agent_state: AgentState;
  };
}

export interface CommandObservation extends OpenHandsObservationEvent<"run"> {
  source: "agent";
  extras: {
    command: string;
    command_id: number;
    exit_code: number;
    hidden: boolean;
  };
}

export interface IPythonObservation
  extends OpenHandsObservationEvent<"run_ipython"> {
  source: "agent";
  extras: {
    code: string;
  };
}

export interface DelegateObservation
  extends OpenHandsObservationEvent<"delegate"> {
  source: "agent";
  extras: {
    outputs: Record<string, unknown>;
  };
}

export interface BrowseObservation extends OpenHandsObservationEvent<"browse"> {
  source: "agent";
  extras: {
    url: string;
    screenshot: string;
    error: boolean;
    open_page_urls: string[];
    active_page_index: number;
    dom_object: Record<string, unknown>;
    axtree_object: Record<string, unknown>;
    extra_element_properties: Record<string, unknown>;
    last_browser_action: string;
    last_browser_action_error: unknown;
    focused_element_bid: string;
  };
}

export interface ErrorObservation extends OpenHandsObservationEvent<"error"> {
  source: "agent";
}

export type OpenHandsObservation =
  | AgentStateChangeObservation
  | CommandObservation
  | IPythonObservation
  | DelegateObservation
  | BrowseObservation
  | ErrorObservation;
