import { OpenHandsActionEvent } from "./base";

export interface UserMessageAction extends OpenHandsActionEvent<"message"> {
  source: "user";
  args: {
    content: string;
    images_urls: string[];
  };
}

export interface CommandAction extends OpenHandsActionEvent<"run"> {
  source: "agent";
  args: {
    command: string;
    is_confirmed: "confirmed" | "rejected" | "awaiting_confirmation";
    thought: string;
    hidden: boolean;
  };
}

export interface AssistantMessageAction
  extends OpenHandsActionEvent<"message"> {
  source: "agent";
  args: {
    content: string;
    images_urls: string[] | null;
    wait_for_response: boolean;
  };
}

export interface IPythonAction extends OpenHandsActionEvent<"run_ipython"> {
  source: "agent";
  args: {
    code: string;
    is_confirmed: "confirmed" | "rejected" | "awaiting_confirmation";
    kernel_init_code: string;
    thought: string;
  };
}

export interface FinishAction extends OpenHandsActionEvent<"finish"> {
  source: "agent";
  args: {
    outputs: Record<string, unknown>;
    thought: string;
  };
}

export interface DelegateAction extends OpenHandsActionEvent<"delegate"> {
  source: "agent";
  timeout: number;
  args: {
    agent: "BrowsingAgent";
    inputs: Record<string, string>;
    thought: string;
  };
}

export interface BrowseAction extends OpenHandsActionEvent<"browse"> {
  source: "agent";
  args: {
    url: string;
    thought: string;
  };
}

export interface BrowseInteractiveAction
  extends OpenHandsActionEvent<"browse_interactive"> {
  source: "agent";
  timeout: number;
  args: {
    browser_actions: string;
    thought: string | null;
    browsergym_send_msg_to_user: string;
  };
}

export interface AddTaskAction extends OpenHandsActionEvent<"add_task"> {
  source: "agent";
  timeout: number;
  args: {
    parent: string;
    goal: string;
    subtasks: unknown[];
    thought: string;
  };
}

export interface ModifyTaskAction extends OpenHandsActionEvent<"modify_task"> {
  source: "agent";
  timeout: number;
  args: {
    task_id: string;
    state: string;
    thought: string;
  };
}

export interface RejectAction extends OpenHandsActionEvent<"reject"> {
  source: "agent";
  args: {
    thought: string;
  };
}

export type OpenHandsAction =
  | UserMessageAction
  | AssistantMessageAction
  | CommandAction
  | IPythonAction
  | FinishAction
  | DelegateAction
  | BrowseAction
  | BrowseInteractiveAction
  | AddTaskAction
  | ModifyTaskAction
  | RejectAction;
