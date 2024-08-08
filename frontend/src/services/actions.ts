import { addAssistantMessage, addUserMessage } from "#/state/chatSlice";
import { setCode, setActiveFilepath } from "#/state/codeSlice";
import { appendInput } from "#/state/commandSlice";
import { appendJupyterInput } from "#/state/jupyterSlice";
import {
  ActionSecurityRisk,
  appendSecurityAnalyzerInput,
} from "#/state/securityAnalyzerSlice";
import { setRootTask } from "#/state/taskSlice";
import store from "#/store";
import ActionType from "#/types/ActionType";
import { ActionMessage } from "#/types/Message";
import { SocketMessage } from "#/types/ResponseType";
import { handleObservationMessage } from "./observations";
import { getRootTask } from "./taskService";

const messageActions = {
  [ActionType.BROWSE]: (message: ActionMessage) => {
    store.dispatch(addAssistantMessage(message.message));
  },
  [ActionType.BROWSE_INTERACTIVE]: (message: ActionMessage) => {
    if (message.args.thought) {
      store.dispatch(addAssistantMessage(message.args.thought));
    } else {
      store.dispatch(addAssistantMessage(message.message));
    }
  },
  [ActionType.WRITE]: (message: ActionMessage) => {
    const { path, content } = message.args;
    store.dispatch(setActiveFilepath(path));
    store.dispatch(setCode(content));
  },
  [ActionType.MESSAGE]: (message: ActionMessage) => {
    if (message.source === "user") {
      store.dispatch(
        addUserMessage({ content: message.args.content, imageUrls: [] }),
      );
    } else {
      store.dispatch(addAssistantMessage(message.args.content));
    }
  },
  [ActionType.FINISH]: (message: ActionMessage) => {
    store.dispatch(addAssistantMessage(message.message));
  },
  [ActionType.REJECT]: (message: ActionMessage) => {
    store.dispatch(addAssistantMessage(message.message));
  },
  [ActionType.DELEGATE]: (message: ActionMessage) => {
    store.dispatch(addAssistantMessage(message.message));
  },
  [ActionType.RUN]: (message: ActionMessage) => {
    if (message.args.thought) {
      store.dispatch(addAssistantMessage(message.args.thought));
    }
    if (
      !message.args.is_confirmed ||
      message.args.is_confirmed !== "rejected"
    ) {
      store.dispatch(appendInput(message.args.command));
    }
  },
  [ActionType.RUN_IPYTHON]: (message: ActionMessage) => {
    if (message.args.thought) {
      store.dispatch(addAssistantMessage(message.args.thought));
    }
    if (
      !message.args.is_confirmed ||
      message.args.is_confirmed !== "rejected"
    ) {
      store.dispatch(appendJupyterInput(message.args.code));
    }
  },
  [ActionType.ADD_TASK]: () => {
    getRootTask().then((fetchedRootTask) =>
      store.dispatch(setRootTask(fetchedRootTask)),
    );
  },
  [ActionType.MODIFY_TASK]: () => {
    getRootTask().then((fetchedRootTask) =>
      store.dispatch(setRootTask(fetchedRootTask)),
    );
  },
};

function getRiskText(risk: ActionSecurityRisk) {
  switch (risk) {
    case ActionSecurityRisk.LOW:
      return "Low Risk";
    case ActionSecurityRisk.MEDIUM:
      return "Medium Risk";
    case ActionSecurityRisk.HIGH:
      return "High Risk";
    case ActionSecurityRisk.UNKNOWN:
    default:
      return "Unknown Risk";
  }
}

export function handleActionMessage(message: ActionMessage) {
  if ("args" in message && "security_risk" in message.args) {
    store.dispatch(appendSecurityAnalyzerInput(message));
  }

  if (
    (message.action === ActionType.RUN ||
      message.action === ActionType.RUN_IPYTHON) &&
    message.args.is_confirmed === "awaiting_confirmation"
  ) {
    if (message.args.thought) {
      store.dispatch(addAssistantMessage(message.args.thought));
    }
    if (message.args.command) {
      store.dispatch(
        addAssistantMessage(
          `Running this command now: \n\`\`\`\`bash\n${message.args.command}\n\`\`\`\`\nEstimated security risk: ${getRiskText(message.args.security_risk as unknown as ActionSecurityRisk)}`,
        ),
      );
    } else if (message.args.code) {
      store.dispatch(
        addAssistantMessage(
          `Running this code now: \n\`\`\`\`python\n${message.args.code}\n\`\`\`\`\nEstimated security risk: ${getRiskText(message.args.security_risk as unknown as ActionSecurityRisk)}`,
        ),
      );
    } else {
      store.dispatch(addAssistantMessage(message.message));
    }
    return;
  }

  if (message.action in messageActions) {
    const actionFn =
      messageActions[message.action as keyof typeof messageActions];
    actionFn(message);
  }
}

export function handleAssistantMessage(data: string | SocketMessage) {
  let socketMessage: SocketMessage;

  if (typeof data === "string") {
    socketMessage = JSON.parse(data) as SocketMessage;
  } else {
    socketMessage = data;
  }

  if ("action" in socketMessage) {
    handleActionMessage(socketMessage);
  } else {
    handleObservationMessage(socketMessage);
  }
}
