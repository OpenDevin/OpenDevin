import socket from "../socket/socket";
import { appendAssistantMessage } from "../state/chatSlice";
import { setInitialized } from "../state/taskSlice";
import store from "../store";

export const MODELS = [
  "gpt-3.5-turbo-1106",
  "gpt-4-0125-preview",
  "claude-3-haiku-20240307",
  "claude-3-sonnet-20240229",
  "claude-3-sonnet-20240307",
];

export type Model = (typeof MODELS)[number];

export const AGENTS = ["LangchainsAgent", "CodeActAgent"];

export type Agent = (typeof AGENTS)[number];

function changeSetting(setting: string, value: string): void {
  const event = { action: "initialize", args: { [setting]: value } };
  const eventString = JSON.stringify(event);
  socket.send(eventString);
  store.dispatch(setInitialized(false));
  store.dispatch(appendAssistantMessage(`Changed ${setting} to "${value}"`));
}

export function changeModel(model: Model): void {
  changeSetting("model", model);
}

export function changeAgent(agent: Agent): void {
  changeSetting("agent_cls", agent);
}

export function changeDirectory(directory: string): void {
  changeSetting("directory", directory);
}
