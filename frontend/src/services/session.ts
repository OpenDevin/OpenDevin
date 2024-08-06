import i18next from "i18next";
import toast from "#/utils/toast";
import { handleAssistantMessage } from "./actions";
import { getToken, setToken, clearToken } from "./auth";
import AgentState from "#/types/AgentState";
import ActionType from "#/types/ActionType";
import { getSettings } from "./settings";
import { I18nKey } from "#/i18n/declaration";

const translate = (key: I18nKey) => i18next.t(key);

// Define a type for the messages
type Message = {
  action: ActionType;
  args: Record<string, unknown>;
};

class Session {
  private static _socket: WebSocket | null = null;

  private static _latest_event_id: number = -1;

  private static _messageQueue: Message[] = [];

  public static _history: Record<string, unknown>[] = [];

  // callbacks contain a list of callable functions
  // event: function, like:
  // open: [function1, function2]
  // message: [function1, function2]
  private static callbacks: {
    [K in keyof WebSocketEventMap]: ((data: WebSocketEventMap[K]) => void)[];
  } = {
    open: [],
    message: [],
    error: [],
    close: [],
  };

  private static _connecting = false;

  private static _disconnecting = false;

  public static restoreOrStartNewSession() {
    if (Session.isConnected()) {
      Session.disconnect();
    }
    Session._connect();
  }

  public static startNewSession() {
    clearToken();
    Session.restoreOrStartNewSession();
  }

  private static _initializeAgent = () => {
    const settings = getSettings();
    const event = { action: ActionType.INIT, args: settings };
    const eventString = JSON.stringify(event);
    Session.send(eventString);
  };

  private static _connect(): void {
    if (Session.isConnected()) return;
    Session._connecting = true;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let wsURL = `${protocol}//${window.location.host}/ws`;
    const token = getToken();
    if (token) {
      wsURL += `?token=${token}`;
      if (Session._latest_event_id !== -1) {
        wsURL += `&latest_event_id=${Session._latest_event_id}`;
      }
    }
    Session._socket = new WebSocket(wsURL);
    Session._setupSocket();
  }

  private static _setupSocket(): void {
    if (!Session._socket) {
      throw new Error(
        translate(I18nKey.SESSION$SOCKET_NOT_INITIALIZED_ERROR_MESSAGE),
      );
    }
    Session._socket.onopen = (e) => {
      toast.success("ws", translate(I18nKey.SESSION$SERVER_CONNECTED_MESSAGE));
      Session._connecting = false;
      Session._initializeAgent();
      Session._flushQueue();
      Session.callbacks.open?.forEach((callback) => {
        callback(e);
      });
    };

    Session._socket.onmessage = (e) => {
      let data = null;
      try {
        data = JSON.parse(e.data);
        Session._history.push(data);
      } catch (err) {
        toast.error(
          "ws",
          translate(I18nKey.SESSION$SESSION_HANDLING_ERROR_MESSAGE),
        );
        return;
      }
      if (data.error && data.error_code === 401) {
        Session._latest_event_id = -1;
        clearToken();
      } else if (data.token) {
        setToken(data.token);
      } else {
        if (data.id !== undefined) {
          Session._latest_event_id = data.id;
        }
        handleAssistantMessage(data);
      }
    };

    Session._socket.onerror = () => {
      toast.error(
        "ws",
        translate(I18nKey.SESSION$SESSION_CONNECTION_ERROR_MESSAGE),
      );
    };

    Session._socket.onclose = () => {
      if (!Session._disconnecting) {
        setTimeout(() => {
          Session.restoreOrStartNewSession();
        }, 3000); // Reconnect after 3 seconds
      }
      Session._disconnecting = false;
    };
  }

  static isConnected(): boolean {
    return (
      Session._socket !== null && Session._socket.readyState === WebSocket.OPEN
    );
  }

  static disconnect(): void {
    Session._disconnecting = true;
    if (Session._socket) {
      Session._socket.close();
    }
    Session._socket = null;
  }

  private static _flushQueue(): void {
    while (Session._messageQueue.length > 0) {
      const message = Session._messageQueue.shift();
      if (message) {
        Session.send(JSON.stringify(message));
      }
    }
  }

  static send(message: string): void {
    const messageObject: Message = JSON.parse(message);

    if (Session._connecting) {
      Session._messageQueue.push(messageObject);
      return;
    }
    if (!Session.isConnected()) {
      throw new Error(
        translate(I18nKey.SESSION$SESSION_CONNECTION_ERROR_MESSAGE),
      );
    }

    if (Session.isConnected()) {
      Session._socket?.send(message);
      Session._history.push(JSON.parse(message));
    } else {
      toast.error(
        "ws",
        translate(I18nKey.SESSION$SESSION_CONNECTION_ERROR_MESSAGE),
      );
    }
  }

  static addEventListener(
    event: string,
    callback: (e: MessageEvent) => void,
  ): void {
    Session._socket?.addEventListener(
      event as keyof WebSocketEventMap,
      callback as (
        this: WebSocket,
        ev: WebSocketEventMap[keyof WebSocketEventMap],
      ) => never,
    );
  }

  static removeEventListener(
    event: string,
    listener: (e: Event) => void,
  ): void {
    Session._socket?.removeEventListener(event, listener);
  }

  static registerCallback<K extends keyof WebSocketEventMap>(
    event: K,
    callbacks: ((data: WebSocketEventMap[K]) => void)[],
  ): void {
    if (Session.callbacks[event] === undefined) {
      return;
    }
    Session.callbacks[event].push(...callbacks);
  }

  static cancelCurrentAction(): void {
    const userMessage = "Task cancelled by user.";
    const changeStateEvent: Message = {
      action: ActionType.CHANGE_AGENT_STATE,
      args: {
        agent_state: AgentState.CANCELLED,
        thought: "User requested cancellation",
        status_message: userMessage,
      },
    };
    Session.send(JSON.stringify(changeStateEvent));
    toast.error("ws", userMessage);
  }
}

export default Session;
