import React from "react";
import { Data } from "ws";

interface WebSocketClientOptions {
  token: string | null;
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent<Data>) => void;
  onError?: (event: Event) => void;
  onClose?: (event: Event) => void;
}

interface WebSocketContextType {
  send: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => void;
  start: (options?: WebSocketClientOptions) => void;
  stop: () => void;
  setRuntimeIsInitialized: () => void;
  runtimeActive: boolean;
  isConnected: boolean;
  events: Record<string, unknown>[];
}

const SocketContext = React.createContext<WebSocketContextType | undefined>(
  undefined,
);

interface SocketProviderProps {
  children: React.ReactNode;
}

function SocketProvider({ children }: SocketProviderProps) {
  const wsRef = React.useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  const [runtimeActive, setRuntimeActive] = React.useState(false);
  const [events, setEvents] = React.useState<Record<string, unknown>[]>([]);

  const setRuntimeIsInitialized = () => {
    setRuntimeActive(true);
  };

  const start = React.useCallback((options?: WebSocketClientOptions): void => {
    if (wsRef.current) {
      console.warn(
        "WebSocket connection is already established, but a new one is starting anyways.",
      );
    }

    /*
    const wsUrl = new URL("/", document.baseURI);
    wsUrl.protocol = wsUrl.protocol.replace("http", "ws");
    if (options?.token) wsUrl.searchParams.set("token", options.token);
    const ws = new WebSocket(`${wsUrl.origin}/ws`);
    */
    // TODO: Remove hardcoded URL; may have to use a proxy
    const ws = new WebSocket(
      `ws://localhost:3000/ws${options?.token ? `?token=${options.token}` : ""}`,
    );

    ws.addEventListener("open", (event) => {
      setIsConnected(true);
      options?.onOpen?.(event);
    });

    ws.addEventListener("message", (event) => {
      console.warn(
        "Received message",
        JSON.stringify(JSON.parse(event.data.toString()), null, 2),
      );

      setEvents((prevEvents) => [...prevEvents, JSON.parse(event.data)]);
      options?.onMessage?.(event);
    });

    ws.addEventListener("error", (event) => {
      console.error("SOCKET ERROR", event);

      options?.onError?.(event);
    });

    ws.addEventListener("close", (event) => {
      console.warn("SOCKET CLOSED", event);

      setIsConnected(false);
      setRuntimeActive(false);
      wsRef.current = null;
      options?.onClose?.(event);
    });

    wsRef.current = ws;
  }, []);

  const stop = React.useCallback((): void => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const send = React.useCallback(
    (data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
      if (!wsRef.current) {
        console.error("WebSocket is not connected.");
        return;
      }
      wsRef.current.send(data);
    },
    [],
  );

  const value = React.useMemo(
    () => ({
      send,
      start,
      stop,
      setRuntimeIsInitialized,
      runtimeActive,
      isConnected,
      events,
    }),
    [
      send,
      start,
      stop,
      setRuntimeIsInitialized,
      runtimeActive,
      isConnected,
      events,
    ],
  );

  return (
    <SocketContext.Provider value={value}>{children}</SocketContext.Provider>
  );
}

function useSocket() {
  const context = React.useContext(SocketContext);
  if (context === undefined) {
    throw new Error("useSocket must be used within a SocketProvider");
  }
  return context;
}

export { SocketProvider, useSocket };
