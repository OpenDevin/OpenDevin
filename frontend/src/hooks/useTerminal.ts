import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import React from "react";
import { Command } from "#/state/commandSlice";
import { getTerminalCommand } from "#/services/terminalService";
import { parseTerminalOutput } from "#/utils/parseTerminalOutput";
import { useSocket } from "#/context/socket";

/*
  NOTE: Tests for this hook are indirectly covered by the tests for the XTermTerminal component.
  The reason for this is that the hook exposes a ref that requires a DOM element to be rendered.
*/

export const useTerminal = (commands: Command[] = []) => {
  const { send } = useSocket();
  const terminal = React.useRef<Terminal | null>(null);
  const fitAddon = React.useRef<FitAddon | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);
  const lastCommandIndex = React.useRef(0);

  const createTerminal = () =>
    new Terminal({
      fontFamily: "Menlo, Monaco, 'Courier New', monospace",
      fontSize: 14,
      theme: {
        background: "#262626",
      },
    });

  const initializeTerminal = () => {
    if (terminal.current) {
      if (fitAddon.current) terminal.current.loadAddon(fitAddon.current);
      if (ref.current) terminal.current.open(ref.current);
    }
  };

  const copySelection = (selection: string) => {
    const clipboardItem = new ClipboardItem({
      "text/plain": new Blob([selection], { type: "text/plain" }),
    });

    navigator.clipboard.write([clipboardItem]);
  };

  const pasteSelection = (callback: (text: string) => void) => {
    navigator.clipboard.readText().then(callback);
  };

  const pasteHandler = (event: KeyboardEvent, cb: (text: string) => void) => {
    const isControlOrMetaPressed =
      event.type === "keydown" && (event.ctrlKey || event.metaKey);

    if (isControlOrMetaPressed) {
      if (event.code === "KeyV") {
        pasteSelection((text: string) => {
          terminal.current?.write(text);
          cb(text);
        });
      }

      if (event.code === "KeyC") {
        const selection = terminal.current?.getSelection();
        if (selection) copySelection(selection);
      }
    }

    return true;
  };

  const handleEnter = (command: string) => {
    terminal.current?.write("\r\n");
    send(getTerminalCommand(command));
  };

  const handleBackspace = (command: string) => {
    terminal.current?.write("\b \b");
    return command.slice(0, -1);
  };

  React.useEffect(() => {
    /* Create a new terminal instance */
    terminal.current = createTerminal();
    fitAddon.current = new FitAddon();

    let resizeObserver: ResizeObserver;
    let commandBuffer = "";

    if (ref.current) {
      /* Initialize the terminal in the DOM */
      initializeTerminal();

      terminal.current.write("$ ");
      terminal.current.onKey(({ key, domEvent }) => {
        if (domEvent.key === "Enter") {
          handleEnter(commandBuffer);
          commandBuffer = "";
        } else if (domEvent.key === "Backspace") {
          if (commandBuffer.length > 0) {
            commandBuffer = handleBackspace(commandBuffer);
          }
        } else {
          // Ignore paste event
          if (key.charCodeAt(0) === 22) {
            return;
          }
          commandBuffer += key;
          terminal.current?.write(key);
        }
      });
      terminal.current.attachCustomKeyEventHandler((event) =>
        pasteHandler(event, (text) => {
          commandBuffer += text;
        }),
      );

      /* Listen for resize events */
      resizeObserver = new ResizeObserver(() => {
        fitAddon.current?.fit();
      });
      resizeObserver.observe(ref.current);
    }

    return () => {
      terminal.current?.dispose();
      resizeObserver.disconnect();
    };
  }, []);

  React.useEffect(() => {
    /* Write commands to the terminal */
    if (terminal.current && commands.length > 0) {
      // Start writing commands from the last command index
      for (let i = lastCommandIndex.current; i < commands.length; i += 1) {
        const command = commands[i];
        terminal.current?.writeln(parseTerminalOutput(command.content));

        if (command.type === "output") {
          terminal.current.write(`\n$ `);
        }
      }

      lastCommandIndex.current = commands.length; // Update the position of the last command
    }
  }, [commands]);

  return ref;
};
