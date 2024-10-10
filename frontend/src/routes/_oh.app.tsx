import { useDisclosure } from "@nextui-org/react";
import React from "react";
import {
  Outlet,
  useFetcher,
  useLoaderData,
  json,
  ClientActionFunctionArgs,
  useRouteLoaderData,
} from "@remix-run/react";
import { useDispatch, useSelector } from "react-redux";
import WebSocket from "ws";
import toast from "react-hot-toast";
import ChatInterface from "#/components/chat/ChatInterface";
import { getSettings } from "#/services/settings";
import Security from "../components/modals/security/Security";
import { Controls } from "#/components/controls";
import store, { RootState } from "#/store";
import { Container } from "#/components/container";
import ActionType from "#/types/ActionType";
import { handleAssistantMessage } from "#/services/actions";
import { addUserMessage, clearMessages } from "#/state/chatSlice";
import { useSocket } from "#/context/socket";
import { sendTerminalCommand } from "#/services/terminalService";
import { appendInput, clearTerminal } from "#/state/commandSlice";
import { useEffectOnce } from "#/utils/use-effect-once";
import CodeIcon from "#/assets/code.svg?react";
import GlobeIcon from "#/assets/globe.svg?react";
import ListIcon from "#/assets/list-type-number.svg?react";
import { createChatMessage } from "#/services/chatService";
import {
  clearFiles,
  clearSelectedRepository,
} from "#/state/initial-query-slice";
import { isGitHubErrorReponse, retrieveLatestGitHubCommit } from "#/api/github";
import OpenHands from "#/api/open-hands";
import AgentState from "#/types/AgentState";
import { base64ToBlob } from "#/utils/base64-to-blob";
import { clientLoader as rootClientLoader } from "#/routes/_oh";
import { clearJupyter } from "#/state/jupyterSlice";
import { FilesProvider } from "#/context/files";

const isAgentStateChange = (
  data: object,
): data is { extras: { agent_state: AgentState } } =>
  "extras" in data &&
  data.extras instanceof Object &&
  "agent_state" in data.extras;

export const clientLoader = async () => {
  const q = store.getState().initalQuery.initialQuery;
  const repo =
    store.getState().initalQuery.selectedRepository ||
    localStorage.getItem("repo");
  const importedProject = store.getState().initalQuery.importedProjectZip;

  const settings = getSettings();
  const token = localStorage.getItem("token");
  const ghToken = localStorage.getItem("ghToken");

  if (token && importedProject) {
    const blob = base64ToBlob(importedProject);
    const file = new File([blob], "imported-project.zip", {
      type: blob.type,
    });
    await OpenHands.uploadFiles(token, [file]);
  }

  if (repo) localStorage.setItem("repo", repo);

  let lastCommit: GitHubCommit | null = null;
  if (ghToken && repo) {
    const data = await retrieveLatestGitHubCommit(ghToken, repo);
    if (isGitHubErrorReponse(data)) {
      // TODO: Handle error
      console.error("Failed to retrieve latest commit", data);
    } else {
      [lastCommit] = data;
    }
  }

  return json({
    settings,
    token,
    ghToken,
    repo,
    q,
    lastCommit,
  });
};

export const clientAction = async ({ request }: ClientActionFunctionArgs) => {
  const formData = await request.formData();

  const token = formData.get("token")?.toString();
  const ghToken = formData.get("ghToken")?.toString();

  if (token) localStorage.setItem("token", token);
  if (ghToken) localStorage.setItem("ghToken", ghToken);

  return json(null);
};

function App() {
  const dispatch = useDispatch();
  const { files } = useSelector((state: RootState) => state.initalQuery);
  const { start, send, setRuntimeIsInitialized, runtimeActive } = useSocket();
  const { settings, token, ghToken, repo, q, lastCommit } =
    useLoaderData<typeof clientLoader>();
  const fetcher = useFetcher();
  const data = useRouteLoaderData<typeof rootClientLoader>("routes/_oh");

  // To avoid re-rendering the component when the user object changes, we memoize the user ID.
  // We use this to ensure the github token is valid before exporting it to the terminal.
  const userId = React.useMemo(() => {
    if (data?.user && !isGitHubErrorReponse(data.user)) return data.user.id;
    return null;
  }, [data?.user]);

  const Terminal = React.useMemo(
    () => React.lazy(() => import("../components/terminal/Terminal")),
    [],
  );

  const exportGitHubTokenToTerminal = (gitHubToken: string) => {
    const command = `export GITHUB_TOKEN=${gitHubToken}`;
    const event = sendTerminalCommand(command);

    send(event);
    dispatch(appendInput(command.replace(gitHubToken, "***")));
  };

  const sendCloneRepoCommandToTerminal = (
    gitHubToken: string,
    repository: string,
  ) => {
    const url = `https://${gitHubToken}@github.com/${repository}.git`;
    const command = `git clone ${url}`;
    const event = sendTerminalCommand(command);

    send(event);
    dispatch(appendInput(command.replace(gitHubToken, "***")));
  };

  const addIntialQueryToChat = (
    query: string,
    base64Files: string[],
    timestamp = new Date().toISOString(),
  ) => {
    dispatch(
      addUserMessage({
        content: query,
        imageUrls: base64Files,
        timestamp,
      }),
    );
  };

  const sendInitialQuery = (query: string, base64Files: string[]) => {
    const timestamp = new Date().toISOString();
    send(createChatMessage(query, base64Files, timestamp));
  };

  const handleOpen = React.useCallback(() => {
    const initEvent = {
      action: ActionType.INIT,
      args: settings,
    };
    send(JSON.stringify(initEvent));

    // display query in UI, but don't send it to the server
    if (q) addIntialQueryToChat(q, files);
  }, [settings]);

  const handleMessage = React.useCallback(
    (message: MessageEvent<WebSocket.Data>) => {
      // set token received from the server
      const parsed = JSON.parse(message.data.toString());
      if ("token" in parsed) {
        fetcher.submit({ token: parsed.token }, { method: "post" });
        return;
      }

      if ("error" in parsed) {
        toast.error(parsed.error);
        fetcher.submit({}, { method: "POST", action: "/end-session" });
        return;
      }

      handleAssistantMessage(message.data.toString());

      // handle first time connection
      if (
        isAgentStateChange(parsed) &&
        parsed.extras.agent_state === AgentState.INIT
      ) {
        setRuntimeIsInitialized();

        // handle new session
        if (!token) {
          if (ghToken && repo) {
            sendCloneRepoCommandToTerminal(ghToken, repo);
            dispatch(clearSelectedRepository()); // reset selected repository; maybe better to move this to '/'?
          }

          if (q) {
            sendInitialQuery(q, files);
            dispatch(clearFiles()); // reset selected files
          }
        }
      }
    },
    [token, ghToken, repo, q, files],
  );

  const startSocketConnection = React.useCallback(() => {
    start({
      token,
      onOpen: handleOpen,
      onMessage: handleMessage,
    });
  }, [token, handleOpen, handleMessage]);

  useEffectOnce(() => {
    // clear and restart the socket connection
    dispatch(clearMessages());
    dispatch(clearTerminal());
    dispatch(clearJupyter());
    startSocketConnection();
  });

  React.useEffect(() => {
    // Export if the user valid, this could happen mid-session so it is handled here
    if (userId && ghToken && runtimeActive) {
      exportGitHubTokenToTerminal(ghToken);
    }
  }, [userId, ghToken, runtimeActive]);

  const {
    isOpen: securityModalIsOpen,
    onOpen: onSecurityModalOpen,
    onOpenChange: onSecurityModalOpenChange,
  } = useDisclosure();

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex h-full overflow-auto gap-3">
        <Container className="w-1/4 max-h-full" label="Chat">
          <ChatInterface />
        </Container>

        <div className="flex flex-col w-3/4 gap-3">
          <Container
            className="h-2/3"
            labels={[
              { label: "Workspace", to: "", icon: <CodeIcon /> },
              { label: "Jupyter", to: "jupyter", icon: <ListIcon /> },
              {
                label: "Browser",
                to: "browser",
                icon: <GlobeIcon />,
                isBeta: true,
              },
            ]}
          >
            <FilesProvider>
              <Outlet />
            </FilesProvider>
          </Container>
          {/* Terminal uses some API that is not compatible in a server-environment. For this reason, we lazy load it to ensure
           * that it loads only in the client-side. */}
          <Container className="h-1/3 overflow-scroll" label="Terminal">
            <React.Suspense fallback={<div className="h-full" />}>
              <Terminal />
            </React.Suspense>
          </Container>
        </div>
      </div>

      <div className="h-[60px]">
        <Controls
          setSecurityOpen={onSecurityModalOpen}
          showSecurityLock={!!settings.SECURITY_ANALYZER}
          lastCommitData={lastCommit}
        />
      </div>
      <Security
        isOpen={securityModalIsOpen}
        onOpenChange={onSecurityModalOpenChange}
        securityAnalyzer={settings.SECURITY_ANALYZER}
      />
    </div>
  );
}

export default App;
