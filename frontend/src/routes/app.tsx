import { useDisclosure } from "@nextui-org/react";
import { lazy, Suspense } from "react";
import { Toaster } from "react-hot-toast";
import { ActionFunctionArgs, json, LoaderFunctionArgs } from "@remix-run/node";
import { Outlet, useFetcher, useLoaderData } from "@remix-run/react";
import { Provider } from "react-redux";
import ChatInterface from "#/components/chat/ChatInterface";
import { DEFAULT_SETTINGS, getSettings } from "#/services/settings";
import Security from "../components/modals/security/Security";
import { Controls } from "#/components/controls";
import { commitSession, getSession, getSettingsSession } from "#/sessions";
import store from "#/store";
import { Container } from "#/components/container";
import { useWebSocketClient } from "#/hooks/useWebSocketClient";
import ActionType from "#/types/ActionType";
import { handleAssistantMessage } from "#/services/actions";

const Terminal = lazy(() => import("../components/terminal/Terminal"));

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const settingsSession = await getSettingsSession(
    request.headers.get("Cookie"),
  );
  const settings = settingsSession.get("settings") || DEFAULT_SETTINGS;

  const session = await getSession(request.headers.get("Cookie"));
  const token = session.get("token");

  return json({ token, securityAnalyzer: settings.SECURITY_ANALYZER });
};

export const action = async ({ request }: ActionFunctionArgs) => {
  const session = await getSession(request.headers.get("Cookie"));

  const formData = await request.formData();
  const token = formData.get("token")?.toString();

  if (token) {
    session.set("token", token);
  }

  return json(null, {
    headers: {
      "Set-Cookie": await commitSession(session),
    },
  });
};

function App() {
  const { token, securityAnalyzer } = useLoaderData<typeof loader>();
  const fetcher = useFetcher();

  const socket = useWebSocketClient({
    token,
    onOpen: () => {
      const settings = getSettings();
      const event = {
        action: ActionType.INIT,
        args: {
          ...settings,
          LLM_MODEL: settings.USING_CUSTOM_MODEL
            ? settings.CUSTOM_LLM_MODEL
            : settings.LLM_MODEL,
        },
      };

      socket.send(JSON.stringify(event));
    },
    onMessage: (message) => {
      console.warn("Received message", message);
      const parsed = JSON.parse(message.data.toString());
      if ("token" in parsed) {
        fetcher.submit({ token: parsed.token }, { method: "post" });
        return;
      }

      handleAssistantMessage(message.data.toString());
    },
    onClose: (event) => {
      console.warn("Socket closed", event);
    },
    onError: (event) => {
      console.error("Socket error", event);
    },
  });

  const {
    isOpen: securityModalIsOpen,
    onOpen: onSecurityModalOpen,
    onOpenChange: onSecurityModalOpenChange,
  } = useDisclosure();

  return (
    <Provider store={store}>
      <div data-testid="app" className="h-full flex flex-col gap-[10px]">
        <div className="h-full flex gap-3">
          <div className="w-1/4">
            <Container className="h-full" label="Chat">
              <ChatInterface />
            </Container>
          </div>

          <div className="flex flex-col gap-3 w-3/4">
            <Container
              className="h-full"
              labels={[
                { label: "Workspace", to: "" },
                { label: "Jupyter", to: "jupyter" },
                { label: "Browser (experimental)", to: "browser" },
              ]}
            >
              <Outlet />
            </Container>
            {/* Terminal uses some API that is not compatible in a server-environment. For this reason, we lazy load it to ensure
             * that it loads only in the client-side. */}
            <Container className="h-2/5 min-h-0" label="Terminal">
              <Suspense fallback={<div className="h-full" />}>
                <Terminal />
              </Suspense>
            </Container>
          </div>
        </div>
        <Controls
          setSecurityOpen={onSecurityModalOpen}
          showSecurityLock={!!securityAnalyzer}
        />
        <Security
          isOpen={securityModalIsOpen}
          onOpenChange={onSecurityModalOpenChange}
          securityAnalyzer={securityAnalyzer}
        />
        <Toaster />
      </div>
    </Provider>
  );
}

export default App;
