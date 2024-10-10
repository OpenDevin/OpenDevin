import React from "react";
import {
  defer,
  useRouteError,
  isRouteErrorResponse,
  useNavigation,
  useLocation,
  useLoaderData,
  useFetcher,
  Outlet,
} from "@remix-run/react";
import { retrieveGitHubUser, isGitHubErrorReponse } from "#/api/github";
import OpenHands from "#/api/open-hands";
import CogTooth from "#/assets/cog-tooth";
import { SettingsForm } from "#/components/form/settings-form";
import AccountSettingsModal from "#/components/modals/AccountSettingsModal";
import { DangerModal } from "#/components/modals/confirmation-modals/danger-modal";
import LoadingProjectModal from "#/components/modals/LoadingProject";
import { ModalBackdrop } from "#/components/modals/modal-backdrop";
import { UserAvatar } from "#/components/user-avatar";
import { useSocket } from "#/context/socket";
import i18n from "#/i18n";
import { getSettings, settingsAreUpToDate } from "#/services/settings";
import AllHandsLogo from "#/assets/branding/all-hands-logo.svg?react";
import NewProjectIcon from "#/assets/new-project.svg?react";
import DocsIcon from "#/assets/docs.svg?react";

export const clientLoader = async () => {
  let token = localStorage.getItem("token");
  const ghToken = localStorage.getItem("ghToken");

  let user: GitHubUser | GitHubErrorReponse | null = null;
  if (ghToken) user = await retrieveGitHubUser(ghToken);

  const settings = getSettings();
  await i18n.changeLanguage(settings.LANGUAGE);

  const settingsIsUpdated = settingsAreUpToDate();
  if (!settingsIsUpdated) {
    localStorage.removeItem("token");
    token = null;
  }

  return defer({
    token,
    ghToken,
    user,
    settingsIsUpdated,
    settings,
  });
};

export function ErrorBoundary() {
  const error = useRouteError();

  if (isRouteErrorResponse(error)) {
    return (
      <div>
        <h1>{error.status}</h1>
        <p>{error.statusText}</p>
        <pre>
          {error.data instanceof Object
            ? JSON.stringify(error.data)
            : error.data}
        </pre>
      </div>
    );
  }
  if (error instanceof Error) {
    return (
      <div>
        <h1>Uh oh, an error occurred!</h1>
        <pre>{error.message}</pre>
      </div>
    );
  }

  return (
    <div>
      <h1>Uh oh, an unknown error occurred!</h1>
    </div>
  );
}

export default function MainApp() {
  const { stop, isConnected } = useSocket();
  const navigation = useNavigation();
  const location = useLocation();
  const { token, user, settingsIsUpdated, settings } =
    useLoaderData<typeof clientLoader>();
  const loginFetcher = useFetcher({ key: "login" });
  const logoutFetcher = useFetcher({ key: "logout" });
  const endSessionFetcher = useFetcher({ key: "end-session" });

  const [accountSettingsModalOpen, setAccountSettingsModalOpen] =
    React.useState(false);
  const [settingsModalIsOpen, setSettingsModalIsOpen] = React.useState(false);
  const [startNewProjectModalIsOpen, setStartNewProjectModalIsOpen] =
    React.useState(false);
  const [data, setData] = React.useState<{
    models: string[];
    agents: string[];
    securityAnalyzers: string[];
  }>({
    models: [],
    agents: [],
    securityAnalyzers: [],
  });

  React.useEffect(() => {
    // We fetch this here instead of the data loader because the server seems to block
    // the retrieval when the session is closing -- preventing the screen from rendering until
    // the fetch is complete
    (async () => {
      const [models, agents, securityAnalyzers] = await Promise.all([
        OpenHands.getModels(),
        OpenHands.getAgents(),
        OpenHands.getSecurityAnalyzers(),
      ]);

      setData({ models, agents, securityAnalyzers });
    })();
  }, []);

  React.useEffect(() => {
    // If the github token is invalid, open the account settings modal again
    if (isGitHubErrorReponse(user)) {
      setAccountSettingsModalOpen(true);
    }
  }, [user]);

  React.useEffect(() => {
    if (location.pathname === "/") {
      // If the user is on the home page, we should stop the socket connection.
      // This is relevant when the user redirects here for whatever reason.
      if (isConnected) stop();
    }
  }, [location.pathname]);

  const handleUserLogout = () => {
    logoutFetcher.submit(
      {},
      {
        method: "POST",
        action: "/logout",
      },
    );
  };

  const handleAccountSettingsModalClose = () => {
    // If the user closes the modal without connecting to GitHub,
    // we need to log them out to clear the invalid token from the
    // local storage
    if (isGitHubErrorReponse(user)) handleUserLogout();
    setAccountSettingsModalOpen(false);
  };

  const handleEndSession = () => {
    setStartNewProjectModalIsOpen(false);
    // call new session action and redirect to '/'
    endSessionFetcher.submit(new FormData(), {
      method: "POST",
      action: "/end-session",
    });
  };

  return (
    <div className="bg-root-primary p-3 h-screen min-w-[1024px] overflow-x-hidden flex gap-3">
      <aside className="px-1 flex flex-col gap-[15px]">
        <button
          type="button"
          aria-label="All Hands Logo"
          onClick={() => {
            if (location.pathname !== "/") setStartNewProjectModalIsOpen(true);
          }}
        >
          <AllHandsLogo width={34} height={23} />
        </button>
        <nav className="py-[18px] flex flex-col items-center gap-[18px]">
          <UserAvatar
            user={user}
            isLoading={loginFetcher.state !== "idle"}
            onLogout={handleUserLogout}
            handleOpenAccountSettingsModal={() =>
              setAccountSettingsModalOpen(true)
            }
          />
          <button
            type="button"
            className="w-8 h-8 rounded-full hover:opacity-80 flex items-center justify-center"
            onClick={() => setSettingsModalIsOpen(true)}
            aria-label="Settings"
          >
            <CogTooth />
          </button>
          <a
            href="https://docs.all-hands.dev"
            target="_blank"
            rel="noreferrer noopener"
            className="w-8 h-8 rounded-full hover:opacity-80 flex items-center justify-center"
            aria-label="Documentation"
          >
            <DocsIcon width={28} height={28} />
          </a>
          {!!token && (
            <button
              type="button"
              aria-label="Start new project"
              onClick={() => setStartNewProjectModalIsOpen(true)}
            >
              <NewProjectIcon width={28} height={28} />
            </button>
          )}
        </nav>
      </aside>
      <div className="h-full w-full relative">
        <Outlet />
        {navigation.state === "loading" && location.pathname !== "/" && (
          <ModalBackdrop>
            <LoadingProjectModal
              message={
                endSessionFetcher.state === "loading"
                  ? "Ending session, please wait..."
                  : undefined
              }
            />
          </ModalBackdrop>
        )}
        {(!settingsIsUpdated || settingsModalIsOpen) && (
          <ModalBackdrop onClose={() => setSettingsModalIsOpen(false)}>
            <div className="bg-root-primary w-[384px] p-6 rounded-xl flex flex-col gap-2">
              <span className="text-xl leading-6 font-semibold -tracking-[0.01em">
                AI Provider Configuration
              </span>
              <p className="text-xs text-[#A3A3A3]">
                To continue, connect an OpenAI, Anthropic, or other LLM account
              </p>
              {isConnected && (
                <p className="text-xs text-danger">
                  Changing settings during an active session will end the
                  session
                </p>
              )}
              <SettingsForm
                settings={settings}
                models={data.models}
                agents={data.agents}
                securityAnalyzers={data.securityAnalyzers}
                onClose={() => setSettingsModalIsOpen(false)}
              />
            </div>
          </ModalBackdrop>
        )}
        {accountSettingsModalOpen && (
          <ModalBackdrop onClose={handleAccountSettingsModalClose}>
            <AccountSettingsModal
              onClose={handleAccountSettingsModalClose}
              selectedLanguage={settings.LANGUAGE}
              gitHubError={isGitHubErrorReponse(user)}
            />
          </ModalBackdrop>
        )}
        {startNewProjectModalIsOpen && (
          <ModalBackdrop onClose={() => setStartNewProjectModalIsOpen(false)}>
            <DangerModal
              title="Are you sure you want to exit?"
              description="You will lose any unsaved information."
              buttons={{
                danger: {
                  text: "Exit Project",
                  onClick: handleEndSession,
                },
                cancel: {
                  text: "Cancel",
                  onClick: () => setStartNewProjectModalIsOpen(false),
                },
              }}
            />
          </ModalBackdrop>
        )}
      </div>
    </div>
  );
}
