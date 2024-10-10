import { IoLockClosed } from "react-icons/io5";
import { useRouteLoaderData } from "@remix-run/react";
import React from "react";
import AgentControlBar from "./AgentControlBar";
import AgentStatusBar from "./AgentStatusBar";
import { ProjectMenuCard } from "./project-menu/ProjectMenuCard";
import { clientLoader as rootClientLoader } from "#/routes/_oh";
import { clientLoader as appClientLoader } from "#/routes/_oh.app";
import { isGitHubErrorReponse } from "#/api/github";

interface ControlsProps {
  setSecurityOpen: (isOpen: boolean) => void;
  showSecurityLock: boolean;
  lastCommitData: GitHubCommit | null;
}

export function Controls({
  setSecurityOpen,
  showSecurityLock,
  lastCommitData,
}: ControlsProps) {
  const rootData = useRouteLoaderData<typeof rootClientLoader>("routes/_oh");
  const appData = useRouteLoaderData<typeof appClientLoader>("routes/_oh.app");

  const projectMenuCardData = React.useMemo(
    () =>
      rootData?.user &&
      !isGitHubErrorReponse(rootData.user) &&
      appData?.repo &&
      lastCommitData
        ? {
            avatar: rootData.user.avatar_url,
            repoName: appData.repo,
            lastCommit: lastCommitData,
          }
        : null,
    [rootData, appData, lastCommitData],
  );

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <AgentControlBar />
        <AgentStatusBar />

        {showSecurityLock && (
          <div
            className="cursor-pointer hover:opacity-80 transition-all"
            style={{ marginRight: "8px" }}
            onClick={() => setSecurityOpen(true)}
          >
            <IoLockClosed size={20} />
          </div>
        )}
      </div>

      <ProjectMenuCard
        isConnectedToGitHub={!!rootData?.ghToken}
        githubData={projectMenuCardData}
      />
    </div>
  );
}
