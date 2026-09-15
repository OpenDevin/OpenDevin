import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import McpService from "#/api/mcp-service/mcp-service.api";
import { SecretsService } from "#/api/secrets-service";
import { I18nKey } from "#/i18n/declaration";
import { getConversationState } from "#/utils/conversation-local-storage";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import type { Backend } from "#/api/backend-registry/types";
import AutomationService from "#/api/automation-service/automation-service.api";
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import {
  RecommendedAutomationsSection,
  getAutomationsByPopularity,
} from "#/components/features/automations/recommended-automations-section";
import {
  AUTOMATION_CATALOG,
  type RecommendedAutomation,
} from "@openhands/extensions/automations";

const {
  mockCreateConversationMutate,
  mockCreateSecret,
  mockDisplayErrorToast,
  mockUseSettings,
} = vi.hoisted(() => ({
  mockCreateConversationMutate: vi.fn(),
  mockCreateSecret: vi.fn(),
  mockDisplayErrorToast: vi.fn(),
  mockUseSettings: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, vars?: Record<string, unknown>) => {
      if (vars?.name) return `${key}:${String(vars.name)}`;
      if (vars?.count != null) return `${key}:${String(vars.count)}`;
      return key;
    },
  }),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mockCreateConversationMutate,
    isPending: false,
  }),
}));

vi.mock("#/hooks/mutation/use-create-secret", () => ({
  useCreateSecret: () => ({ mutateAsync: mockCreateSecret }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => ({
  ...(await importOriginal()),
  displayErrorToast: mockDisplayErrorToast,
}));

const localBackend: Backend = {
  id: "local-backend",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "",
  kind: "local",
};

const GITHUB_HOSTED_MCP_URL = "https://api.githubcopilot.com/mcp/";

const cloudBackend: Backend = {
  id: "cloud-backend",
  name: "Cloud",
  host: "https://staging.all-hands.dev/",
  apiKey: "cloud-token",
  kind: "cloud",
};

const mockNavigate = vi.fn();

const navigationValue: NavigationContextValue = {
  currentPath: "/automations",
  conversationId: null,
  isNavigating: false,
  navigate: mockNavigate,
};

function renderLauncher({
  withBackendProvider = false,
  variant = "catalog",
}: {
  withBackendProvider?: boolean;
  variant?: "catalog" | "rail";
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  const launcher = (
    <NavigationProvider value={navigationValue}>
      <RecommendedAutomationsLauncher variant={variant} />
    </NavigationProvider>
  );

  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        {withBackendProvider ? (
          <ActiveBackendProvider>{launcher}</ActiveBackendProvider>
        ) : (
          launcher
        )}
      </QueryClientProvider>,
    ),
    queryClient,
  };
}

function settingsWithMcpConfig(mcp_config: unknown) {
  return {
    agent_settings: {
      mcp_config,
    },
  };
}

function settingsWithGithubMcp() {
  return settingsWithMcpConfig({
    github: {
      url: GITHUB_HOSTED_MCP_URL,
      auth: { strategy: "bearer", value: "github-token" },
    },
  });
}

function continueGithubResponderLocally() {
  fireEvent.click(
    screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
  );
  const continueButton = screen.getByTestId(
    "responder-deployment-continue-local",
  );
  fireEvent.click(continueButton);
  return continueButton;
}

describe("recommended automations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    mockUseSettings.mockReturnValue({
      data: settingsWithMcpConfig({}),
    });
    vi.spyOn(SecretsService, "getSecretsOrThrow").mockResolvedValue([]);
    mockCreateSecret.mockResolvedValue(undefined);
    // Pre-flight connectivity test must pass so save mutations are reached.
    vi.spyOn(McpService, "testServer").mockResolvedValue({
      ok: true,
      tools: [],
    });
  });

  afterEach(() => {
    localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("renders the proven automations before the beta ones, each in popularity order", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const cardIds = screen
      .getAllByTestId(/^recommended-automation-card-/)
      .map((card) =>
        card
          .getAttribute("data-testid")
          ?.replace("recommended-automation-card-", ""),
      );

    expect(cardIds).toEqual([
      "github-pr-reviewer",
      "custom-automation",
      "github-issue-to-pr",
      "slack-channel-monitor",
      "github-agents-md-maintainer",
      "news-digest",
      "github-repo-monitor",
      "slack-standup-digest",
      "linear-triage-assistant",
      "linear-issue-to-github-pr",
      "gitlab-issue-to-mr",
      "linear-issue-to-gitlab-mr",
      "linear-issue-to-bitbucket-pr",
      "jira-issue-to-pr",
      "qa-changes",
      "jira-issue-to-gitlab-mr",
      "research-brief-writer",
      "jira-issue-to-bitbucket-pr",
      "github-issue-triage",
      "upstream-fork-sync",
      "incident-retrospective-drafter",
    ]);
  });

  it("groups the non-proven automations under a labeled Beta section", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const provenHeading = screen.getByText(
      I18nKey.RECOMMENDED_AUTOMATIONS$SECTION_TITLE,
    ).parentElement!;
    expect(within(provenHeading).getByText("6")).toBeInTheDocument();

    const betaHeading = screen.getByTestId(
      "recommended-automations-beta-heading",
    );
    expect(betaHeading).toHaveTextContent(
      I18nKey.RECOMMENDED_AUTOMATIONS$BETA_LABEL,
    );
    expect(within(betaHeading).getByText("15")).toBeInTheDocument();

    const betaSection = screen.getByTestId(
      "recommended-automations-beta-section",
    );
    expect(
      within(betaSection).getByTestId(
        "recommended-automation-card-slack-standup-digest",
      ),
    ).toBeInTheDocument();
    expect(
      within(betaSection).queryByTestId(
        "recommended-automation-card-github-pr-reviewer",
      ),
    ).not.toBeInTheDocument();
  });

  it("sorts recommendation popularity deterministically when ranks are missing or tied", () => {
    const makeAutomation = (
      id: string,
      popularityRank?: number,
    ): RecommendedAutomation =>
      ({
        ...AUTOMATION_CATALOG[0],
        id,
        popularityRank,
      }) as RecommendedAutomation;

    expect(
      getAutomationsByPopularity([
        makeAutomation("missing-first"),
        makeAutomation("tie-a", 10),
        makeAutomation("top", 20),
        makeAutomation("tie-b", 10),
        makeAutomation("missing-second"),
      ]).map((automation) => automation.id),
    ).toEqual(["top", "tie-a", "tie-b", "missing-first", "missing-second"]);
  });

  it("filters recommendations by required MCP keywords", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        query="standup"
        onSelect={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("recommended-automation-card-slack-standup-digest"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("recommended-automation-card-github-pr-reviewer"),
    ).not.toBeInTheDocument();
  });

  it("shows a left-aligned MCP icon stack on each card", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("recommended-automation-icon-github-pr-reviewer"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("recommended-automation-icon-research-brief-writer"),
    ).toHaveAttribute("data-layout", "overlap");
    expect(
      screen.getByTestId(
        "recommended-automation-icon-incident-retrospective-drafter",
      ),
    ).toHaveAttribute("data-layout", "quadrants");
  });

  it("shows the declared glyph instead of a logo stack when an entry names one", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    // `news-digest` connects to nothing, so there are no logos to stack; it
    // names its own glyph, and the badge must render that rather than the
    // generic placeholder a bare empty stack would give.
    const badge = screen.getByTestId("recommended-automation-icon-news-digest");
    expect(badge).not.toHaveAttribute("data-layout");
    expect(badge.querySelector("svg")).toBeInTheDocument();
    expect(badge.querySelector("img")).not.toBeInTheDocument();
  });

  it("renders missing MCP connect copy as a pill on the same row", () => {
    const offsetWidthDescriptor = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "offsetWidth",
    );
    Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
      configurable: true,
      get() {
        return 120;
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return 2000;
      },
    });
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}

        unobserve() {}

        disconnect() {}
      },
    );

    try {
      render(
        <RecommendedAutomationsSection
          backendKind="local"
          installedServers={[]}
          onSelect={vi.fn()}
        />,
      );

      const pillRow = screen.getByTestId(
        "recommended-automation-pills-research-brief-writer",
      );
      expect(pillRow).toHaveTextContent(
        "RECOMMENDED_AUTOMATIONS$MISSING_CONNECT:2",
      );
      expect(pillRow).toHaveClass("flex-nowrap");
      expect(pillRow).not.toHaveClass("flex-wrap");
    } finally {
      if (offsetWidthDescriptor) {
        Object.defineProperty(
          HTMLElement.prototype,
          "offsetWidth",
          offsetWidthDescriptor,
        );
      } else {
        Reflect.deleteProperty(HTMLElement.prototype, "offsetWidth");
      }
      Reflect.deleteProperty(HTMLElement.prototype, "clientWidth");
      vi.unstubAllGlobals();
    }
  });

  /**
   * Puts a non-MCP-installable requirement back on `jira-issue-to-pr`.
   *
   * It declared the HTTP-only `jira` until @openhands/extensions 0.17.0 swapped it
   * for the MCP `atlassian-rovo`, and no catalog automation declares a non-MCP
   * integration any more. The cases below are about what a card does with one, so
   * the requirement is restored for their duration rather than the assertions
   * rewritten around a property the catalog stopped having. Mirrors the
   * mutate-and-restore already used for the unknown-ID case.
   *
   * @returns the restore function, which the caller must run in a `finally`.
   */
  function requireNonMcpIntegration(): () => void {
    const automation = AUTOMATION_CATALOG.find(
      (item) => item.id === "jira-issue-to-pr",
    )!;
    const mutable = automation as RecommendedAutomation & {
      requires: { integrations: Record<string, { message?: string }> };
    };
    const original = mutable.requires.integrations;
    const { "atlassian-rovo": rovo, ...rest } = original;
    // Keyed first, so the pill order and the install queue start where they did.
    mutable.requires.integrations = {
      jira: {
        message: rovo?.message ?? "Reads the project for issues.",
      },
      ...rest,
    };
    return () => {
      mutable.requires.integrations = original;
    };
  }

  it("keeps a non-MCP-installable integration visible on its card instead of dropping it", () => {
    const restoreRequirement = requireNonMcpIntegration();
    // SkillCardPillRow folds pills behind "+N more" when it measures zero
    // widths in jsdom; give it room so every pill renders.
    const offsetWidthDescriptor = Object.getOwnPropertyDescriptor(
      HTMLElement.prototype,
      "offsetWidth",
    );
    Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
      configurable: true,
      get() {
        return 120;
      },
    });
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return 2000;
      },
    });
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}

        unobserve() {}

        disconnect() {}
      },
    );

    try {
      render(
        <RecommendedAutomationsSection
          backendKind="local"
          installedServers={[]}
          onSelect={vi.fn()}
        />,
      );

      // jira-issue-to-pr declares jira (HTTP-only catalog entry) and github
      // (MCP). Both belong on the card; jira is labeled as external setup.
      const pillRow = screen.getByTestId(
        "recommended-automation-pills-jira-issue-to-pr",
      );
      expect(pillRow).toHaveTextContent("Jira");
      expect(pillRow).toHaveTextContent("GitHub");
      expect(
        within(pillRow).getByTestId("automation-integration-external-jira"),
      ).toHaveTextContent("RECOMMENDED_AUTOMATIONS$EXTERNAL_SETUP");
      expect(
        within(pillRow).queryByTestId("automation-integration-external-github"),
      ).not.toBeInTheDocument();

      // The connect-before-launch count only covers what the install flow can
      // actually connect, so jira does not inflate it.
      expect(pillRow).toHaveTextContent(
        "RECOMMENDED_AUTOMATIONS$MISSING_CONNECT:1",
      );
    } finally {
      restoreRequirement();
      if (offsetWidthDescriptor) {
        Object.defineProperty(
          HTMLElement.prototype,
          "offsetWidth",
          offsetWidthDescriptor,
        );
      } else {
        Reflect.deleteProperty(HTMLElement.prototype, "offsetWidth");
      }
      Reflect.deleteProperty(HTMLElement.prototype, "clientWidth");
      vi.unstubAllGlobals();
    }
  });

  it("finds an automation by searching for its non-MCP-installable integration", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        query="jira"
        onSelect={vi.fn()}
      />,
    );

    expect(
      screen.getByTestId("recommended-automation-card-jira-issue-to-pr"),
    ).toBeInTheDocument();
  });

  it("does not silently hide an unknown required integration ID", () => {
    const automation = AUTOMATION_CATALOG.find(
      (item) => item.id === "jira-issue-to-pr",
    )!;
    const mutableAutomation = automation as RecommendedAutomation & {
      requires: {
        integrations: Record<
          string,
          { required?: false; setupRequired?: boolean }
        >;
      };
    };
    const originalIntegrations = mutableAutomation.requires.integrations;
    mutableAutomation.requires.integrations = {
      ...originalIntegrations,
      "unknown-integration": {},
    };

    try {
      render(
        <RecommendedAutomationsSection
          backendKind="local"
          installedServers={[]}
          onSelect={vi.fn()}
        />,
      );

      // The current implementation drops unknown IDs while resolving the
      // catalog, so this assertion intentionally fails until they are surfaced.
      expect(
        screen.getByTestId(
          "recommended-automation-integration-unknown-integration",
        ),
      ).toBeInTheDocument();
    } finally {
      mutableAutomation.requires.integrations = originalIntegrations;
    }
  });

  it("queues installs only for MCP-installable required integrations", async () => {
    const restoreRequirement = requireNonMcpIntegration();

    try {
      renderLauncher();

      fireEvent.click(
        screen.getByTestId("recommended-automation-card-jira-issue-to-pr"),
      );

      // jira cannot go through the local MCP install flow, so the queue starts
      // directly at github rather than failing or skipping the automation.
      const modal = await screen.findByTestId("mcp-install-modal");
      expect(modal).toHaveAttribute("data-marketplace-id", "github");
      expect(mockCreateConversationMutate).not.toHaveBeenCalled();
    } finally {
      restoreRequirement();
    }
  });

  it("shows a decorative plus badge on each card without toggle behavior", () => {
    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={vi.fn()}
      />,
    );

    const plusBadge = screen.getByTestId(
      "recommended-automation-plus-github-pr-reviewer",
    );
    expect(plusBadge.tagName).toBe("SPAN");
    expect(plusBadge).toHaveAttribute("aria-hidden", "true");
    expect(plusBadge.className).toContain(
      "hover:bg-[var(--oh-interactive-hover)]",
    );
    expect(plusBadge.querySelector('[role="switch"]')).not.toBeInTheDocument();
  });

  it("selects a recommendation directly from its card", () => {
    const automation = AUTOMATION_CATALOG.find(
      (item) => item.id === "github-pr-reviewer",
    )!;
    const onSelect = vi.fn();

    render(
      <RecommendedAutomationsSection
        backendKind="local"
        installedServers={[]}
        onSelect={onSelect}
      />,
    );

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    expect(onSelect).toHaveBeenCalledWith(automation);
  });

  it("opens the MCP install modal instead of launching when the required MCP is missing", async () => {
    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));

    const modal = await screen.findByTestId("mcp-install-modal");
    expect(modal).toHaveAttribute("data-marketplace-id", "github");
    expect(screen.getByTestId("mcp-install-field-url")).toHaveValue(
      GITHUB_HOSTED_MCP_URL,
    );
    expect(screen.getByTestId("mcp-install-field-api_key")).toBeInTheDocument();
    expect(
      screen.queryByTestId("mcp-install-field-command-readonly"),
    ).toBeNull();
    expect(
      screen.queryByTestId("mcp-install-field-GITHUB_PERSONAL_ACCESS_TOKEN"),
    ).toBeNull();
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();
  });

  it("saves OPENHANDS_URL before opening a responder setup form", async () => {
    mockUseSettings.mockReturnValue({
      data: settingsWithGithubMcp(),
    });

    renderLauncher();
    continueGithubResponderLocally();

    await waitFor(() =>
      expect(mockCreateSecret).toHaveBeenCalledWith({
        name: "OPENHANDS_URL",
        value: window.location.origin,
      }),
    );
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();
  });

  it("preserves an existing OPENHANDS_URL when starting a local responder", async () => {
    mockUseSettings.mockReturnValue({
      data: settingsWithGithubMcp(),
    });
    vi.mocked(SecretsService.getSecretsOrThrow).mockResolvedValue([
      { name: "OPENHANDS_URL" },
    ]);

    renderLauncher();

    continueGithubResponderLocally();

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        "/automations/new/github-pr-reviewer",
      ),
    );
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });

  it("uses fresh secrets instead of a cached OPENHANDS_URL", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    const { queryClient } = renderLauncher();
    queryClient.setQueryData(
      ["secrets", localBackend.id, null],
      [{ name: "OPENHANDS_URL" }],
    );

    continueGithubResponderLocally();

    await waitFor(() => expect(mockCreateSecret).toHaveBeenCalledTimes(1));
    expect(SecretsService.getSecretsOrThrow).toHaveBeenCalledTimes(1);
  });

  it("matches the OPENHANDS_URL secret name exactly", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    vi.mocked(SecretsService.getSecretsOrThrow).mockResolvedValue([
      { name: "OPENHANDS_URL_BACKUP" },
    ]);

    renderLauncher();
    continueGithubResponderLocally();

    await waitFor(() => expect(mockCreateSecret).toHaveBeenCalledTimes(1));
  });

  it("waits for the secret save and invalidates the cache before continuing", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    let resolveSave: (() => void) | undefined;
    mockCreateSecret.mockReturnValue(
      new Promise<void>((resolve) => {
        resolveSave = resolve;
      }),
    );
    const { queryClient } = renderLauncher();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const continueButton = continueGithubResponderLocally();

    await waitFor(() => expect(mockCreateSecret).toHaveBeenCalledTimes(1));
    expect(continueButton).toBeDisabled();
    expect(mockNavigate).not.toHaveBeenCalled();

    await act(async () => resolveSave?.());

    await waitFor(() => expect(mockNavigate).toHaveBeenCalledTimes(1));
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["secrets"] });
    expect(invalidateSpy.mock.invocationCallOrder[0]).toBeLessThan(
      mockNavigate.mock.invocationCallOrder[0],
    );
  });

  it("keeps the responder modal open when the fresh secret read fails", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    vi.mocked(SecretsService.getSecretsOrThrow).mockRejectedValue(
      new Error("secret read failed"),
    );

    renderLauncher();
    continueGithubResponderLocally();

    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("secret read failed"),
    );
    expect(
      screen.getByTestId("responder-deployment-modal"),
    ).toBeInTheDocument();
    expect(mockCreateSecret).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("blocks duplicate local continues while secrets are loading", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    vi.mocked(SecretsService.getSecretsOrThrow).mockReturnValue(
      new Promise(() => {}),
    );

    renderLauncher();
    const continueButton = continueGithubResponderLocally();
    fireEvent.click(continueButton);

    await waitFor(() =>
      expect(SecretsService.getSecretsOrThrow).toHaveBeenCalledTimes(1),
    );
    expect(continueButton).toBeDisabled();
    expect(
      screen.getByTestId("responder-deployment-modal-close"),
    ).toBeDisabled();
    expect(
      screen.getByTestId("responder-deployment-open-openhands-cloud"),
    ).toBeDisabled();
    expect(mockCreateSecret).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("does not continue or close the modal when the secret save fails", async () => {
    mockUseSettings.mockReturnValue({ data: settingsWithGithubMcp() });
    mockCreateSecret.mockRejectedValue(new Error("secret save failed"));

    renderLauncher();
    continueGithubResponderLocally();

    await waitFor(() => expect(mockCreateSecret).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(
        screen.getByTestId("responder-deployment-continue-local"),
      ).not.toBeDisabled(),
    );
    expect(
      screen.getByTestId("responder-deployment-modal"),
    ).toBeInTheDocument();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("launches an automation that ships no setup form with its slash command", () => {
    // Arrange
    mockUseSettings.mockReturnValue({
      data: settingsWithMcpConfig({
        linear: { url: "https://mcp.linear.app/mcp" },
      }),
    });

    renderLauncher();

    // Act
    fireEvent.click(
      screen.getByTestId("recommended-automation-card-linear-triage-assistant"),
    );

    // Assert
    expect(mockCreateConversationMutate).toHaveBeenCalledTimes(1);
    const [, options] = mockCreateConversationMutate.mock.calls[0];
    options.onSuccess({ conversation_id: "conversation-1" });

    const draft = getConversationState("conversation-1").draftMessage;
    expect(draft).toBe("/linear-triage:setup");
  });

  it("launches without waiting for an integration the automation can start without", () => {
    // Arrange — Slack and Linear are connected; Notion, which the entry marks
    // as connectable later, is not.
    mockUseSettings.mockReturnValue({
      data: settingsWithMcpConfig({
        slack: { url: "https://mcp.slack.com/mcp" },
        linear: { url: "https://mcp.linear.app/mcp" },
      }),
    });

    renderLauncher();

    // Act
    fireEvent.click(
      screen.getByTestId(
        "recommended-automation-card-incident-retrospective-drafter",
      ),
    );

    // Assert — nothing stands between the click and the launch.
    expect(screen.queryByTestId("mcp-install-modal")).not.toBeInTheDocument();
    expect(mockNavigate).toHaveBeenCalledTimes(1);
  });

  it("prompts to install when the required MCP server is disabled", async () => {
    // A disabled server is withheld from the agent, so treating it as
    // installed would launch an automation that then fails at runtime.
    mockUseSettings.mockReturnValue({
      data: settingsWithMcpConfig({
        github: {
          url: GITHUB_HOSTED_MCP_URL,
          auth: { strategy: "bearer", value: "github-token" },
          enabled: false,
        },
      }),
    });

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));

    await screen.findByTestId("mcp-install-modal");
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();
  });

  it("ignores repeated launches once a responder deployment choice is in flight", async () => {
    mockUseSettings.mockReturnValue({
      data: settingsWithGithubMcp(),
    });

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));
    // The launch is now in flight; re-selecting the card must not launch again.
    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );

    await waitFor(() => expect(mockNavigate).toHaveBeenCalledTimes(1));
  });

  it("hides the recommended automations section on cloud backends", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    renderLauncher({ withBackendProvider: true });

    expect(
      screen.queryByTestId("recommended-automations-section"),
    ).not.toBeInTheDocument();
  });

  it("renders the compact rail instead of the catalog section", async () => {
    // Earlier cases call `vi.unstubAllGlobals()`, which also removes the
    // setup file's ResizeObserver stub the rail's fade tracking needs.
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}

        unobserve() {}

        disconnect() {}
      },
    );
    vi.spyOn(AutomationService, "getAutomations").mockResolvedValue({
      automations: [
        {
          id: "installed-1",
          name: "GitHub code review",
          trigger: { type: "cron", schedule: "0 9 * * *" },
          enabled: true,
          prompt: "Review PRs",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      ],
      total: 1,
    });

    renderLauncher({ variant: "rail" });

    expect(
      await screen.findByTestId("recommended-automations-rail"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("recommended-automations-section"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId(
        "recommended-automation-rail-card-github-pr-reviewer",
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId(
        "recommended-automation-rail-card-slack-standup-digest",
      ),
    ).toBeInTheDocument();
  });

  it("launches the recommendation after the missing MCP is installed", async () => {
    const createSpy = vi
      .spyOn(SettingsService, "createMcpServer")
      .mockResolvedValue(true);

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(screen.getByTestId("responder-deployment-continue-local"));
    await screen.findByTestId("mcp-install-modal");

    fireEvent.change(screen.getByTestId("mcp-install-field-api_key"), {
      target: { value: "github-token" },
    });
    fireEvent.click(screen.getByTestId("mcp-install-submit"));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        "/automations/new/github-pr-reviewer",
      ),
    );
  });

  it("opens the OpenHands Cloud integrations page without launching when the cloud option is chosen", () => {
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);

    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-github-pr-reviewer"),
    );
    fireEvent.click(
      screen.getByTestId("responder-deployment-open-openhands-cloud"),
    );

    expect(openSpy).toHaveBeenCalledWith(
      "https://app.all-hands.dev/settings/integrations",
      "_blank",
      "noopener,noreferrer",
    );
    expect(mockCreateConversationMutate).not.toHaveBeenCalled();
    expect(mockCreateSecret).not.toHaveBeenCalled();

    openSpy.mockRestore();
  });

  it("does not show the deployment choice modal for non-responder automations", () => {
    renderLauncher();

    fireEvent.click(
      screen.getByTestId("recommended-automation-card-linear-triage-assistant"),
    );

    expect(
      screen.queryByTestId("responder-deployment-modal"),
    ).not.toBeInTheDocument();
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });
});
