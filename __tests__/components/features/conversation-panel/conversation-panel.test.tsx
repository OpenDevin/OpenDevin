import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import { NavigationProvider } from "#/context/navigation-context";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import React from "react";
import { renderWithProviders } from "test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import type { Backend } from "#/api/backend-registry/types";

// Mock the unified stop conversation hook
const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({
    mutate: mockStopConversationMutate,
  }),
}));

// Helper to create complete AppConversation mock data
// Default timestamps use "now" so conversations are considered recent and
// rendered eagerly by the panel.  Each call produces a timestamp 1 s older
// than the previous one so that the sort-by-updated_at order matches the
// array insertion order (first created → newest → cards[0]).
let _mockConversationCounter = 0;
const createMockConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => {
  const ts = new Date(
    Date.now() - _mockConversationCounter++ * 1000,
  ).toISOString();
  return {
    id: "test-id",
    title: "Test Conversation",
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    updated_at: ts,
    created_at: ts,
    execution_status: ExecutionStatus.FINISHED,
    conversation_url: null,
    created_by_user_id: "user1",
    metrics: null,
    llm_model: null,
    trigger: null,
    pr_number: [],
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  };
};

// Mock toast handlers to prevent unhandled rejection errors
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel", () => {
  const onCloseMock = vi.fn();
  const RouterStub = createRoutesStub([
    {
      Component: () => <ConversationPanel onClose={onCloseMock} />,
      path: "/",
    },
    {
      // Add route to prevent "No routes matched location" warning
      Component: () => null,
      path: "/conversations/:conversationId",
    },
  ]);

  const renderConversationPanel = (
    options?: Parameters<typeof renderWithProviders>[1],
  ) => renderWithProviders(<RouterStub />, options);

  // The old single-click filter menu is now the layouts menu, with the
  // display toggles one level deeper in the Advanced options modal. The
  // modal stays open across row clicks, so a second call is a no-op.
  const openAdvancedOptions = async (
    user: ReturnType<typeof userEvent.setup>,
  ) => {
    if (screen.queryByTestId("advanced-conversation-options-modal")) return;
    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("advanced-options-row"));
  };

  beforeAll(() => {
    vi.mock("react-router", async (importOriginal) => ({
      ...(await importOriginal<typeof import("react-router")>()),
      Link: ({ children }: React.PropsWithChildren) => children,
      useNavigate: vi.fn(() => vi.fn()),
      useLocation: vi.fn(() => ({ pathname: "/conversation" })),
      useParams: vi.fn(() => ({ conversationId: "2" })),
    }));
  });

  const mockConversations: AppConversation[] = [
    createMockConversation({ id: "1", title: "Conversation 1" }),
    createMockConversation({ id: "2", title: "Conversation 2" }),
    createMockConversation({ id: "3", title: "Conversation 3" }),
  ];

  const cloudBackend: Backend = {
    id: "cloud-prod",
    name: "Production",
    host: "https://app.all-hands.dev",
    apiKey: "bearer-key",
    kind: "cloud",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockStopConversationMutate.mockClear();
    _mockConversationCounter = 0;
    usePinnedConversationsStore.setState({ pinsByBackendId: {} });
    useArchivedConversationsStore.setState({ archivesByBackendId: {} });
    useConversationPanelPreferencesStore.setState({
      showOlderConversations: true,
      olderConversationCutoff: "7d",
      showArchivedConversations: false,
      automationFilterMode: "all",
      selectedAutomationNames: [],
      selectedTagFacets: [],
      showTagsMetadata: false,
    });
    // Setup default mock for searchConversations
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [...mockConversations],
      next_page_id: null,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    window.localStorage.clear();
    window.sessionStorage.clear();
    __resetActiveStoreForTests();
  });

  it("pins the active backend onto each conversation link", async () => {
    // A tab opened with cmd/ctrl-click does not reliably inherit the opener's
    // sessionStorage, so the link has to carry the backend it belongs to or
    // the new tab resolves the conversation against whichever backend
    // localStorage happens to hold.
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");

    const href = cards[0].closest("a")?.getAttribute("href");
    expect(href).toBe(`/conversations/1?backend=${SEEDED_DEFAULT_BACKEND_ID}`);
  });

  it("should render the conversations", async () => {
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");

    // NOTE that we filter out conversations that don't have a created_at property
    // (mock data has 4 conversations, but only 3 have a created_at property)
    expect(cards).toHaveLength(3);
  });

  it("includes the active backend scope in conversation card links", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const ScopedRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel onClose={onCloseMock} />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ActiveBackendProvider>
            <NavigationProvider
              value={{
                currentPath: "/",
                conversationId: null,
                isNavigating: false,
                navigate: vi.fn(),
              }}
            >
              <ScopedRouterStub />
            </NavigationProvider>
          </ActiveBackendProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    const title = await screen.findByText("Conversation 1");
    expect(title.closest("a")).toHaveAttribute(
      "href",
      "/conversations/1?backend=cloud-prod&org=org-2",
    );
  });

  it("includes the active backend scope in compact conversation row links", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "running",
          title: "Running Conversation",
          execution_status: ExecutionStatus.RUNNING,
        }),
      ],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ActiveBackendProvider>
            <NavigationProvider
              value={{
                currentPath: "/",
                conversationId: null,
                isNavigating: false,
                navigate: vi.fn(),
              }}
            >
              <CompactRouterStub />
            </NavigationProvider>
          </ActiveBackendProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    expect(
      await screen.findByLabelText("Running Conversation"),
    ).toHaveAttribute(
      "href",
      "/conversations/running?backend=cloud-prod&org=org-2",
    );
  });

  it("should display an empty state when there are no conversations", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [],
      next_page_id: null,
    });

    renderConversationPanel();

    const emptyState = await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(emptyState).toBeInTheDocument();
  });

  it("keeps load more available when the visible list is empty and another page exists", async () => {
    // Client-side filters (archiving, thread scope) can hide every row of the
    // loaded pages. Hiding "Load more" there would strand the remaining
    // backend pages behind an empty list with no way to reach them.
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [],
      next_page_id: "page-2",
    });

    renderConversationPanel();

    await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();
  });

  it("can reach an unarchived conversation on the next page after archiving every loaded row", async () => {
    // Archiving filters rows out of the visible list without changing
    // backend pagination. If every currently loaded conversation is archived
    // while hasNextPage is still true, Load more must stay available and
    // fetching the next page must surface the unarchived conversation.
    const user = userEvent.setup();
    const page1 = [
      createMockConversation({ id: "archived-1", title: "Archived 1" }),
      createMockConversation({ id: "archived-2", title: "Archived 2" }),
    ];
    const page2 = [
      createMockConversation({
        id: "visible-next",
        title: "Unarchived on next page",
      }),
    ];
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockImplementation(async (_limit, pageId) => {
      if (pageId === "page-2") {
        return { items: page2, next_page_id: null };
      }
      return { items: page1, next_page_id: "page-2" };
    });

    useArchivedConversationsStore.setState({
      archivesByBackendId: {
        "default-local": ["archived-1", "archived-2"],
      },
    });
    useConversationPanelPreferencesStore.setState({
      showArchivedConversations: false,
    });

    renderConversationPanel();

    await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(
      screen.queryByText("Unarchived on next page"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("load-more-conversations"));

    expect(
      await screen.findByText("Unarchived on next page"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION$NO_CONVERSATIONS"),
    ).not.toBeInTheDocument();
  });

  it("scopes the list to the automation filter mode across hide and only", async () => {
    // Arrange: two manual conversations plus one automation run recognized
    // by its tags (local backend) and one by its trigger (cloud backend).
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({ id: "1", title: "Manual 1" }),
        createMockConversation({ id: "2", title: "Manual 2" }),
        createMockConversation({
          id: "3",
          title: "Tagged Run",
          tags: { automationname: "Nightly Audit", automationtrigger: "cron" },
        }),
        createMockConversation({
          id: "4",
          title: "Cloud Run",
          trigger: "automation",
        }),
      ],
      next_page_id: null,
    });
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "hide-automations",
    });

    // Act + Assert: hide mode keeps only the manual conversations.
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(2);
    expect(screen.queryByText("Tagged Run")).not.toBeInTheDocument();
    expect(screen.queryByText("Cloud Run")).not.toBeInTheDocument();

    // Act + Assert: only mode inverts the scope.
    act(() => {
      useConversationPanelPreferencesStore.setState({
        automationFilterMode: "only-automations",
      });
    });
    expect(await screen.findByText("Tagged Run")).toBeInTheDocument();
    expect(screen.getByText("Cloud Run")).toBeInTheDocument();
    expect(screen.queryByText("Manual 1")).not.toBeInTheDocument();
  });

  it("filters from the persistent filter bar and couples automation chips to the mode", async () => {
    // Arrange: one tagged manual conversation, one untagged, one automation
    // run (recognized by its automation tags).
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Manual 1",
          tags: { project: "vault" },
        }),
        createMockConversation({ id: "2", title: "Manual 2" }),
        createMockConversation({
          id: "3",
          title: "Nightly Run",
          tags: { automationname: "Nightly Audit", automationtrigger: "cron" },
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    // Selecting a facet in the layouts menu narrows the list.
    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("tag-filters-section"));
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    expect(await screen.findByText("Manual 1")).toBeInTheDocument();
    expect(screen.queryByText("Manual 2")).not.toBeInTheDocument();
    expect(
      useConversationPanelPreferencesStore.getState().selectedTagFacets,
    ).toEqual(["project=vault"]);

    // Deselecting the facet restores the full list.
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    expect(await screen.findByText("Manual 2")).toBeInTheDocument();

    // The automation scope lives in the Advanced options modal.
    await user.click(screen.getByTestId("advanced-options-row"));
    await user.click(screen.getByTestId("automation-filter-only"));
    expect(
      useConversationPanelPreferencesStore.getState().automationFilterMode,
    ).toBe("only-automations");
    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();
    expect(screen.queryByText("Manual 1")).not.toBeInTheDocument();
  });

  it("keeps an active tag filter visible outside the layouts menu", async () => {
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Manual 1",
          tags: { project: "vault" },
        }),
        createMockConversation({ id: "2", title: "Manual 2" }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    // Nothing to announce until something is actually filtering.
    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("tag-filters-section"));
    await user.click(screen.getByTestId("tag-facet-row-project=vault"));
    await user.click(screen.getByTestId("conversation-layouts-toggle"));

    // With the menu closed, the strip is the only thing on screen that says
    // where Manual 2 went — the facet checkmark is two levels inside a menu.
    expect(
      await screen.findByTestId("active-tag-filter-project=vault"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Manual 2")).not.toBeInTheDocument();

    // And the way back out is on the strip too, not buried with the facets.
    await user.click(screen.getByTestId("clear-tag-filters"));
    expect(await screen.findByText("Manual 2")).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-active-tag-filters"),
    ).not.toBeInTheDocument();
  });

  it("keeps a persisted automation-name filter both reachable and visible", async () => {
    // `selectedAutomationNames` is persisted and narrows the list on its own,
    // so it needs a control that can see and undo it — otherwise a reload
    // hides conversations with nothing on screen to say why.
    const user = userEvent.setup();
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Nightly Run",
          tags: { automationid: "a-1", automationname: "Nightly Audit" },
        }),
        createMockConversation({
          id: "2",
          title: "Weekly Run",
          tags: { automationid: "a-2", automationname: "Weekly Sweep" },
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();
    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();

    // The name rows live in the advanced-options modal, under the scope.
    await user.click(screen.getByTestId("conversation-layouts-toggle"));
    await user.click(screen.getByTestId("advanced-options-row"));
    await user.click(screen.getByTestId("automation-filter-only"));
    await user.click(
      await screen.findByTestId("automation-name-row-Nightly Audit"),
    );
    await user.click(screen.getByTestId("advanced-options-close"));

    expect(await screen.findByText("Nightly Run")).toBeInTheDocument();
    expect(screen.queryByText("Weekly Run")).not.toBeInTheDocument();

    // With every menu closed, the strip is the only thing naming the
    // narrowing — and the way back out.
    const chip = await screen.findByTestId(
      "active-automation-filter-Nightly Audit",
    );
    expect(chip).toHaveTextContent("Nightly Audit");

    await user.click(chip);
    expect(await screen.findByText("Weekly Run")).toBeInTheDocument();
    expect(
      useConversationPanelPreferencesStore.getState().selectedAutomationNames,
    ).toEqual([]);
  });

  it("clears tag chips from the cards when the Tag chips preference is off", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "1",
          title: "Tagged 1",
          tags: { project: "vault" },
        }),
      ],
      next_page_id: null,
    });
    useConversationPanelPreferencesStore.setState({ showTagsMetadata: true });

    renderConversationPanel();

    expect(
      await screen.findByTestId("conversation-card-tag-chip"),
    ).toBeInTheDocument();

    act(() => {
      useConversationPanelPreferencesStore.setState({
        showTagsMetadata: false,
      });
    });

    await waitFor(() => {
      expect(
        screen.queryByTestId("conversation-card-tag-chip"),
      ).not.toBeInTheDocument();
    });
  });

  it("keeps load more reachable with the filtered empty message when the automation filter hides every loaded conversation", async () => {
    // Arrange: page 1 holds only an automation run (hidden by the active
    // filter); a manual conversation sits on page 2.
    const user = userEvent.setup();
    vi.spyOn(AgentServerConversationService, "searchConversations")
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "run",
            title: "Tagged Run",
            tags: { automationrunid: "run-1" },
          }),
        ],
        next_page_id: "page-2",
      })
      .mockResolvedValueOnce({
        items: [createMockConversation({ id: "manual", title: "Manual 1" })],
        next_page_id: null,
      });
    useConversationPanelPreferencesStore.setState({
      automationFilterMode: "hide-automations",
    });

    renderConversationPanel();

    // Assert: the filter-specific empty message shows and load more stays
    // reachable.
    await screen.findByText("CONVERSATION_PANEL$NO_AUTOMATION_MATCHES");
    const loadMore = await screen.findByTestId("load-more-conversations");

    // Act: fetching the next page surfaces the manual conversation.
    await user.click(loadMore);
    expect(await screen.findByText("Manual 1")).toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION_PANEL$NO_AUTOMATION_MATCHES"),
    ).not.toBeInTheDocument();
  });

  it("does not flash the loading skeleton during a background refetch when the list is empty", async () => {
    // Arrange: first call (initial load) resolves with an empty list.
    // Second call (the background refetch) is kept in-flight so we can
    // observe the UI while `isFetching` is true — the exact window in
    // which the buggy `isFetching`-gated code flashed the skeleton.
    let resolveRefetch:
      | ((value: { items: AppConversation[]; next_page_id: null }) => void)
      | undefined;
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy
      .mockResolvedValueOnce({ items: [], next_page_id: null })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefetch = resolve;
          }),
      );

    // Use a QueryClient we can reach so we can trigger the background
    // refetch directly. This drives the same code path as the hook's 10s
    // `refetchInterval` (both flip `isFetching` to true while existing
    // data stays in the cache) without paying the cost of fake-timer
    // gymnastics around React Query's async state machine.
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const PanelRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <NavigationProvider
            value={{
              currentPath: "/",
              conversationId: "test-conversation-id",
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <PanelRouterStub />
          </NavigationProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    // Wait for the initial fetch to settle into the empty state.
    expect(
      await screen.findByText("CONVERSATION$NO_CONVERSATIONS"),
    ).toBeInTheDocument();

    // Act: trigger a background refetch directly on the cached query.
    // This drives the same code path as the hook's 10s `refetchInterval`
    // (both flip `isFetching` to true while the cached data stays
    // intact). The second mock holds the request in-flight so the
    // in-flight UI is observable. We fire-and-forget the fetch because
    // awaiting it would hang on the held promise.
    const conversationsQuery = queryClient
      .getQueryCache()
      .getAll()
      .find(
        (query) =>
          query.queryKey[0] === "user" && query.queryKey[1] === "conversations",
      );
    if (!conversationsQuery) {
      throw new Error("conversations query was not registered");
    }
    await act(async () => {
      void conversationsQuery.fetch();
      // Yield once so React Query can dispatch the in-flight state.
      await Promise.resolve();
    });

    // Assert: the background refetch fired, but the skeleton did not
    // flicker back in.
    await waitFor(() => {
      expect(searchConversationsSpy).toHaveBeenCalledTimes(2);
    });
    expect(
      screen.queryByTestId("conversation-card-skeleton"),
    ).not.toBeInTheDocument();

    // Settle the in-flight refetch so React Query can clean up.
    resolveRefetch?.({ items: [], next_page_id: null });
  });

  it("should not display the empty state when there are no conversations and the panel is compact", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    renderWithProviders(<CompactRouterStub />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("conversation-card-skeleton-compact"),
      ).not.toBeInTheDocument();
    });

    expect(
      screen.queryByText("CONVERSATION$NO_CONVERSATIONS"),
    ).not.toBeInTheDocument();
  });

  it("hides closed conversations in compact mode", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "running",
          title: "Running Conversation",
          execution_status: ExecutionStatus.RUNNING,
        }),
        createMockConversation({
          id: "closed",
          title: "Closed Conversation",
          execution_status: ExecutionStatus.PAUSED,
        }),
      ],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    renderWithProviders(<CompactRouterStub />);

    expect(
      await screen.findByLabelText("Running Conversation"),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Closed Conversation"),
    ).not.toBeInTheDocument();
  });

  it("should not render fetch errors in the conversation panel", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockRejectedValue(
      new Error("Failed to fetch conversations"),
    );

    renderConversationPanel();

    await waitFor(() => {
      expect(
        screen.queryByText("Failed to fetch conversations"),
      ).not.toBeInTheDocument();
    });
  });

  it("should cancel deleting a conversation", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    let cards = await screen.findAllByTestId("conversation-card");
    // Closed state is observable via the data-context-menu-open attr on the
    // conversation-card root; visual hiding is covered by Playwright.
    expect(cards[0]).toHaveAttribute("data-context-menu-open", "false");

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");

    // Click the first delete button
    await user.click(deleteButton);

    // Cancel the deletion
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    expect(
      screen.queryByRole("button", { name: /cancel/i }),
    ).not.toBeInTheDocument();

    // Ensure the conversation is not deleted
    cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);
  });

  it("should delete a conversation", async () => {
    const user = userEvent.setup();
    const mockData: AppConversation[] = [
      createMockConversation({ id: "1", title: "Conversation 1" }),
      createMockConversation({ id: "2", title: "Conversation 2" }),
      createMockConversation({ id: "3", title: "Conversation 3" }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockImplementation(async () => ({
      items: mockData,
      next_page_id: null,
    }));

    const deleteConversationSpy = vi.spyOn(
      AgentServerConversationService,
      "deleteConversation",
    );
    deleteConversationSpy.mockImplementation(async (id: string) => {
      const index = mockData.findIndex((conv) => conv.id === id);
      if (index !== -1) {
        mockData.splice(index, 1);
      }
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    // Initially shows 3 conversations (no filtering)
    expect(cards).toHaveLength(3);

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");

    // Click the first delete button
    await user.click(deleteButton);

    // Confirm the deletion
    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    await user.click(confirmButton);

    // Verify modal is closed after confirmation
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });

  it("should archive a conversation and remove it from the list", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    let cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    const firstCardTitle = within(cards[0]).getByText("Conversation 1");
    expect(firstCardTitle).toBeInTheDocument();

    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    await user.click(screen.getByTestId("archive-button"));

    expect(
      screen.getByText("CONVERSATION$CONFIRM_ARCHIVE"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /archive/i }));

    await waitFor(() => {
      expect(
        screen.queryByText("CONVERSATION$CONFIRM_ARCHIVE"),
      ).not.toBeInTheDocument();
    });

    cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(2);
    expect(screen.queryByText("Conversation 1")).not.toBeInTheDocument();
  });

  it("shows archived conversations when the preference is on and restores them", async () => {
    // Archiving must stay reversible — that is the whole distinction from
    // deleting, which removes the conversation from the agent server.
    const user = userEvent.setup();
    useArchivedConversationsStore
      .getState()
      .archiveConversation("default-local", "1");
    useConversationPanelPreferencesStore.setState({
      showArchivedConversations: true,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);
    const archivedCard = cards.find((card) =>
      within(card).queryByText("Conversation 1"),
    )!;
    expect(
      within(archivedCard).getByTestId("conversation-card-archived-chip"),
    ).toBeInTheDocument();

    await user.click(within(archivedCard).getByTestId("ellipsis-button"));
    await user.click(screen.getByTestId("unarchive-button"));

    await waitFor(() => {
      expect(
        useArchivedConversationsStore
          .getState()
          .isArchived("default-local", "1"),
      ).toBe(false);
    });
    expect(
      screen.queryByTestId("conversation-card-archived-chip"),
    ).not.toBeInTheDocument();
  });

  it("should call onClose after clicking a card", async () => {
    const user = userEvent.setup();
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");
    const firstCard = cards[1];

    await user.click(firstCard);

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should refetch data on rerenders", async () => {
    const user = userEvent.setup();
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [...mockConversations],
      next_page_id: null,
    });

    function PanelWithToggle() {
      const [isOpen, setIsOpen] = React.useState(true);
      return (
        <>
          <button type="button" onClick={() => setIsOpen((prev) => !prev)}>
            Toggle
          </button>
          {isOpen && <ConversationPanel onClose={onCloseMock} />}
        </>
      );
    }

    const MyRouterStub = createRoutesStub([
      {
        Component: PanelWithToggle,
        path: "/",
      },
    ]);

    renderWithProviders(<MyRouterStub />);

    const toggleButton = screen.getByText("Toggle");

    // Initial render
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Toggle off
    await user.click(toggleButton);
    expect(screen.queryByTestId("conversation-card")).not.toBeInTheDocument();

    // Toggle on
    await user.click(toggleButton);
    const newCards = await screen.findAllByTestId("conversation-card");
    expect(newCards).toHaveLength(3);
  });

  it("keeps invalid timestamps recent and shows older conversations by default", async () => {
    const now = Date.now();
    const minutesAgo = (minutes: number) =>
      new Date(now - minutes * 60 * 1000).toISOString();

    const user = userEvent.setup();
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockReset();
    searchConversationsSpy
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent Conversation",
            updated_at: minutesAgo(59),
          }),
          createMockConversation({
            id: "invalid",
            title: "Invalid Timestamp",
            updated_at: "invalid-date",
          }),
          createMockConversation({
            id: "missing",
            title: "Missing Timestamp",
            updated_at: undefined as unknown as string,
          }),
          createMockConversation({
            id: "older",
            title: "Older Conversation",
            updated_at: minutesAgo(61),
          }),
        ],
        next_page_id: "page-2",
      })
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "paged",
            title: "Paged Conversation",
            updated_at: minutesAgo(30),
          }),
        ],
        next_page_id: null,
      });

    renderConversationPanel();

    expect(await screen.findByText("Recent Conversation")).toBeInTheDocument();
    expect(screen.getByText("Invalid Timestamp")).toBeInTheDocument();
    expect(screen.getByText("Missing Timestamp")).toBeInTheDocument();
    expect(screen.getByText("Older Conversation")).toBeInTheDocument();
    expect(
      screen.getByTestId("older-conversations-summary"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();

    await user.click(screen.getByTestId("load-more-conversations"));

    await waitFor(() => {
      expect(searchConversationsSpy).toHaveBeenCalledWith(20, "page-2");
    });
    expect(await screen.findByText("Paged Conversation")).toBeInTheDocument();
  });

  it("orders the entire visible list by created_at when Created sort is selected, across the recent/older partition", async () => {
    // Arrange: three conversations whose `created_at` ordering diverges
    // from `updated_at` across the 1-hour partition cutoff. If the panel
    // honored only the within-bucket sort, "Old Touched" (created 10d ago
    // but touched 30m ago) would render in the recent bucket *above* the
    // "Mid Stale" / "Newest Stale" entries that live in the older bucket
    // but were created more recently. Created-sort must order the whole
    // visible list by `created_at`, not just within each partition.
    const now = Date.now();
    const isoMinutesAgo = (m: number) =>
      new Date(now - m * 60 * 1000).toISOString();
    const isoDaysAgo = (d: number) =>
      new Date(now - d * 24 * 60 * 60 * 1000).toISOString();

    useConversationPanelPreferencesStore.setState({
      conversationSort: "updated",
      organizeMode: "chronological",
      showOlderConversations: true,
    });

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "old-touched",
          title: "Old Touched",
          created_at: isoDaysAgo(10),
          updated_at: isoMinutesAgo(30),
        }),
        createMockConversation({
          id: "newest-stale",
          title: "Newest Stale",
          created_at: isoDaysAgo(1),
          updated_at: isoDaysAgo(1),
        }),
        createMockConversation({
          id: "mid-stale",
          title: "Mid Stale",
          created_at: isoDaysAgo(3),
          updated_at: isoDaysAgo(3),
        }),
      ],
      next_page_id: null,
    });

    const user = userEvent.setup();
    renderConversationPanel();
    await screen.findByText("Old Touched");

    // Act: open the filter menu and switch sort to Created.
    await openAdvancedOptions(user);
    await user.click(
      screen.getByRole("menuitemradio", {
        name: /CONVERSATION_PANEL\$SORT_CREATED/,
      }),
    );

    // Assert: rendered cards are in strict created_at desc order across
    // the full visible list, regardless of which partition they came from.
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards.map((card) => card.textContent ?? "")).toEqual([
      expect.stringContaining("Newest Stale"),
      expect.stringContaining("Mid Stale"),
      expect.stringContaining("Old Touched"),
    ]);
  });

  it("should cancel stopping a conversation", async () => {
    const user = userEvent.setup();

    // Create mock data with a RUNNING conversation
    const mockRunningConversations: AppConversation[] = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Stop button should be available for RUNNING conversation
    const stopButton = screen.getByTestId("stop-button");
    expect(stopButton).toBeInTheDocument();

    // Click the stop button
    await user.click(stopButton);

    // Cancel the stopping action
    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    expect(
      screen.queryByRole("button", { name: /cancel/i }),
    ).not.toBeInTheDocument();

    // Ensure the conversation status hasn't changed
    const updatedCards = await screen.findAllByTestId("conversation-card");
    expect(updatedCards).toHaveLength(3);
  });

  it("should stop a conversation", async () => {
    const user = userEvent.setup();

    const mockData: AppConversation[] = [
      createMockConversation({
        id: "1",
        title: "Conversation 1",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Conversation 2",
        execution_status: ExecutionStatus.FINISHED,
      }),
      createMockConversation({
        id: "3",
        title: "Conversation 3",
        execution_status: ExecutionStatus.FINISHED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockImplementation(async () => ({
      items: mockData,
      next_page_id: null,
    }));

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    // Component shows all 3 conversations (no filtering by status)
    expect(cards).toHaveLength(3);

    // Click ellipsis on the first card (RUNNING status)
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const stopButton = screen.getByTestId("stop-button");

    // Click the stop button
    await user.click(stopButton);

    // Confirm the stopping action
    const confirmButton = screen.getByRole("button", { name: /confirm/i });
    await user.click(confirmButton);

    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();

    // Verify the mutation was called
    expect(mockStopConversationMutate).toHaveBeenCalledWith({
      conversationId: "1",
    });
    expect(mockStopConversationMutate).toHaveBeenCalledTimes(1);
  });

  it("should only show stop button for STARTING or RUNNING conversations", async () => {
    const user = userEvent.setup();

    const mockMixedStatusConversations: AppConversation[] = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: mockMixedStatusConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    const getCardByTitle = async (title: string) => {
      const currentCards = await screen.findAllByTestId("conversation-card");
      const card = currentCards.find((candidate) =>
        within(candidate).queryByText(title),
      );
      expect(card).toBeDefined();
      return card as HTMLElement;
    };

    // Test RUNNING conversation - should show stop button
    const runningCard = await getCardByTitle("Running Conversation");
    const runningEllipsisButton =
      within(runningCard).getByTestId("ellipsis-button");
    await user.click(runningEllipsisButton);

    expect(await screen.findByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close before opening the next one.
    await waitFor(() => {
      expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
    });

    // Test STARTING/RUNNING conversation - should show stop button
    const startingCard = await getCardByTitle("Starting Conversation");
    const startingEllipsisButton =
      within(startingCard).getByTestId("ellipsis-button");
    await user.click(startingEllipsisButton);

    expect(await screen.findByTestId("stop-button")).toBeInTheDocument();

    // Click outside to close the menu
    await user.click(document.body);

    // Wait for context menu to close before opening the next one.
    await waitFor(() => {
      expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
    });

    // Test STOPPED conversation - should NOT show stop button
    const stoppedCard = await getCardByTitle("Stopped Conversation");
    const stoppedEllipsisButton =
      within(stoppedCard).getByTestId("ellipsis-button");
    await user.click(stoppedEllipsisButton);

    await waitFor(() => {
      expect(stoppedCard).toHaveAttribute("data-context-menu-open", "true");
    });
    expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
  });

  it("should show edit button in context menu", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Edit button should be visible within the first card's context menu
    const editButton = screen.getByTestId("edit-button");
    expect(editButton).toBeInTheDocument();
    expect(editButton).toHaveTextContent("BUTTON$RENAME");
  });

  it("should enter edit mode when edit button is clicked", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Click edit button within the first card's context menu
    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Should find input field instead of title text
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    expect(titleInput).toBeInTheDocument();
    expect(titleInput.tagName).toBe("INPUT");
    expect(titleInput).toHaveValue("Conversation 1");
    expect(titleInput).toHaveFocus();
  });

  it("should successfully update conversation title", async () => {
    const user = userEvent.setup();

    // Mock the updateConversationTitle API call
    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Updated Title");

    // Blur the input to save
    await user.tab();

    // Verify API call was made with correct parameters
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Updated Title",
    );
  });

  it("should save title when Enter key is pressed", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title and press Enter
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Title Updated via Enter");
    await user.keyboard("{Enter}");

    // Verify API call was made
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Title Updated via Enter",
    );
  });

  it("should trim whitespace from title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with extra whitespace
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "   Trimmed Title   ");
    await user.tab();

    // Verify API call was made with trimmed title
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Trimmed Title",
    );
  });

  it("should revert to original title when empty", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Clear the title completely
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.tab();

    // Verify API was not called
    expect(updateConversationTitleSpy).not.toHaveBeenCalled();
  });

  it("should handle API error when updating title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockRejectedValue(new Error("API Error"));
    // Provide return type for mock

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Failed Update");
    await user.tab();

    // Verify API call was made
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Failed Update",
    );

    // Wait for error handling
    await waitFor(() => {
      expect(updateConversationTitleSpy).toHaveBeenCalled();
    });
  });

  it("should close context menu when edit button is clicked", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Click ellipsis to open context menu
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    // Verify context menu is open (portaled to document.body)
    const contextMenu = screen.getByTestId("context-menu");
    expect(contextMenu).toBeInTheDocument();

    // Click edit button within the open context menu
    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Wait for context menu to close after edit button click.
    await waitFor(() => {
      expect(cards[0]).toHaveAttribute("data-context-menu-open", "false");
    });
  });

  it("should not call API when title is unchanged", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Don't change the title, just blur
    await user.tab();

    // Verify API was NOT called with the same title (since handleConversationTitleChange will always be called)
    expect(updateConversationTitleSpy).not.toHaveBeenCalledWith("1", {
      title: "Conversation 1",
    });
  });

  it("should handle special characters in title", async () => {
    const user = userEvent.setup();

    const updateConversationTitleSpy = vi.spyOn(
      AgentServerConversationService,
      "updateConversationTitle",
    );
    updateConversationTitleSpy.mockResolvedValue(
      createMockConversation({ id: "1", title: "Updated Title" }),
    );

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Enter edit mode
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);

    const editButton = screen.getByTestId("edit-button");
    await user.click(editButton);

    // Edit the title with special characters
    const titleInput = within(cards[0]).getByTestId("conversation-card-title");
    await user.clear(titleInput);
    await user.type(titleInput, "Special @#$%^&*()_+ Characters");
    await user.tab();

    // Verify API call was made with special characters
    expect(updateConversationTitleSpy).toHaveBeenCalledWith(
      "1",
      "Special @#$%^&*()_+ Characters",
    );
  });

  it("should close delete modal when clicking backdrop", async () => {
    const user = userEvent.setup();
    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click delete
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const deleteButton = screen.getByTestId("delete-button");
    await user.click(deleteButton);

    // Modal should be visible
    expect(
      screen.getByRole("button", { name: /confirm/i }),
    ).toBeInTheDocument();

    // Click the backdrop (the dark overlay behind the modal)
    const backdrop = document.querySelector(".bg-black.opacity-60");
    expect(backdrop).toBeInTheDocument();
    await user.click(backdrop!);

    // Modal should be closed
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });

  it("should close stop modal when clicking backdrop", async () => {
    const user = userEvent.setup();

    // Create mock data with a RUNNING conversation
    const mockRunningConversations: AppConversation[] = [
      createMockConversation({
        id: "1",
        title: "Running Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "2",
        title: "Starting Conversation",
        execution_status: ExecutionStatus.RUNNING,
      }),
      createMockConversation({
        id: "3",
        title: "Stopped Conversation",
        execution_status: ExecutionStatus.PAUSED,
      }),
    ];

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: mockRunningConversations,
      next_page_id: null,
    });

    renderConversationPanel();

    const cards = await screen.findAllByTestId("conversation-card");

    // Open context menu and click stop
    const ellipsisButton = within(cards[0]).getByTestId("ellipsis-button");
    await user.click(ellipsisButton);
    const stopButton = screen.getByTestId("stop-button");
    await user.click(stopButton);

    // Modal should be visible
    expect(
      screen.getByRole("button", { name: /confirm/i }),
    ).toBeInTheDocument();

    // Click the backdrop
    const backdrop = document.querySelector(".bg-black.opacity-60");
    expect(backdrop).toBeInTheDocument();
    await user.click(backdrop!);

    // Modal should be closed
    expect(
      screen.queryByRole("button", { name: /confirm/i }),
    ).not.toBeInTheDocument();
  });

  describe("older conversations cutoff", () => {
    const recentIso = () => new Date().toISOString();
    const olderIso = () =>
      new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();

    it("shows conversations older than 1 week and includes a summary line", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(3);
      expect(screen.getByText("Recent")).toBeInTheDocument();
      expect(screen.getByText("Old 1")).toBeInTheDocument();
      expect(screen.getByText("Old 2")).toBeInTheDocument();

      const summary = screen.getByTestId("older-conversations-summary");
      expect(summary).toHaveTextContent("SIDEBAR$CONVERSATIONS");
      expect(
        within(summary).getByTestId("conversation-layouts-toggle"),
      ).toBeInTheDocument();
    });

    it("always renders the conversations header with the filter control", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent1",
            title: "Recent 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "recent2",
            title: "Recent 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      const summary = screen.getByTestId("older-conversations-summary");
      expect(summary).toBeInTheDocument();
      expect(
        within(summary).getByTestId("conversation-layouts-toggle"),
      ).toBeInTheDocument();
    });

    it("shows icons on hide and delete-all filter menu actions", async () => {
      const user = userEvent.setup();
      renderConversationPanel();

      // Delete all lives in the layouts menu itself.
      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllRow = screen.getByTestId("delete-all-conversations");
      expect(deleteAllRow.querySelector("svg")).toBeInTheDocument();
      expect(deleteAllRow).toHaveClass("text-danger");
      expect(deleteAllRow).not.toHaveClass("text-[var(--oh-foreground)]");

      // The older-conversations toggle lives in the Advanced options modal.
      await user.click(screen.getByTestId("advanced-options-row"));
      const hideRow = await screen.findByTestId("toggle-older-conversations");
      expect(hideRow.querySelector("svg")).toBeInTheDocument();
      expect(hideRow).toHaveClass("group");
    });

    it("toggles older conversations visibility via the filter dropdown", async () => {
      const user = userEvent.setup();
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();

      let cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(2);

      await openAdvancedOptions(user);
      let toggle = await screen.findByTestId("toggle-older-conversations");
      expect(toggle).toHaveAttribute("aria-checked", "false");
      await user.click(toggle);

      cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(1);

      toggle = await screen.findByTestId("toggle-older-conversations");
      expect(toggle).toHaveAttribute("aria-checked", "true");
      await user.click(toggle);
      cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(2);
    });

    it("keeps repo/branch metadata hidden by default and toggles it from the filter dropdown", async () => {
      const user = userEvent.setup();
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old-with-repo",
            title: "Old With Repo",
            updated_at: olderIso(),
            selected_repository: "openhands/agent-canvas",
            selected_branch: "main",
            git_provider: "github",
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();
      await screen.findByText("Old With Repo");

      expect(
        screen.queryByTestId("conversation-card-selected-repository"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("conversation-card-selected-branch"),
      ).not.toBeInTheDocument();

      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-repo-branch-metadata"));

      expect(
        await screen.findByTestId("conversation-card-selected-repository"),
      ).toHaveTextContent("openhands/agent-canvas");
      expect(
        await screen.findByTestId("conversation-card-selected-branch"),
      ).toHaveTextContent("main");
    });

    it("delete-all is enabled when no conversations are older than the cutoff and deletes every loaded conversation", async () => {
      const user = userEvent.setup();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockResolvedValue();

      // Fixture: only recent conversations (none older than 1h). Before the
      // fix the "Delete all" button was disabled in this state.
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent-1",
            title: "Recent 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "recent-2",
            title: "Recent 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      renderConversationPanel();
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllButton = await screen.findByTestId(
        "delete-all-conversations",
      );
      expect(deleteAllButton).toBeEnabled();

      await user.click(deleteAllButton);
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(2);
      });
      expect(deleteSpy).toHaveBeenCalledWith("recent-1");
      expect(deleteSpy).toHaveBeenCalledWith("recent-2");
    });

    it("delete-all still deletes hidden archived conversations", async () => {
      // "Show archived" only controls rendering. Delete all must keep using
      // the full loaded collection so archived server-side conversations are
      // not silently left behind (or the action disabled when every loaded
      // row is archived).
      const user = userEvent.setup();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockResolvedValue();

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "visible-1",
            title: "Visible 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "archived-1",
            title: "Archived 1",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "archived-2",
            title: "Archived 2",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: null,
      });

      useArchivedConversationsStore.setState({
        archivesByBackendId: {
          "default-local": ["archived-1", "archived-2"],
        },
      });
      useConversationPanelPreferencesStore.setState({
        showArchivedConversations: false,
      });

      renderConversationPanel();
      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(1);
      expect(screen.getByText("Visible 1")).toBeInTheDocument();
      expect(screen.queryByText("Archived 1")).not.toBeInTheDocument();

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      const deleteAllButton = await screen.findByTestId(
        "delete-all-conversations",
      );
      expect(deleteAllButton).toBeEnabled();

      await user.click(deleteAllButton);
      expect(
        await screen.findByText(/CONVERSATION\$CONFIRM_DELETE_ALL_DESC/),
      ).toBeInTheDocument();
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(deleteSpy).toHaveBeenCalledWith("visible-1");
      expect(deleteSpy).toHaveBeenCalledWith("archived-1");
      expect(deleteSpy).toHaveBeenCalledWith("archived-2");
    });

    it("navigates away after the active conversation is deleted successfully even when another deletion fails", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockImplementation(async (conversationId: string) => {
          if (conversationId === "old2") {
            throw new Error("delete failed");
          }
        });

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      // Active conversation is "old1" — it is among the conversations that
      // get deleted successfully, so we should navigate away.
      renderConversationPanel({
        navigation: { conversationId: "old1", navigate },
      });
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      await user.click(screen.getByTestId("delete-all-conversations"));
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(displayErrorToast).toHaveBeenCalledWith(
        "1 conversation could not be deleted.",
      );
      expect(navigate).toHaveBeenCalledWith("/conversations");
    });

    it("does not navigate away when the active conversation fails to delete", async () => {
      const user = userEvent.setup();
      const navigate = vi.fn();
      const deleteSpy = vi
        .spyOn(AgentServerConversationService, "deleteConversation")
        .mockImplementation(async (conversationId: string) => {
          if (conversationId === "old1") {
            throw new Error("delete failed");
          }
        });

      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
          createMockConversation({
            id: "old2",
            title: "Old 2",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: null,
      });

      // Active conversation is "old1" — its deletion fails, so we must
      // not navigate away from it.
      renderConversationPanel({
        navigation: { conversationId: "old1", navigate },
      });
      await screen.findAllByTestId("conversation-card");

      await user.click(screen.getByTestId("conversation-layouts-toggle"));
      await user.click(screen.getByTestId("delete-all-conversations"));
      await user.click(await screen.findByRole("button", { name: /confirm/i }));

      await waitFor(() => {
        expect(deleteSpy).toHaveBeenCalledTimes(3);
      });
      expect(displayErrorToast).toHaveBeenCalledWith(
        "1 conversation could not be deleted.",
      );
      expect(navigate).not.toHaveBeenCalled();
    });
  });

  describe("active conversation highlight", () => {
    it("marks the currently active conversation with data-active=true", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({ id: "1", title: "Conversation 1" }),
          createMockConversation({ id: "2", title: "Conversation 2" }),
          createMockConversation({ id: "3", title: "Conversation 3" }),
        ],
        next_page_id: null,
      });

      renderWithProviders(<RouterStub />, {
        navigation: { conversationId: "2", currentPath: "/conversations/2" },
      });

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(3);
      expect(cards[0]).toHaveAttribute("data-active", "false");
      expect(cards[1]).toHaveAttribute("data-active", "true");
      expect(cards[2]).toHaveAttribute("data-active", "false");
    });

    it("renders no active card when no conversation is selected", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [createMockConversation({ id: "1", title: "Conversation 1" })],
        next_page_id: null,
      });

      renderWithProviders(<RouterStub />, {
        navigation: { conversationId: null, currentPath: "/" },
      });

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards[0]).toHaveAttribute("data-active", "false");
    });
  });

  describe("load-more link", () => {
    const recentIso = () => new Date().toISOString();
    const olderIso = () =>
      new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();

    it("shows a load-more link when there is a next page and no older conversations are hidden", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
        ],
        next_page_id: "page-2",
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      const loadMore = await screen.findByTestId("load-more-conversations");
      expect(loadMore).toHaveTextContent("CONVERSATION$LOAD_MORE");
    });

    it("hides the load-more link after older conversations are hidden from the filter dropdown", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent",
            updated_at: recentIso(),
          }),
          createMockConversation({
            id: "old1",
            title: "Old 1",
            updated_at: olderIso(),
          }),
        ],
        next_page_id: "page-2",
      });

      renderConversationPanel();

      await screen.findAllByTestId("conversation-card");
      // Older conversations are visible by default, so load-more is visible.
      expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();

      // Hide older conversations via the filter dropdown.
      const user = userEvent.setup();
      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-older-conversations"));

      // Older conversations are hidden → no load-more.
      expect(
        screen.queryByTestId("load-more-conversations"),
      ).not.toBeInTheDocument();

      // After showing older conversations again, the link reappears.
      await openAdvancedOptions(user);
      await user.click(screen.getByTestId("toggle-older-conversations"));
      expect(
        await screen.findByTestId("load-more-conversations"),
      ).toBeInTheDocument();
    });

    it("fetches the next page when the load-more link is clicked", async () => {
      const user = userEvent.setup();
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "recent",
              title: "Recent",
              updated_at: recentIso(),
            }),
          ],
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "page2-1",
              title: "Page 2 Conversation",
              updated_at: recentIso(),
            }),
          ],
          next_page_id: null,
        });

      renderConversationPanel();

      const loadMore = await screen.findByTestId("load-more-conversations");
      await user.click(loadMore);

      await waitFor(() => {
        expect(searchSpy).toHaveBeenCalledTimes(2);
      });

      // After the second page resolves, the link disappears (no more pages).
      await waitFor(() => {
        expect(
          screen.queryByTestId("load-more-conversations"),
        ).not.toBeInTheDocument();
      });
    });

    it("keeps collapsed folder previews stable while still exposing later-page conversations on expand", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      const noWorkspaceConversations = Array.from({ length: 6 }, (_, index) =>
        createMockConversation({
          id: `no-workspace-${index + 1}`,
          title: `No workspace ${index + 1}`,
        }),
      );
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: noWorkspaceConversations,
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "no-workspace-7",
              title: "No workspace 7",
            }),
            createMockConversation({
              id: "no-workspace-8",
              title: "No workspace 8",
            }),
          ],
          next_page_id: "page-3",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "alpha",
              title: "Alpha conversation",
              selected_workspace: "/workspace/alpha",
            }),
          ],
          next_page_id: null,
        });

      const user = userEvent.setup();
      renderConversationPanel();

      const noWorkspaceFolder = await screen.findByTestId(
        "thread-folder-__none_workspace",
      );
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      expect(
        within(noWorkspaceFolder).queryByText("No workspace 7"),
      ).not.toBeInTheDocument();

      // A single click walks past the deepen-only page 2 (its rows are hidden
      // from the collapsed preview, so they are not success) and keeps paging
      // until the brand-new folder on page 3 is discovered.
      await user.click(screen.getByTestId("load-more-conversations"));
      await screen.findByTestId("thread-folder-ws--workspace-alpha");
      expect(searchSpy).toHaveBeenCalledTimes(3);
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      expect(
        within(noWorkspaceFolder).queryByText("No workspace 7"),
      ).not.toBeInTheDocument();

      // Collapsed preview stays frozen; expanding reveals every loaded row.
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      await user.click(
        within(noWorkspaceFolder).getByTestId(
          "thread-folder-view-more-__none_workspace",
        ),
      );
      expect(
        within(noWorkspaceFolder).getAllByTestId("conversation-card"),
      ).toHaveLength(8);
      expect(
        within(noWorkspaceFolder).getByText("No workspace 7"),
      ).toBeInTheDocument();
      expect(
        within(noWorkspaceFolder).getByText("No workspace 8"),
      ).toBeInTheDocument();
    });

    it("caps a grouped load-more click at three pages when no new folder appears", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      const deepenOnlyPage = (page: number, nextPageId: string) => ({
        items: Array.from({ length: 2 }, (_, index) =>
          createMockConversation({
            id: `no-workspace-p${page}-${index + 1}`,
            title: `No workspace p${page}-${index + 1}`,
          }),
        ),
        next_page_id: nextPageId,
      });
      // Exactly four pages are queued: the drive must consume the initial page
      // plus MAX_PAGES_PER_LOAD_MORE_CLICK more and stop — a fetch beyond the
      // cap would find no queued response and fail the test loudly.
      const searchSpy = vi
        .spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce(deepenOnlyPage(1, "page-2"))
        .mockResolvedValueOnce(deepenOnlyPage(2, "page-3"))
        .mockResolvedValueOnce(deepenOnlyPage(3, "page-4"))
        .mockResolvedValueOnce(deepenOnlyPage(4, "page-5"));

      const user = userEvent.setup();
      renderConversationPanel();

      await screen.findByTestId("thread-folder-__none_workspace");
      expect(searchSpy).toHaveBeenCalledTimes(1);

      // Every remaining page only deepens the existing folder, so the driver
      // must stop at the per-click page cap instead of draining the cursor.
      await user.click(screen.getByTestId("load-more-conversations"));
      await waitFor(() => {
        expect(searchSpy).toHaveBeenCalledTimes(4);
      });
      // The drive has ended: the control is idle again and no further page
      // was requested beyond the cap.
      await waitFor(() => {
        expect(
          screen.getByTestId("load-more-conversations"),
        ).toBeInTheDocument();
      });
      expect(searchSpy).toHaveBeenCalledTimes(4);
    });

    it("force-includes the active conversation in a folder preview even when it arrived on a later page", async () => {
      useConversationPanelPreferencesStore.setState({
        organizeMode: "grouped",
      });
      vi.spyOn(AgentServerConversationService, "searchConversations")
        .mockResolvedValueOnce({
          items: Array.from({ length: 5 }, (_, index) =>
            createMockConversation({
              id: `alpha-${index + 1}`,
              title: `Alpha ${index + 1}`,
              selected_workspace: "/workspace/alpha",
            }),
          ),
          next_page_id: "page-2",
        })
        .mockResolvedValueOnce({
          items: [
            createMockConversation({
              id: "alpha-active",
              title: "Alpha Active",
              selected_workspace: "/workspace/alpha",
            }),
          ],
          next_page_id: null,
        });

      const user = userEvent.setup();
      renderConversationPanel({
        navigation: {
          conversationId: "alpha-active",
          currentPath: "/conversations/alpha-active",
        },
      });

      const alphaFolder = await screen.findByTestId(
        "thread-folder-ws--workspace-alpha",
      );
      expect(
        within(alphaFolder).queryByText("Alpha Active"),
      ).not.toBeInTheDocument();

      await user.click(screen.getByTestId("load-more-conversations"));

      await waitFor(() => {
        expect(
          within(
            screen.getByTestId("thread-folder-ws--workspace-alpha"),
          ).getByText("Alpha Active"),
        ).toBeInTheDocument();
      });
      // Still a collapsed preview (limit 5), not the full expanded list.
      expect(
        within(
          screen.getByTestId("thread-folder-ws--workspace-alpha"),
        ).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
    });
  });

  it("toggles all grouped workspace folders from the Conversations header", async () => {
    useConversationPanelPreferencesStore.setState({ organizeMode: "grouped" });
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "alpha-chat",
          title: "Alpha Chat",
          selected_workspace: "/workspace/alpha",
        }),
        createMockConversation({
          id: "beta-chat",
          title: "Beta Chat",
          selected_workspace: "/workspace/beta",
        }),
      ],
      next_page_id: null,
    });

    const user = userEvent.setup();
    renderConversationPanel();

    const toggle = await screen.findByTestId("conversations-folder-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Alpha Chat")).toBeInTheDocument();
    expect(screen.getByText("Beta Chat")).toBeInTheDocument();

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Alpha Chat")).not.toBeInTheDocument();
    expect(screen.queryByText("Beta Chat")).not.toBeInTheDocument();

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Alpha Chat")).toBeInTheDocument();
    expect(screen.getByText("Beta Chat")).toBeInTheDocument();
  });

  it("reorders grouped folders via drag and drop", async () => {
    useConversationPanelPreferencesStore.setState({
      organizeMode: "grouped",
      groupFolderOrder: [],
    });

    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "alpha-chat",
          title: "Alpha Chat",
          selected_workspace: "/workspace/alpha",
        }),
        createMockConversation({
          id: "beta-chat",
          title: "Beta Chat",
          selected_workspace: "/workspace/beta",
        }),
      ],
      next_page_id: null,
    });

    renderConversationPanel();

    const alphaFolder = await screen.findByTestId(
      "thread-folder-ws--workspace-alpha",
    );
    const betaFolder = screen.getByTestId("thread-folder-ws--workspace-beta");
    expect(alphaFolder.compareDocumentPosition(betaFolder)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );

    const dragHandle = screen.getByTestId(
      "thread-folder-drag-ws--workspace-alpha",
    );
    const dataTransfer = {
      effectAllowed: "move",
      dropEffect: "move",
      data: {} as Record<string, string>,
      setData(format: string, value: string) {
        this.data[format] = value;
      },
      getData(format: string) {
        return this.data[format];
      },
    };
    fireEvent.dragStart(dragHandle, { dataTransfer });
    fireEvent.dragOver(betaFolder, { dataTransfer });
    fireEvent.drop(betaFolder, { dataTransfer });

    const reorderedAlpha = screen.getByTestId(
      "thread-folder-ws--workspace-alpha",
    );
    const reorderedBeta = screen.getByTestId(
      "thread-folder-ws--workspace-beta",
    );
    expect(reorderedBeta.compareDocumentPosition(reorderedAlpha)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(
      useConversationPanelPreferencesStore.getState().groupFolderOrder,
    ).toEqual(["ws:/workspace/beta", "ws:/workspace/alpha"]);
  });

  it("shows a pinned section above the conversations list when pins exist", async () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation("default-local", "2");

    renderConversationPanel();

    const pinnedSection = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    expect(
      within(pinnedSection).getByText("CONVERSATION_PANEL$PINNED"),
    ).toBeInTheDocument();
    expect(
      within(pinnedSection).getAllByTestId("conversation-card"),
    ).toHaveLength(1);
    expect(
      within(pinnedSection).getByTestId("conversation-pin-toggle-2"),
    ).toBeInTheDocument();
  });

  it("renders pinned conversations only in the pinned section in chronological mode", async () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation("default-local", "2");

    renderConversationPanel();

    const pinnedSection = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    expect(
      within(pinnedSection).getAllByTestId("conversation-card"),
    ).toHaveLength(1);
    expect(await screen.findAllByTestId("conversation-card")).toHaveLength(3);
    expect(screen.getAllByText("Conversation 2")).toHaveLength(1);
  });

  it("renders pinned conversations only in the pinned section in grouped mode", async () => {
    useConversationPanelPreferencesStore.setState({ organizeMode: "grouped" });
    usePinnedConversationsStore
      .getState()
      .pinConversation("default-local", "2");

    renderConversationPanel();

    const pinnedSection = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    expect(
      within(pinnedSection).getAllByTestId("conversation-card"),
    ).toHaveLength(1);
    expect(await screen.findAllByTestId("conversation-card")).toHaveLength(3);
    expect(screen.getAllByText("Conversation 2")).toHaveLength(1);
  });

  it("hides the pinned section after the last pin is removed", async () => {
    usePinnedConversationsStore
      .getState()
      .pinConversation("default-local", "2");

    const user = userEvent.setup();
    renderConversationPanel();

    const pinnedSection = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    const pinnedCard = within(pinnedSection).getByTestId("conversation-card");
    await user.hover(pinnedCard);
    await user.click(
      within(pinnedSection).getByTestId("conversation-pin-toggle-2"),
    );

    await waitFor(() => {
      expect(
        screen.queryByTestId("conversation-panel-pinned-section"),
      ).not.toBeInTheDocument();
    });
  });

  it("shows only five pinned conversations before a More control", async () => {
    const manyConversations = Array.from({ length: 6 }, (_, index) =>
      createMockConversation({
        id: String(index + 1),
        title: `Conversation ${index + 1}`,
      }),
    );
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: manyConversations,
      next_page_id: null,
    });
    for (const conversation of manyConversations) {
      usePinnedConversationsStore
        .getState()
        .pinConversation("default-local", conversation.id);
    }

    renderConversationPanel();

    const pinnedSection = await screen.findByTestId(
      "conversation-panel-pinned-section",
    );
    expect(
      within(pinnedSection).getAllByTestId("conversation-card"),
    ).toHaveLength(5);
    expect(
      within(pinnedSection).getByTestId("conversation-panel-pinned-view-more"),
    ).toHaveTextContent("CONVERSATION_PANEL$MORE");
  });
});
