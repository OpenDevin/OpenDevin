# Repository Notes

## General

- This repository is the OpenHands frontend.
- Frontend API adaptation lives mainly in `src/api/`:
  - `option-service` fabricates a web-client config and reads models/providers through `@openhands/typescript-client` LLM endpoints.
  - `settings-service` uses `@openhands/typescript-client` settings APIs for persistence; reads schemas from `/api/settings/agent-schema` and `/api/settings/conversation-schema`, fetches settings with optional `X-Expose-Secrets: encrypted` header for conversation start payloads, and saves settings via PATCH with diffs.
  - `agent-server-conversation-service`, `event-service`, `agent-server-git-service`, and `skills-service` route local agent-server access through `@openhands/typescript-client` rather than direct HTTP calls.
- Supported env vars for deployment:
  - `VITE_BACKEND_BASE_URL` for the agent server base URL.
  - `VITE_SESSION_API_KEY` for optional session auth.
  - `VITE_WORKING_DIR` for the default workspace path sent when starting conversations.
  - `VITE_ENABLE_BROWSER_TOOLS=false` to omit the `browser_tool_set` tool from new conversation payloads.
  - `VITE_BASE_PATH` for serving the SPA under a subpath such as `/canvas`; pair it with `scripts/static-server.mjs --base-path` at runtime.
- Public skills are loaded from the `@openhands/extensions` npm package at build time via `SKILLS_CATALOG` (exported from `@openhands/extensions/skills`). The frontend's `SkillsService` maps catalog entries to `SkillInfo` objects and merges them with user/project skills fetched from the agent-server (with `load_public: false`). Bundled catalog skills use the persisted `enabled_skills` allow-list (defaulted from `DEFAULT_ENABLED_SKILL_NAMES`); user/project skills remain enabled unless named in `disabled_skills`. Keep this logic centralized in `src/utils/skill-enablement.ts`. The agent-server no longer clones the extensions repo or uses `EXTENSIONS_REF` for public skills.
- Default working-dir fallback is now the relative path `workspace/project` (exported as `DEFAULT_WORKING_DIR` from `src/api/agent-server-config.ts`); git-path heuristics and the default PLAN preview path should reuse that constant instead of hardcoding `/workspace/project`.
- Current Cloud behavior is implemented explicitly through the backend registry, Cloud service layer, and device authorization flow.
- Primary verification commands: `npm run lint`, `npm test`, `npm run build`, and `npm run build:lib`.
- GitHub automation now includes `.github/workflows/ci.yml` for `npm ci`, `npm test`, and `npm run build`, plus `.github/dependabot.yml` with weekly npm/github-actions updates gated by a 7-day cooldown.

## Repository Map — what belongs where

This repo (`OpenHands/OpenHands`) is **only the agent-canvas frontend**. It is one
piece of a multi-repo system. Before adding code here, check the change belongs in
*this* repo — several kinds of work belong in a sibling repo instead.

| Repo | Owns | Add code here when… |
|------|------|---------------------|
| **`OpenHands/OpenHands`** (this repo) | The React/TypeScript **frontend** (agent-canvas): UI, routes, frontend services in `src/api/` that *consume* backend APIs. | You are changing UI, frontend state, or how the frontend *calls* an existing backend endpoint. |
| **`OpenHands/software-agent-sdk`** | The Python **SDK + agent-server**: agents, tools, conversations, events, and the REST/WebSocket **API surface** (`openhands-sdk`, `openhands-tools`, `openhands-agent-server`, `openhands-workspace`). | You are adding or changing a backend endpoint, agent/tool behaviour, or server-side logic. New API **endpoints** live here, not in the frontend. |
| **`OpenHands/typescript-client`** (`@openhands/typescript-client`) | The generated/maintained **TypeScript client** that mirrors the agent-server API. The frontend's *only* sanctioned way to reach the agent-server (see "API Access Rules"). | You are adding client-side **access to an agent-server endpoint** (typed client method, request/response types). API-access code belongs here, **not** re-implemented in this repo. |
| **`OpenHands/extensions`** (`@openhands/extensions`) | Public **skills, automations, and integrations** (loaded here at build time via `SKILLS_CATALOG`). | You are adding or editing a skill, automation, or MCP integration. |

Common mis-placements to avoid:

- **API endpoint access** → belongs in `typescript-client`, then consumed here. Do **not**
  add raw `axios`/`fetch` endpoint code to the frontend (CI guard:
  `src/api/no-direct-agent-server-calls.test.ts`; see "API Access Rules").
- **New server endpoints / agent or tool logic** → belongs in `software-agent-sdk`.
- **Skills / automations / integrations** → belong in `extensions`.

## Cross-Repository Boundaries

The four repositories have distinct ownership boundaries:

| Repository | Owns |
|---|---|
| [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands) | Agent Canvas frontend, user-facing control center, backend selection, and local-stack orchestration. |
| [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) | Python SDK, Agent Server, agent/tool behavior, conversations, workspaces, events, and the canonical server API. |
| [`OpenHands/typescript-client`](https://github.com/OpenHands/typescript-client) | Browser-compatible TypeScript client and generated/maintained types for the Agent Server API. |
| [`OpenHands/automation`](https://github.com/OpenHands/automation) | Automation definitions, scheduling, webhooks, run history, and dispatching. It manages when automations run; the Agent Server/SDK executes them. |

The usual dependency direction is `software-agent-sdk` / Agent Server → OpenAPI contract → `typescript-client` → Agent Canvas. Automation scheduling and dispatching flow from Agent Canvas to `automation`, which starts work on the Agent Server/SDK. Put new server behavior and endpoints in `software-agent-sdk`, client access in `typescript-client`, UI and frontend integration in this repository, and scheduling/webhook lifecycle behavior in `automation`.

All pull requests for this repository must comply with [`.agents/skills/custom-codereview-guide.md`](.agents/skills/custom-codereview-guide.md), in addition to the general contribution requirements and CI checks.


## PR Description Human Check

The `HUMAN:` section in PR descriptions is reserved for human contributors only.
AI agents MUST NOT add to, edit, move, or remove it. If the PR description
CI fails because the section is missing or empty, stop and ask the
human user to update it in their own words. If the section was already updated
by a human, report the exact validator error rather than editing it yourself.

## Tracking / Analytics Architecture

One Canvas-owned PostHog client owns telemetry and app analytics.

- `src/services/telemetry.ts` is the only module that accesses the named `agent-canvas` PostHog client. The name isolates Canvas identity, persistence, configuration, and consent from an embedding host's default singleton. React code declares Cloud user identity and event context through the service and captures through the service; it never receives, identifies, or resets the SDK client directly.
- `TelemetryProvider` configures bootstrap/runtime options, eagerly initializes the service, and is the sole owner of the `useTelemetry()` lifecycle that emits install/session events. Do not mount that lifecycle hook separately in Canvas routes or internal components. The provider does not expose PostHog context or maintain a second client lifecycle.
- The default PostHog key and direct ingestion host live in `config/defaults.json` under `telemetry`. Local launchers (`dev-with-automation`, `dev-static`, published binary path) and Docker default `AUTOMATION_POSTHOG_API_KEY` from explicit automation env, then `VITE_POSTHOG_API_KEY`, then that shared default key, so the automation backend can emit local consent-gated telemetry without extra user config. Keep `VITE_DO_NOT_TRACK=1` disabling the zero-config default.
- Unconfigured source builds use the staging key and route through `https://z.openhands.dev`. Release workflows pass the public production key through `VITE_POSTHOG_API_KEY`. Precompiled npm consumers override `apiKey`, `apiHost`, and `uiHost` at runtime through `AgentServerUIProviders.analytics` or `configureTelemetry()`.
- `setTelemetryConsent` is the only user-consent controller; `configureTelemetry(false)` is the embedding host's hard disable. An explicit first-run browser decision remains pending across local backends until `useSyncTelemetryConsent` persists it to Cloud; a stale/default backend value must not overwrite that newer choice during login or navigation. Once Cloud confirms the choice, backend `user_consents_to_analytics` changes are authoritative and mirrored to the client. No other hook or component should call `opt_in_capturing` / `opt_out_capturing` directly.
- `subscribeTelemetryConsent` is the sole React-facing consent store. Hooks that render consent state must use `useSyncExternalStore`; do not mirror consent in component state or gate events outside `telemetry.ts`.
- `canvas_install` fires once, pre-consent, with the client's anonymous distinct ID. After consent and Cloud authentication, Canvas identifies PostHog with the stable Cloud user ID so PostHog joins the earlier anonymous activity to that person. Merely switching to a local backend clears Cloud event context without resetting the identified person; a resolved logout/account change, consent revocation, or privacy clear owns the reset. Local-only and never-authenticated traffic remains on the anonymous browser/install ID. Cloud account context (`cloud_user_id`, `cloud_user_email`, `cloud_org_id`) is attached as event properties only while a Cloud backend is active.
- `telemetry.ts` adds immutable `client_source`, `client_version`, `package_name`, and `package_version` properties in `before_send`, so reset cannot remove attribution and event producers cannot override it. Repeated business milestones use deterministic PostHog `$insert_id` values instead of process-local caches.
- `trackEvent` and `useTelemetry` remain the public library telemetry API for npm consumers (the `TelemetryConsentBanner` component was removed; hosts needing a consent UI build their own on `useTelemetry`). Non-React state machines use typed functions in `cloud-funnel-analytics.ts`; they do not call `trackEvent` directly.
- React app events use typed functions in `src/hooks/use-tracking.ts`; components never call `posthog.capture()` raw. The hook attaches `current_url` automatically and captures through the telemetry service; Cloud account email is attached centrally as `cloud_user_email` while Cloud context is active. It may read backend settings for event properties, but must never gate capture on a settings snapshot: `useSyncTelemetryConsent` has already mirrored the authoritative decision to the telemetry service, and settings can be stale during a backend transition.
- A business milestone has one canonical event capture. Do not conditionally switch between telemetry and app clients or emit duplicate events.

### Cloud funnel observability
- OAuth device authorization and Cloud conversation-start requests include the coarse `X-OpenHands-Client: agent_canvas` and `X-OpenHands-Client-Version` headers from `src/api/client-source.ts`. Never put device codes, API keys, conversation content, raw hosts, or other user data in these headers.
- Production ingress must retain those two headers as structured Datadog facets before source-specific operational queries will work.
- The consented OSS funnel uses typed `cloud_device_authorization_started`, `cloud_device_authorization_succeeded`, and `cloud_conversation_ready` events from `cloud-funnel-analytics.ts`; React emits the canonical `backend_added` event through `useTracking`.

### Adding a new event
1. Add a typed function to `useTracking` in `src/hooks/use-tracking.ts`
2. Add the function to the hook's `return` object
3. Destructure and call it from the component: `const { trackFoo } = useTracking()`

### Event dictionary: onboarding_link_clicked

One stable event for every onboarding link/CTA click. New onboarding links must
reuse this contract (extend the unions in `use-tracking.ts`), never add one-off
events per destination.

Properties (all values controlled enums or booleans — never raw destination
URLs, query params, or link text; `current_url` is the standard app-page common
property, not a destination):
- `link_id` (`OnboardingLinkId`): `configure_llm` | `start_conversation` |
  `schedule_task` | `customize_agent` | `connect_mcp` | `join_slack` |
  `open_docs`
- `destination_type` (`OnboardingLinkDestinationType`): `community` |
  `integration` | `documentation` | `settings` | `conversation` | `automation`
- `surface` (`OnboardingLinkSurface`): `landing_checklist` |
  `onboarding_modal` (reserved; no modal links are instrumented yet)
- `checklist_item` (optional): the owning checklist item's `link_id`; set on
  every `landing_checklist` emission, including `open_docs` clicks
- `step_id` (optional): reserved for future onboarding-modal links
- `is_external` (boolean): whether the destination leaves the app

Instrumented CTAs (sidebar "Getting started" checklist; the row link and its
preview action CTA intentionally share one `link_id` — same destination):

| Checklist item | Row + preview action | Preview docs link |
|---|---|---|
| Add LLM API key | `configure_llm` / `settings` / internal | `open_docs` / `documentation` / external |
| Start your first chat | `start_conversation` / `conversation` / internal | `open_docs` |
| Schedule a task | `schedule_task` / `automation` / internal | `open_docs` |
| Customize your agent | `customize_agent` / `settings` / internal | `open_docs` |
| Connect an MCP integration | `connect_mcp` / `integration` / internal | `open_docs` |
| Join the OpenHands Slack | `join_slack` / `community` / external | `open_docs` |

Excluded CTAs (per the one-canonical-capture rule above):
- Onboarding-modal wizard controls (back/next/skip/close, agent cards) →
  covered by `onboarding_step_viewed` / `onboarding_completed` /
  `onboarding_skipped`
- Modal backend-connect CTAs and the backend form's docs links →
  `backend_added` with `source: "onboarding"`
- LLM settings help links inside the embedded settings screen (shared with
  non-onboarding surfaces) → setup outcome captured by `settings_saved`
- Recommended-automation cards → `prebuilt_automation_enabled`
- Checklist expand/collapse toggle and the settings visibility switch → UI
  state, not destination links

Known limitation: middle-click (`auxclick`) opens are not captured; tracking
uses React `onClick` only and never prevents default navigation.

### Env vars
`VITE_POSTHOG_API_KEY` is the sole build-time PostHog key. Unconfigured source builds use staging; official release workflows set production explicitly. Precompiled consumers use runtime configuration instead.

## Runtime Services in Dev Stacks

- When the agent-canvas dev launchers (`npm run dev` / `dev:static` / the published `agent-canvas` binary) start a stack with ingress/static-server, the backend-facing server appends runtime service metadata to `/server_info` as the optional `runtime_services` field. The frontend reads that backend-provided value when creating conversations and forwards it as `AgentContext.system_message_suffix` on `POST /api/conversations`, so conversations land with a `<RUNTIME_SERVICES>` block appended to the system prompt.
- The block lists URLs **from the agent's point of view**:
  - The Agent Server is always reachable as `http://localhost:<port>` from inside the sandbox — but that is _you_, not the automation backend.
  - Host-side services (ingress, Vite, automation) are reachable as `http://localhost:<port>`.
- Agents should treat the `<RUNTIME_SERVICES>` block as authoritative: don't hardcode `localhost:8000` for "the automation server", and don't probe random ports trying to discover services. If the block says automation is not running, skip `/api/automation` calls; otherwise use the listed `url_from_agent` + `api_prefix` (default `/api/automation`) and the `X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY` header.
- The launcher → backend → frontend → suffix plumbing is:
  - `scripts/runtime-services-info.mjs::buildRuntimeServicesInfo()` — dependency-free module that constructs the info object; also runs as a CLI for the Docker entrypoint. Re-exported by `scripts/dev-safe.mjs` for backward compat.
  - `scripts/dev-with-automation.mjs::buildAutomationRuntimeServicesInfo()` — wraps it with automation details. `dev-with-automation`, `dev-static`, and the published binary pass the JSON to `scripts/ingress.mjs` or `scripts/static-server.mjs` via `--runtime-services-info`.
  - `scripts/ingress.mjs` and `scripts/static-server.mjs` proxy the real agent-server `/server_info` response and append `runtime_services` when configured. This keeps version/tool compatibility fields authoritative from the SDK while letting the Agent Canvas stack advertise automation/frontend/ingress topology.
  - `src/api/agent-server-adapter.ts::fetchBackendRuntimeServicesInfo()` reads `runtime_services` from cached or freshly fetched `/server_info`; `buildRuntimeServicesSystemSuffix()` renders the `<RUNTIME_SERVICES>` markdown block; `buildAgentContext()` attaches it to `agent_context.system_message_suffix` when present.
  - E2E coverage: the mock-LLM automation test (`tests/e2e/mock-llm/automations/mock-llm-automation.spec.ts`) verifies the `<RUNTIME_SERVICES>` block reaches the LLM via `getMockLLMRequests()` and checks for Agent Server, Automation backend, and `/api/automation` entries.

### `/server_info.runtime_services` shape

The `runtime_services` value is a JSON object of:

```json
{
  "mode": "dev:automation",
  "services": {
    "agent_server": {
      "description": "The OpenHands Agent Server this agent is running inside. ...",
      "url_from_agent": "http://localhost:18000"
    },
    "ingress": {
      "description": "Unified entry point. Routes /api/automation/* ...",
      "url_from_agent": "http://localhost:8000"
    },
    "frontend": {
      "kind": "vite",
      "description": "Vite dev server hosting the agent-canvas frontend.",
      "url_from_agent": "http://localhost:3001"
    },
    "automation": {
      "description": "OpenHands Automations service. All routes are mounted under '/api/automation'. Authenticate with header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'.",
      "url_from_agent": "http://localhost:18001",
      "api_prefix": "/api/automation",
      "docs_url": "http://localhost:18001/api/automation/docs",
      "openapi_url": "http://localhost:18001/api/automation/openapi.json",
      "auth_env_var": "OPENHANDS_AUTOMATION_API_KEY"
    }
  }
}
```

All keys under `services` are optional and omitted when the corresponding service isn't running. `frontend.kind` is `"vite"` for dev launchers running the Vite dev server and `"static"` for stacks serving a pre-built `build/` directory (`dev:static`, the published `agent-canvas` binary).

### Example `<RUNTIME_SERVICES>` block (dev with automation)

```
<RUNTIME_SERVICES>
You are running inside an agent-canvas dev stack started in 'dev:automation' mode.
The following services are reachable from your sandbox. URLs are written
from your point of view (i.e., as you should curl/fetch them).

* Agent Server (you): http://localhost:18000
    The OpenHands Agent Server this agent is running inside. Tool calls (terminal, file_editor, browser, etc.) execute here.
* Ingress: http://localhost:8000
    Unified entry point. Routes /api/automation/* to the automation backend, /api/* and /sockets to the agent-server, and /* to the frontend.
* Frontend: http://localhost:3001
    Vite dev server hosting the agent-canvas frontend.
* Automation backend: http://localhost:18001
    OpenHands Automations service. All routes are mounted under '/api/automation'. Authenticate with header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'.
    Docs:    http://localhost:18001/api/automation/docs
    OpenAPI: http://localhost:18001/api/automation/openapi.json
    Auth:    header 'X-Session-API-Key: $OPENHANDS_AUTOMATION_API_KEY'

Trust this block over guessing: do not assume any other URLs are running.
In particular, http://localhost:18000 inside your sandbox is the Agent Server
you are running inside of — NOT the automation backend.
</RUNTIME_SERVICES>
```

## Live End-to-End Test Framework

- The live QA path is intentionally separate from ordinary mocked Playwright coverage. If ordinary browser tests are added, keep them outside `tests/e2e/live/` so `playwright.config.ts` can run them while ignoring `**/live/**`; live LLM-backed tests must never run as part of `npm run test:e2e`.
- Live tests live under `tests/e2e/live/` and are run only through `npm run test:e2e:live`, which uses `playwright.live.config.ts`. Keep the spec names descriptive; the primary conversation smoke test is `tests/e2e/live/real-agent-server-conversation.spec.ts`.
- `npm run test:e2e:live` loads `.env` through Node's `--env-file-if-exists` flag and invokes `tests/e2e/live/scripts/run-live-e2e.mjs`. The runner validates the required local environment, explains missing credentials/prerequisites, and then runs `playwright test --config=playwright.live.config.ts`. Use `npm run test:e2e:live -- --check` to validate local setup without running the test, and pass Playwright flags after `--` (for example `npm run test:e2e:live -- --headed`).
- Local live E2E requires one LLM credential: `LIVE_E2E_LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLM_API_KEY`. Optional overrides are `LIVE_E2E_LLM_BASE_URL`, `LIVE_E2E_LLM_MODEL`, `LIVE_E2E_SESSION_API_KEY`, `LIVE_E2E_BACKEND_URL`, and `LIVE_E2E_FRONTEND_PORT`. The local runner prints which variables are missing without printing secret values.
- Live-test-only helpers belong under `tests/e2e/live/utils/`. The current helper module is `tests/e2e/live/utils/agent-server-conversation.ts`; do not put live-only helpers in the shared `tests/e2e/support/` directory.
- `playwright.live.config.ts` starts the real local Agent Server/UI stack via `npm run dev:minimal`, not MSW mocks. It uses `LIVE_E2E_SESSION_API_KEY` when set, otherwise generates a per-run random session key and passes it through `SESSION_API_KEY`, `OH_SESSION_API_KEYS_0`, and `VITE_SESSION_API_KEY`; specs that need direct backend requests must inject `X-Session-API-Key` only for the configured backend origin through `routeBackendSessionApiKey(page)`, never through global Playwright `extraHTTPHeaders`. Live tests default to frontend port `3101` and Agent Server `http://127.0.0.1:18100` so they do not accidentally reuse a normal local dev stack.
- `tests/e2e/live/utils/agent-server-conversation.ts` configures the running Agent Server before each live conversation by PATCHing `${LIVE_E2E_BACKEND_URL ?? "http://127.0.0.1:18100"}/api/settings` with LLM settings and low-risk conversation settings. LLM credentials are read from `LIVE_E2E_LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LLM_API_KEY`; CI defaults use `LIVE_E2E_LLM_BASE_URL` (default `https://llm-proxy.app.all-hands.dev`) and `LIVE_E2E_LLM_MODEL` (default `openhands/claude-haiku-4-5-20251001`).
- The live conversation test should stay cheap and as deterministic as possible while still exercising one real tool call: it asks the model to run the exact `EXPECTED_BASH_COMMAND`, waits for the bash output token to appear outside the user's message in the UI, confirms a successful `ExecuteBashObservation`/`TerminalObservation` through the real Agent Server events API, and then waits for the final `EXPECTED_REPLY_TOKEN`. This exercises the real UI, Agent Server settings API, conversation creation, websocket/event path, terminal tool execution, and LLM response path. Because LLM behavior is not perfectly deterministic even at temperature 0, CI keeps one retry for live E2E; future live tests should document any expected variance and avoid prompts that require unnecessary formatting obedience.
- Live E2E must not pollute analytics. `playwright.live.config.ts` starts the app with `VITE_DO_NOT_TRACK=1`; the live helper seeds local storage with telemetry/analytics opt-out values before app code runs; and each live spec should install `guardAgainstPostHogRequests(page)` before navigation so any attempted request to `*.posthog.com` or `z.openhands.dev` is blocked locally and fails the test.
- Live Playwright videos are intentionally recorded for CI QA debugging when `LIVE_E2E_RECORD_VIDEO=on` is set; local default video mode is `retain-on-failure`. Do not add live tests that render API keys, tokens, secret values, or credential-bearing error messages in the browser. Screenshots should target a safe app/chat region such as `data-testid="chat-interface"` instead of `page.screenshot({ fullPage: true })`, and should apply `getLiveArtifactMask(page)` for text/field redaction; if a future live test must exercise sensitive UI, change that test/media path to redact the sensitive output or retain video only on failure.
- `.github/workflows/ci.yml` runs live E2E automatically after pushes reach `main`, not on pull-request events. Manual `workflow_dispatch` with a required `pr_number` remains available for pre-merge QA. Main runs test the trusted pushed commit and publish their report and media only as the workflow summary and GitHub Actions artifact. Manual PR runs retain the PR comment/media flow and must skip fork PRs before checking out PR code so LLM credentials and artifact-push tokens are never exposed to untrusted code.
- Keep live E2E secrets out of job-level `env`. The workflow should check whether credentials exist before checkout, but inject the LLM key only into the trusted step that actually runs the live test.
- The live job uploads the Playwright HTML report plus screenshot/video output as a GitHub Actions artifact, and also extracts the primary screenshot/video attachments. It converts the WebM recording to a GIF preview with `ffmpeg` so GitHub PR comments can inline the preview. Keep Playwright trace capture disabled for live tests because the setup flow sends LLM credentials to the Agent Server settings API, and traces can record request bodies. Failure messages around live Agent Server settings must not print response bodies from credential-bearing requests.
- Inline PR-comment media is stored as PR-only files under `.pr/live-e2e/<github_run_id>/` on the PR branch, not on a long-lived orphan media branch. The comment uses `raw.githubusercontent.com/<repo>/<artifact_commit>/.pr/live-e2e/...` URLs for the GIF and PNG so GitHub can render them inline. The WebM is linked as the full recording because GitHub comments do not reliably inline WebM.
- `.github/workflows/pr-artifacts.yml` owns cleanup for `.pr/live-e2e/`: it comments when `.pr/` artifacts exist, removes them after PR approval for same-repo PRs, and opens or updates a cleanup PR against `main` if artifacts reach `main` through a fork PR or a missed approval cleanup.
- The live reporting scripts live beside the live tests under `tests/e2e/live/scripts/`: `run-live-e2e.mjs`, `extract-live-e2e-media.mjs`, `render-live-e2e-report.mjs`, and `upsert-pr-comment.mjs`. Keep report/comment/local-runner logic there rather than in top-level `scripts/`, because these scripts are part of the live E2E framework.
- When changing any part of this framework — live workflow triggers, artifact publishing, `.pr` cleanup, live Playwright config, live test file layout, helper locations, local runner behavior, or report/comment scripts — update this `AGENTS.md` section in the same PR so future agents have the current operating model.

## Mock-LLM E2E Test Framework

- Mock-LLM tests live under `tests/e2e/mock-llm/` and exercise the complete stack — from the browser through the real agent-server to a scripted mock LLM server — without any real LLM credentials. Run locally with `npm run test:e2e:mock-llm`.
- **Production-fidelity launch**: The Playwright config (`playwright.mock-llm.config.ts`) starts the full `agent-canvas` stack via `bin/agent-canvas.mjs` — the same binary that `npx @openhands/agent-canvas` executes when users install the npm package. This means mock-LLM tests exercise the actual production path: pre-built static frontend + static-server.mjs + agent-server via uvx + automation backend via uvx + ingress proxy, all behind a single port.
- A pre-built `build/` directory is required. The Playwright webServer command runs `npm run build:app` when `build/index.html` is absent, but CI should run the build step explicitly for caching (`npm run build:app` in `.github/workflows/mock-llm-e2e.yml`).
- **Single ingress URL**: Tests use one URL for both the browser (`baseURL`) and backend API assertions (`BACKEND_URL`). The ingress proxy routes `/api/*` to the agent-server, `/api/automation/*` to the automation backend, and `/*` to the static frontend. Default ingress port for tests is `18300` (override via `MOCK_LLM_INGRESS_PORT` env var).
- **State isolation**: `OH_CANVAS_SAFE_STATE_DIR=.tmp/mock-llm-state` isolates test state from the user's real `~/.openhands/agent-canvas/` directory. Both `STATE_DIR` (`.tmp/mock-llm-state`) and the automation DB dir (`.tmp/automation/`) are cleaned before each test run — the automation DB now lives outside STATE_DIR at `dirname(STATE_DIR)/automation/automations.db`, mirroring Docker's `~/.openhands/automation/automations.db`.
- **Session API key**: A random key is generated per test run and passed to the stack via `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0` / `VITE_SESSION_API_KEY`. The static server injects it into `index.html` at serve time so the frontend authenticates automatically.
- **Mock LLM server** (`tests/e2e/mock-llm/scripts/mock-llm-server.py`): Python HTTP server using openhands-sdk's `TestLLM` to return scripted tool-call + text trajectories. Supports admin API endpoints for dynamic trajectory management:
  - `POST /admin/reset` — reset to the default trajectory (terminal printf + text reply); also clears the stored completion-request history
  - `POST /admin/trajectory/register` — register a named trajectory (JSON body: `{name, turns}` where each turn is `{tool_call: {name, arguments}}` or `{text: "..."}`)
  - `POST /admin/trajectory/activate` — activate a previously registered trajectory
  - `GET /admin/requests` — return the list of all `/v1/chat/completions` request bodies captured since the last reset (used by the image-upload test to verify the image was forwarded to the LLM)
  - Profile pre-flight ping: agent-server ≥ 1.43 sends a 1-token `ping` completion whenever an LLM profile is saved (`POST /api/profiles/{name}/validate`, 30 s budget in the canvas). The mock answers it with a canned `pong` instead of feeding it to `TestLLM`, so it neither consumes a scripted turn nor 500s-and-retries past the canvas timeout when the trajectory is exhausted; it is also left out of the `/admin/requests` history.
- **Real automation backend**: The automation test uses the production automation backend (started by `bin/agent-canvas.mjs`), NOT a mock server. Terminal `curl` commands from the agent hit the automation API through the ingress proxy at the test's `BACKEND_URL` (default `http://localhost:18300`). Auth uses the `X-Session-API-Key` header matching the stack's session key.
- **Test helpers** (`tests/e2e/mock-llm/utils/mock-llm-helpers.ts`): Exports `registerTrajectory()`, `activateTrajectory()`, `resetMockLLM()`, `ensureMockLLMProfile()`, `getMockLLMRequests()` (fetches captured completion bodies from `GET /admin/requests`), `IMAGE_REPLY_TOKEN` + `MINIMAL_PNG_BASE64` (constants for the image-upload spec), ACP helpers (`configureAcpAgent()`, `verifyAcpAgentSettings()`, `resetToOpenHandsAgent()`, `ACP_REPLY_TOKEN`, `MOCK_ACP_SERVER_PATH`), and more.
- **Padding response for internal LLM call**: The agent-server makes an internal LLM call (condenser/skill-analysis) before the agent's main loop starts when skills are activated. This consumes one trajectory response. Automation tests prepend a throwaway `{ text: "" }` response as padding. The conversation test does NOT need this because its user message doesn't trigger skill activation. Since `openhands-automation==1.10.0` (openhands/automation#405), the automation lifecycle spec scripts the required `finish` tool turn in its run-conversation budget — that release requires preset automation conversations to callthe `finish` tool before the run reaches COMPLETED; blank turns alone would loop and exhaust the trajectory. The mock server logs "Mock LLM exhausted after N calls" pinpoint the exact count if it drifts again.
- **Mock ACP server** (`tests/e2e/mock-llm/scripts/mock-acp-server.py`): A minimal stdio-based ACP agent that speaks JSON-RPC using the `acp` Python library (installed as a dependency of `openhands-sdk`). Handles `initialize`, `session/new`, and `session/prompt`; sends a scripted `session/update` notification with `ACP_REPLY_TOKEN` in a text content block, then returns `stop_reason: "end_turn"`. The agent-server spawns it as a subprocess via `acp_command`. Accepts `--reply-token TOKEN` to customize the reply token.
- **Test directory layout**: Specs are organized into feature subdirectories that mirror the source code structure, enabling selective test execution based on which source files changed:
  - `settings/` — LLM profile management, ACP agent config, model switching (`mock-llm-acp-agent.spec.ts`, `mock-llm-profile-management.spec.ts`, `mock-llm-model-switch.spec.ts`)
  - `conversations/` — Core conversation flow, image upload (`mock-llm-conversation.spec.ts`, `mock-llm-image-upload.spec.ts`)
  - `files/` — Files tab, Browser tab, and git control bar coverage (`mock-llm-files-and-git.spec.ts`)
  - `automations/` — Automation lifecycle, preset cards (`mock-llm-automation.spec.ts`, `mock-llm-preset-automation.spec.ts`)
  - `onboarding/` — First-run onboarding flow (`mock-llm-onboarding-happy-path.spec.ts`, `mock-llm-onboarding-regressions.spec.ts`)
  - `backends/` — Auth modes, cross-connect, partial stack (`mock-llm-auth-modes.spec.ts`, `mock-llm-cross-connect.spec.ts`, `mock-llm-partial-stack.spec.ts`)
  - `home/` — Workspace selection, folder browser (`mock-llm-folder-workspace.spec.ts`)
  - `mcp/` — MCP marketplace/server management and credential verification (`mock-llm-mcp-github.spec.ts`, `mock-llm-mcp-slack-credentials.spec.ts`)
  - `skills/` — Skill loading and activation (`mock-llm-skills.spec.ts`)
  - `canvas-extensions/` — Canvas Extension install → enable → page render → disable → uninstall lifecycle (`mock-llm-canvas-extensions.spec.ts`). The pinned agent-server predates `/api/canvas-extensions`, so the spec serves that contract from `src/fixtures/canvas-extensions/demo-page` via `page.route()`; delete the stub once the pin ships the endpoints and install the fixture by absolute path instead.
  - `regressions/` — CSS isolation, event pagination, workspace persistence (`mock-llm-ui-regressions.spec.ts`). Always included in selective runs.
- **Selective test resolver**: `tests/e2e/mock-llm/test-mapping.json` maps source paths to test subdirectories, and `tests/e2e/mock-llm/scripts/resolve-affected-tests.mjs` can resolve a changed-file list for local investigation. Post-merge CI and `workflow_dispatch` intentionally run the full suite, so this resolver is not part of the workflow path.
- Tests run serially (`workers: 1`, `mode: "serial"` per describe block). Each spec is self-contained (configures its own LLM profile, resets mock LLM in `afterEach`). The `afterEach` hook resets the mock LLM to its default trajectory so subsequent specs start fresh even when a preceding test fails.
- CI workflow: `.github/workflows/mock-llm-e2e.yml` runs the full suite after pushes reach `main` and on manual dispatch; it does not run on pull-request events. The workflow builds the frontend, starts the mock LLM server, runs the tests, uploads artifacts, and writes the rendered report to the workflow summary. Npm-path Mock-LLM CI uses `MOCK_LLM_GLOBAL_TIMEOUT_MS` (default 20 min) and the workflow deadline adds a 60-second teardown buffer; keep `playwright.mock-llm.config.ts` and `.github/workflows/mock-llm-e2e.yml` in sync if that timeout changes.
- The custom `DoneMarkerReporter` writes `.mock-llm-markers/.tests-done` after all tests complete (before webServer teardown) so the CI wrapper can detect completion and kill the lingering teardown process.

### Docker Image Testing (Shared Specs)

- The same test specs and helpers are reused to validate the Docker image via `playwright.mock-llm-docker.config.ts`. Run locally with `npm run test:e2e:mock-llm:docker` (requires Docker daemon and a built image).
- **Architecture**: The Docker config replaces the npm path's `bin/agent-canvas.mjs` webServer with a `docker run --network host` command. The mock LLM server still runs on the host. On Linux (including CI), `--network host` lets the container share the host's network stack so all `127.0.0.1` URLs work identically. On macOS/Windows Docker Desktop (bridge networking), set `MOCK_LLM_AGENT_URL=http://host.docker.internal:<port>` so the agent-server inside Docker can reach the host-side mock LLM server.
- **Dual-stack binding**: Both `scripts/static-server.mjs` and `scripts/ingress.mjs` default to `::` (dual-stack, accepting IPv4 and IPv6 connections). The Docker entrypoint passes `--host ::` explicitly. This means `localhost` is safe in both the Docker and npm Playwright configs — whether it resolves to `127.0.0.1` (IPv4) or `::1` (IPv6), the server accepts the connection. The mock LLM server URL (`MOCK_LLM_URL`) still uses `127.0.0.1` because the Python mock server is a separate process whose bind behavior we don't control.
- **Entrypoint crash resilience**: `docker/entrypoint.sh` uses a `while kill -0 "$STATIC_PID"; do sleep 10 & wait $!; done` loop instead of `wait -n "${PIDS[@]}"` (any child). If the agent-server or automation backend exits mid-test, the static-server proxy stays up and returns 502s for backend routes — the container doesn't disappear with `ECONNREFUSED`. The container exits only when the static-server (ingress) dies or on SIGTERM/SIGINT. The `sleep & wait $!` pattern ensures `wait` (a bash builtin) is the foreground op, so trapped signals fire immediately. `cleanup()` includes `exit 0` so the script terminates after a signal-triggered trap return.
- **URL split**: `mock-llm-helpers.ts` exports two mock LLM URL constants:
  - `MOCK_LLM_BASE_URL` — always `http://127.0.0.1:<port>`, used by tests for the mock LLM admin API (register/activate/reset trajectories).
  - `MOCK_LLM_AGENT_URL` — defaults to `MOCK_LLM_BASE_URL`, overridable via `MOCK_LLM_AGENT_URL` env var. Used when configuring the LLM profile (`base_url` field) — this is the URL the agent-server uses for inference calls. The npm path and Docker-with-`--network host` path use the same value; Docker on macOS needs the override.
- **Docker image**: Set `MOCK_LLM_DOCKER_IMAGE` to the image tag (default: `ghcr.io/openhands/agent-canvas:latest`). The container is started with `--rm --network host` and a unique `--name` for cleanup.
- **State isolation**: The Docker container uses its internal state directory (no host mount needed for tests). Each test run starts a fresh container.
- **Skill test volume mounts**: Tests that create files the agent-server needs to read (skill repos, user skills) require Docker volume mounts because the container has an isolated filesystem. The Docker config mounts `.tmp/mock-llm-skill-repos/` → `/tmp/mock-llm-skill-repos/` for project skills and `.tmp/mock-llm-user-skills/` → `/home/openhands/.openhands/skills/` for user skills. Env vars `MOCK_LLM_SKILL_REPOS_CONTAINER_DIR` and `MOCK_LLM_USER_SKILLS_HOST_DIR` tell `skill-test-helpers.ts` which paths to use for agent-server API registration vs. host-side file operations.
- CI workflow: `.github/workflows/mock-llm-docker-e2e.yml` has two triggers, both using an already-built image from GHCR: (1) `workflow_run` fires automatically after a successful `Docker` workflow on `main`; (2) `workflow_dispatch` accepts a custom `docker_image` input. It does not run on pull-request events. The default image tag is derived from the tested commit SHA (`ghcr.io/openhands/agent-canvas:sha-<short>-amd64`). Report artifacts go to `test-results-mock-llm-docker/` and `playwright-report-mock-llm-docker/`.

## Debugging E2E Test Failures

When an E2E test fails in CI, use this workflow to diagnose the root cause efficiently:

### 1. Read the workflow summary first
The mock-LLM E2E workflows write a structured report to the GitHub Actions workflow summary with a test results table, pass/fail status, and collapsible failure details including the Playwright error message. **Start here** — the error message usually reveals whether the failure is a locator mismatch, a timeout, or a missing element.

### 2. Download CI artifacts
Every failing test run uploads artifacts (`mock-llm-e2e-results` for npm, `mock-llm-docker-e2e-results` for Docker). Download them with:
```bash
gh run download <run_id> --repo OpenHands/OpenHands --name mock-llm-e2e-results --dir /tmp/artifacts
```
Artifacts contain:
- `test-results-mock-llm/` — per-test directories with `test-failed-N.png` (screenshot at failure) and `error-context.md` (Playwright page snapshot as YAML accessibility tree + test source with the failing line marked)
- `playwright-report-mock-llm/` — full HTML report (`npx playwright show-report /tmp/artifacts/playwright-report-mock-llm`)

### 3. Inspect the error-context.md page snapshot
The `error-context.md` file contains a YAML accessibility tree of the entire page at the moment of failure. This is the single most useful artifact — it shows exactly what DOM elements exist, which tabs are selected, what text is in inputs, and whether a component rendered at all. Search for the element your test expects (e.g. `llm-provider-input`) to see if it's present or absent, and check surrounding context (tab selection state, form view mode, etc.) to understand why.

### 4. Common failure patterns

**"element(s) not found"** — The locator matched zero elements. The component either:
- Didn't render (conditional rendering path not taken — check the page snapshot for what DID render)
- Has a different `name`/`data-testid` than expected
- Is behind a lazy-load boundary that hasn't resolved

**Stale state from earlier serial tests** — Mock-LLM tests run serially (`workers: 1`) against a real agent-server. Earlier tests (conversation, automation) persist settings on the server. If your test depends on "clean" state but a prior test configured `llm_base_url`, `llm_model`, etc., the form may render in a different view mode. Use Playwright `page.route()` to intercept and normalize the settings response. Example: `routeOnboardingLlmCatalog` in `tests/e2e/support/onboarding-helpers.ts` intercepts `GET /api/settings` to clear `llm_base_url` so the LLM form always opens in "Basic" view.

**View mode mismatch (Basic vs Advanced)** — `LlmSettingsScreen` switches between "Basic" (renders `ModelSelector` with provider/model dropdowns) and "Advanced" (renders plain text inputs). The view is determined by `getInitialView()` which checks `currentSettings.llm_base_url` — a non-default base URL triggers "Advanced" view. If your test expects `input[name="llm-provider-input"]` but sees text inputs instead, the settings have a stale `base_url`.

**Playwright route interception vs real server** — In mock-LLM tests, routes registered with `page.route()` intercept at the browser level before requests reach the real agent-server. However, `page.route()` must be set up BEFORE `page.goto()`. The `showOnboarding` helper handles this correctly (routes are registered before navigation). Non-GET methods should use `route.fallback()` to pass through to the real server.

### 5. Running locally
```bash
npm run test:e2e:mock-llm                    # full suite
npm run test:e2e:mock-llm -- --headed        # watch in browser
npm run test:e2e:mock-llm -- -g "test name"  # run single test by name
```

## Testing Rules

<TESTING_RULES>
Create TDD tests for behavioral changes. Focus on user behavior and follow TDD best practices, including:

- AAA structure (Arrange, Act, Assert)
- Clear test focus
- Proper test data management

Before writing any test:

- Avoid duplicating test cases or logic
- Do not assert the same condition more than once
- Do not mock the hook. Instead, mock the underlying service that the hook depends on
- Prefer adding to or extending existing test files whenever possible. Create new test files only if no suitable ones exist
- Avoid brittle visual-presentation assertions. Functional CSS contracts such as style scoping and selector transformation may be tested directly
- Keep the number of test cases to the minimum necessary while still fully covering the intended changes and behaviors

Ensure each test is meaningful, concise, and covers a unique aspect of user interaction.
</TESTING_RULES>

## Additional Notes

- **Published binary auth fix**: When users install the npm package globally (`npm install -g @openhands/agent-canvas`) and run `agent-canvas`, the pre-built static frontend has NO `VITE_SESSION_API_KEY` baked in (npm publish runs `npm run build` with no such env var). The runtime session key is generated when the CLI launches and reaches the frontend via `scripts/static-server.mjs --session-api-key <key>`, which injects a `<head>` script that does two things: (a) sets `window.__AGENT_CANVAS_SESSION_API_KEY__ = <key>` — read by `getBakedSessionApiKey()` in `src/api/agent-server-config.ts` as a fallback when the env var is empty, symmetric with `__AGENT_CANVAS_AUTH_REQUIRED__` / `isAuthRequired()`; (b) writes the same key into `localStorage['openhands-agent-server-config'].sessionApiKey`, always overwriting when the value differs, so any code path that still reads the legacy storage key (e.g. e2e fixtures) sees the live key. The window-global path is the load-bearing one — without it, `makeDefaultLocalBackend()` returns null on a fresh install, the backend registry seeds empty, and `root.tsx` traps the user behind the Manage Backends modal instead of onboarding. `scripts/dev-with-automation.mjs` and `scripts/dev-static.mjs` both pass `--session-api-key ${config.sessionApiKey}` when starting the static server.

- Direct `dependencies` and `devDependencies` in `package.json` are exact-pinned (no caret ranges); reproducible installs should use the committed `package-lock.json` plus `npm ci`, and targeted transitive fixes still belong in `overrides`.
- Current `overrides` in `package.json` and the advisories they address (keep this list in sync when adding/removing overrides):
  - `@vercel/static-config > ajv: 8.20.0` — **GHSA-2g4f-4pwh-qvx6** (ReDoS in ajv's `$data` option, affects 7.0.0-alpha.0–8.17.1, reaches us through `@vercel/react-router` → `@vercel/static-config` → `ajv@8.6.3`). The override is intentionally **scoped** to `@vercel/static-config`: a top-level `ajv` override forces ajv 8.x everywhere and breaks ESLint's `@eslint/eslintrc`, which requires ajv ^6.x (incompatible API). Scoping lets ESLint keep its nested `ajv@6.x` while only the vulnerable consumer is bumped.
  - `dompurify: 3.4.14` — keeps Monaco's exact-pinned `dompurify@3.2.7` on a patched release for **GHSA-55q2-fjhq-7xh7**, **GHSA-39q2-94rc-95cp**, and related XSS bypasses.
  - `typed-rest-client > qs: 6.15.3` — overrides the vulnerable `qs@6.15.1` exact-pinned by every published `typed-rest-client` release in the `@stryker-mutator/core` dependency tree.
- When bumping pinned versions, use npm to update `package.json` and `package-lock.json` together; do not hand-edit only one of them.
- `npm test` now runs `npm run make-i18n` first so clean environments generate `src/i18n/declaration.ts` before Vitest loads aliased imports.
- `__tests__/vite-config.test.ts` should import `vite.config` directly under `// @vitest-environment node`; spawning plain `node -e 'import ./vite.config.ts'` is not portable across Node patch releases in CI.
- `vitest.setup.ts` must guard DOM-specific globals (`HTMLCanvasElement`, `HTMLElement`, `window`) because some suites run in the Node environment instead of jsdom.
- WebSocket hook regression note: `__tests__/hooks/use-websocket.test.ts`'s `onClose` callback assertion was flaky against the shared MSW websocket server in CI; keep that single test on a deterministic stubbed `WebSocket` close path instead of relying on MSW close timing.
- Library i18n regression note: `__tests__/i18n/library-namespace.test.ts` imports `../../src/index`, which can take >5s under the full Vitest suite after `vi.resetModules()`. Keep an explicit per-test timeout (currently 15s) so the suite doesn't fail on slow workers.

- `src/components/shared/buttons/styled-tooltip.tsx` should keep HeroUI tooltip animations disabled in Vitest (`disableAnimation` when `import.meta.env.MODE === "test"`); otherwise full-suite runs can end with unhandled `window is not defined` rejections from `framer-motion` after jsdom teardown (seen via `recent-conversation` tests in CI).
- `@openhands/typescript-client` should be pinned to a released npm version rather than an unreleased commit SHA; when agent-canvas needs new client API, release the client first and then update the dependency. Released versions should include the typed clients, agent-server version compatibility helpers, `WorkspacesClient`, `ConversationClient.switchLLM`, and subpath exports for `events/remote-events-list` and `workspace/remote-workspace` needed by the agent-canvas agent-server integration. `RemoteWorkspace.gitChanges`/`gitDiff` accept an optional `{ ref }` option; agent-canvas passes `'HEAD'` so the changes panel reflects working-tree + index versus the latest commit (i.e. staged + unstaged) instead of a diff against the upstream/default branch.
- If a GitHub-hosted git dependency is introduced, npm may normalize its lockfile URL to SSH. `scripts/vercel-install.sh` (wired through `vercel.json`) rewrites GitHub SSH URLs to HTTPS before `npm ci`; keep that generic protection in sync with any future git dependencies.

## API Access Rules

Two strict conventions govern every REST call in the frontend. Violations break CI
via `src/api/no-direct-agent-server-calls.test.ts`.

### Rule 1 -- Agent-server calls must use `@openhands/typescript-client`

All calls that target the local agent-server (`/api/*`, `/server_info`, `/sockets`)
**must** go through typed client classes from `@openhands/typescript-client`, **never**
raw `axios`, `fetch`, or the legacy shared `openHands` axios instance.

Available clients and their subpath imports:

- `ConversationClient` -- `@openhands/typescript-client/clients`
- `FileClient` -- `@openhands/typescript-client/clients`
- `VSCodeClient` -- `@openhands/typescript-client/clients`
- `ServerClient` -- `@openhands/typescript-client/clients`
- `RemoteWorkspace` -- `@openhands/typescript-client/workspace/remote-workspace`
- `RemoteEventsList` -- `@openhands/typescript-client/events/remote-events-list`

Client options are always assembled via helpers in `src/api/agent-server-client-options.ts`:

- `getAgentServerClientOptions(overrides?)` -- for SDK client constructors
- `getAgentServerHttpClientOptions(overrides?)` -- for typed wrappers such as `RemoteEventsList`; application code must not import or construct the low-level `HttpClient`

These helpers read host, session API key, and working directory from the active backend
registry and env config, so callers never hardcode URLs or auth tokens.

```ts
// CORRECT
const data = await new ConversationClient(
  getAgentServerClientOptions(),
).getConversation(id);
const file = await new FileClient(
  getAgentServerClientOptions(),
).downloadTextFile(path);

// WRONG -- raw axios/fetch calls fail the no-direct-agent-server-calls.test.ts guard
const data = await axios.get(`${host}/api/conversations/${id}`);
const data = await fetch(`/api/conversations/${id}`);
```

**Allowed exceptions** (files that may use axios directly for infrastructure reasons):

- `src/api/automation-service/automation-service.api.ts`
- `src/api/cloud/proxy.ts` -- the proxy envelope POST itself
- `src/api/main-app-auth.ts` -- the local main-app authentication endpoint

### Rule 2 -- Cloud backend routes must go through `callCloudProxy`

Any call from the browser to the cloud backend (`app.all-hands.dev`) or a cloud
runtime sandbox (`*.prod-runtime.all-hands.dev`) **must** go through `callCloudProxy()`
in `src/api/cloud/proxy.ts`. These origins do not permit CORS from `localhost`;
`callCloudProxy` POSTs the request envelope to `/api/cloud-proxy` on the local
agent-server, which forwards it server-side.

```ts
import { callCloudProxy } from "../cloud/proxy";

// CORRECT -- cloud endpoint
const result = await callCloudProxy<ResponseType>({
  backend,
  method: "GET",
  path: `/api/v1/app-conversations/search?${params}`,
});

// CORRECT -- cloud runtime sandbox, auth via session key
const result = await callCloudProxy<ResponseType>({
  backend,
  method: "GET",
  hostOverride: buildHttpBaseUrl(conversationUrl),
  path: `/api/git/changes?path=${path}`,
  authMode: "session-api-key",
  sessionApiKey,
});

// WRONG -- direct fetch/axios to a cloud host is blocked by CORS in the browser
const result = await axios.get(`${backend.host}/api/v1/app-conversations`);
```

`callCloudProxy` key options:

- `backend` -- the cloud `Backend` object (provides host and bearer token)
- `hostOverride` -- override for runtime-sandbox calls; replaces `backend.host`
- `authMode` -- `"bearer"` (default, cloud) | `"session-api-key"` (runtime sandbox) | `"none"`
- `sessionApiKey` -- required when `authMode === "session-api-key"`

Standard cloud/local branch pattern used throughout the service layer:

```ts
if (getActiveBackend().backend.kind === "cloud") {
  return callCloudProxy({ backend: active, ... });
}
return new ConversationClient(getAgentServerClientOptions()).someMethod(...);
```

## No Magic Strings

Avoid inline string literals when they represent reusable user-facing copy or shared program identifiers. The `i18next/no-literal-string` rule is set to `"error"` for configured JSX text and attributes, and targeted `no-restricted-syntax` rules enforce shared translation and query-key patterns. Do not claim broader lint enforcement than `eslint.config.js` provides.

### Rule 1 — User-facing strings go through i18n

Every visible string (button labels, headings, validation messages, `aria-label`, `title`, `alt`, toast copy, placeholders) **must** be routed through `react-i18next`'s `t()` keyed by an `I18nKey` enum member. Keys are declared once in `src/i18n/translation.json` with values for all 15 supported languages (see `src/i18n/index.ts::AvailableLanguages`), and `npm run make-i18n` regenerates `src/i18n/declaration.ts` + `public/locales/<lang>/openhands.json`. Run `npm run check-translation-completeness` when translations change; it is also part of the staged-file checks.

```tsx
// CORRECT
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

const { t } = useTranslation("openhands");
return (
  <button aria-label={t(I18nKey.CHAT$DISMISS_LABEL)}>
    {t(I18nKey.CHAT$DISMISS)}
  </button>
);

// WRONG -- ships English to every locale; flagged by i18next/no-literal-string
return <button aria-label="Dismiss">Dismiss</button>;
```

Key naming follows the existing `CATEGORY$IDENTIFIER` convention (see `src/i18n/translation.json` — common prefixes: `CHAT_INTERFACE$`, `SETTINGS$`, `COMMON$`, `BUTTON$`, `HOME$`, `MICROAGENT$`, etc.). Reuse an existing prefix; only introduce a new one when no sensible bucket exists.

The configured `jsx-attributes.include` list catches literal values for common user-facing attributes such as `aria-label`, `placeholder`, `title`, and `alt`. Continue to route user-facing strings through `t()` even when an uncommon prop is outside that list.

### Rule 2 — Non-UI identifiers live in named constants, not inline literals

For strings the user never sees but the program reads (storage keys, event names, query keys, route paths, env-var names, header names, hardcoded paths, feature-flag identifiers), declare a single named constant in the closest module that owns the concept and import it everywhere else. Co-locate related constants in a tiny dedicated file (`*-keys.ts`, `*-constants.ts`) when more than two callers need them.

```ts
// CORRECT
const ONBOARDING_COMPLETED_KEY = "openhands-onboarded";
localStorage.setItem(ONBOARDING_COMPLETED_KEY, "true");

// CORRECT -- query keys go through SETTINGS_QUERY_KEYS / SECRETS_QUERY_KEYS / …
//            in src/hooks/query/query-keys.ts (enforced by no-restricted-syntax)
queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });

// WRONG -- duplicated literal across files, no compile-time link, silent typo risk
localStorage.setItem("openhands-onboarded", "true");
queryClient.invalidateQueries({ queryKey: ["settings"] });
```

Already-named constants in this repo include `DEFAULT_WORKING_DIR` (`src/api/agent-server-config.ts`), `OPENHANDS_I18N_NAMESPACE` (`src/i18n/index.ts`), `BUNDLED_BACKEND_ID` (backend registry), and the `*_QUERY_KEYS` helpers in `src/hooks/query/query-keys.ts`. Reuse these instead of re-inlining the literal.

### Rule 3 — Discriminated-union tags use string-literal types, not bare strings

When a string is part of a discriminated union or enum-like set (event kinds, backend kinds, tab IDs, agent statuses, observation result statuses), the type itself should constrain the literal. Pass values typed against that union, not raw `string`, so callers get autocomplete and the compiler catches typos.

```ts
// CORRECT
type BackendKind = "local" | "cloud";
if (backend.kind === "cloud") { … }

// WRONG -- `backend.kind` typed as `string`; "clould" compiles fine
if (backend.kind === "clould") { … }
```

### Allowed exceptions

- Test fixtures (`__tests__/`, `tests/e2e/`) may use inline literals for setup data — tests are the boundary where strings stop being magic.
- Non-localizable display glyphs (keyboard shortcuts like `⌘↩`, currency symbols, etc.) may stay inline behind an `eslint-disable-next-line i18next/no-literal-string` comment. Keep the disable on the single offending line; never widen it to a file-level disable for a single glyph.
- Generated files (`src/i18n/declaration.ts`, `public/locales/<lang>/openhands.json`) are produced by `npm run make-i18n`; do not hand-edit, do not lint-target.

When adding code that needs a new string, decide up front which rule it falls under: if a user reads it → Rule 1; if the program reads it → Rule 2; if it tags a union → Rule 3. Do not commit code that fails any of these rules just because the linter happens not to catch it.

- Use `@openhands/typescript-client` classes directly for agent-server-backed REST/workspace/event/VS Code calls. Centralize host/session API key/working-directory option assembly through `src/api/agent-server-client-options.ts`; the backend fallback policy itself lives in `src/api/backend-registry/active-store.ts`.
- Local verification/build gotchas:
  - `npm run typecheck` assumes generated translation types exist; run `npm run make-i18n` first if `src/i18n/declaration.ts` is missing.
- Original OpenHands hosted routes were removed, while current Cloud behavior is implemented explicitly through the backend registry and `src/api/cloud/`. Use `src/routes.ts` as the source of truth and do not restore old project-management, invitation, account, or git-settings routes from upstream without restoring their full API and i18n dependencies.

- `npm run dev:mock` needs MSW handlers for the direct agent-server routes used by the adapted frontend, not the original OpenHands mock paths. Key routes that must stay covered are:
  - bootstrap/model loading: `/server_info`, `/api/llm/models/verified`, `/api/llm/providers`
  - settings schemas: `/api/settings/agent-schema`, `/api/settings/conversation-schema`
  - settings CRUD: `GET /api/settings`, `PATCH /api/settings`
  - secrets CRUD: `GET /api/settings/secrets` (list), `GET /api/settings/secrets/:name` (value), `PUT /api/settings/secrets` (upsert), `DELETE /api/settings/secrets/:name`
  - conversation browsing/loading: `/api/conversations/search`, `/api/conversations?ids=...`, `/api/conversations/:id`, `/api/conversations/:id/events/*`
  - runtime git panels: `/api/git/changes`, `/api/git/diff`
- Files that MSW handlers import (demo bundles, fixture JSON) must live under `src/fixtures/` and be imported via `#/fixtures/...`. `.dockerignore` excludes `tests/` and `__tests__/` from the Docker build context, so a handler importing from those directories builds locally but fails `npm run build` inside the image (`UNRESOLVED_IMPORT`).
- Static mock verification needs a build created with `VITE_MOCK_API=true` (use `npm run build:mock`); the client must start MSW whenever that flag is enabled, even in production/static builds, otherwise routes like `/settings` and the conversations pane fall through to the static server and crash on undefined `.filter`/`.map` assumptions.
- Frontend compatibility is enforced by `assertAgentServerVersionIsSupported()` in `src/api/agent-server-compatibility.ts`, using `compatibility.minimumAgentServer` from `config/defaults.json`. `OptionService.getConfig()` calls `loadAgentServerInfo()` to enforce that floor, detect unavailable/auth-failing servers, and cache `usable_tools` for tool gating.
- Backend registry: there is no longer a separate "bundled" backend. On first read of the `openhands-backends` localStorage key (`raw === null`), `readStoredBackends()` seeds the registry with one default local backend (`makeDefaultLocalBackend()`, id `BUNDLED_BACKEND_ID = "default-local"`, host/api-key from `agent-server-config`). After that the seed is just an ordinary registered backend — users can rename or remove it like any other. `getEffectiveLocalBackend()` returns the first registered local, falling back to a synthesized default if the registry has no locals (used by API clients that need a baseline `local` target). The "Manage backends" modal and the BackendSelector dropdown both read from the single registered list, so the seeded default appears in both without any special-casing.
- Shared `Dropdown` open behavior: when the menu opens, it clears the input/search text so callers can show the current selection via `placeholder` while still rendering the full option list. Generic dropdown tests should not expect the selected label to remain in the input after reopening unless the parent explicitly controls that display.
- `useLoadOlderEvents` needs ref-based `isLoading` / `hasMore` guards in addition to React state because `ChatInterface` can trigger pagination from `onScroll`, `onWheel`, and the no-overflow effect in the same tick; closure-based state alone allows duplicate page requests.
- `ChatInterface` continuity tests should assert that conversation messages render without the full `chat-messages-skeleton`, not that `data-testid="loading-spinner"` is absent: the lazy older-events indicator reuses the shared `LoadingSpinner` component and legitimately renders that inner test id while history backfill is running.
- `useConversationHistory` now mirrors the older-events pagination fallback when the first page is exactly `INITIAL_HISTORY_PAGE_SIZE`: treat `next_page_id` **or** a full page as `hasMore`, so older agent-server variants that omit `next_page_id` still allow one more backfill request. The hook and `useLoadOlderEvents` also defensively reject mocked/malformed `page.items` responses before reversing them.

- `/server_info` tool capability metadata from `software-agent-sdk` PR #3028 ended up shipping as `usable_tools` (not `available_tools`). Frontend browser-tool gating should key off `usable_tools`, and still default to allowing tools when the server does not advertise tool metadata.

- Useful regression tests for mock mode live in `__tests__/api/option-service.test.ts`, `__tests__/api/mock-conversation-handlers.test.ts`, and `__tests__/api/mock-settings-handlers.test.ts`.
- Agent-server compatibility conventions:
  - Authenticated schema and settings calls must send the configured `X-Session-API-Key`.
  - Local provider/model discovery uses `/api/llm/providers`, `/api/llm/models`, and `/api/llm/models/verified`; Cloud model/provider search goes through the Cloud service layer.
  - `GET /api/conversations` uses repeated `ids` query parameters (`?ids=a&ids=b`).
  - Runtime git panels prefer the conversation's reported `workspace.working_dir`.
  - Conversation-start payloads use SDK-registered snake-case tool names such as `terminal`, `file_editor`, `task_tracker`, and `browser_tool_set`.
  - The `/server_info` bootstrap uses a 5-second timeout and surfaces unavailable, authentication, and unsupported-version states through the backend recovery UI.

- Git provider tokens are stored exclusively on the agent-server via `SecretsService` (`PUT /api/settings/secrets`). They are NOT mirrored to localStorage; the frontend reads which providers are connected from `settings.provider_tokens_set` (populated by `GET /api/settings`). Older notes about an `openhands-agent-server-git-provider-tokens` localStorage key are obsolete — no such key is read or written anywhere in the codebase.
- App-level user preferences (language, sound notifications, analytics consent, git identity, disabled skills) are persisted server-side under `PersistedSettings.misc_settings.app_preferences` since agent-server 1.27. `misc_settings` is a generic container for frontend-owned settings the agent doesn't interpret; `app_preferences` is currently its only nested category, but additional categories (e.g. a future `ui_preferences` for sidebar layout / view modes) drop in as additional siblings without churning the top-level wire shape. `GET /api/settings` returns the block under `misc_settings.app_preferences`, and `PATCH /api/settings` accepts a `misc_settings_diff` that is **deep-merged** into the persisted block (same semantics as `agent_settings_diff` / `conversation_settings_diff`). Partial diffs like `{"misc_settings_diff": {"app_preferences": {"language": "fr"}}}` update only the named nested field; sibling `app_preferences` fields are left untouched. Lists (`disabled_skills`) are replaced wholesale by the deep-merge. `SettingsService.transformApiResponse` hoists the nested fields onto the flat `Settings` shape so the rest of the GUI keeps reading them as top-level keys (`settings.language`, `settings.disabled_skills`, …). Do NOT reintroduce a localStorage fallback for these fields — `disabled_skills` and the rest belong in `misc_settings_diff.app_preferences`. The previous flat `app_preferences` / `app_preferences_diff` API (introduced in SDK PR #3539, never shipped to users) was replaced by the `misc_settings` container before either side reached a stable release; on-disk v2 settings files written by the flat shape are migrated automatically on first read by the agent-server. The legacy localStorage migration (`src/api/settings-service/legacy-app-preferences-migration.ts`) was removed in issue #1337 after a one-release drain period — it is no longer needed.
- Auth modes for `agent-canvas` (dev and production):
  - **Local mode** (default, no `--public` flag): A session API key is auto-generated and persisted to `~/.openhands/agent-canvas/session-api-key.txt`. The key is baked into the Vite dev server via `VITE_SESSION_API_KEY` or injected into static builds via `static-server.mjs --session-api-key`. Users never need to paste a key.
  - **Public mode** (`--public` flag): Requires `LOCAL_BACKEND_API_KEY` env var. The key is used as the agent-server session key (`OH_SESSION_API_KEYS_0`) but is NOT baked into the frontend (no `VITE_SESSION_API_KEY`, no `--session-api-key` to static-server). The frontend detects a 401 from `/server_info` via `isAgentServerAuthError()` and shows `ApiKeyEntryScreen` (`src/components/features/backends/api-key-entry-screen.tsx`). The screen reuses `BackendForm` with the host pre-filled (read-only) and prompts for the API key. On submit, the key is persisted to `localStorage['openhands-agent-server-config']` and the page reloads.
  - Dev usage: `LOCAL_BACKEND_API_KEY=my-secret npm run dev -- --public`
  - Production usage: `LOCAL_BACKEND_API_KEY=my-secret npx @openhands/agent-canvas --public`
  - The `--public` flag is supported by both `scripts/dev-with-automation.mjs` (parsed in `parseArgs()`, propagated via `config.isPublic`) and `bin/agent-canvas.mjs` (passed as `isPublic` to `main()`).
  - The 401 detection lives in `src/api/agent-server-compatibility.ts` (`isAgentServerAuthError()`), and the gate is in `src/root.tsx`'s `App` component, between the `AgentServerUnavailableError` check and the `<Outlet />` render.
  - **Key rotation resilience (non-public):** `syncLauncherDefaultLocalBackend()` in `src/api/backend-registry/storage.ts` re-runs at module init: for any stored backend whose id is `"default-local"` and whose host matches (or is loopback-equivalent to) the launcher's default, its `apiKey` is overwritten with the current `makeDefaultLocalBackend().apiKey` (sourced from `VITE_SESSION_API_KEY` or, in the published-binary path, `window.__AGENT_CANVAS_SESSION_API_KEY__`). E2E coverage: `tests/e2e/mock-llm/backends/mock-llm-auth-modes.spec.ts` (fresh-install, key-rotation, and public-mode scenarios).
- Backend/footer actions that launch modals from inside a dropdown or popover should intercept `onMouseDown` to keep the menu mounted, then perform the actual open on `onClick`. Current examples: `Add backend` / `Manage backends` in `src/components/features/backends/backend-selector.tsx`, plus the mirrored workspace-footer buttons in `src/components/features/conversation-panel/local-new-conversation-menu.tsx`.
- `BackendSelector`'s cloud-org switch paths should never rethrow from the dropdown `onChange` handler: unexpected non-Axios failures need a generic error toast instead of an unhandled promise rejection, and the malformed `(cloud backend, null org)` self-heal path should fall back to the bundled backend if `/switch` fails.
- `LocalNewConversationMenu` should support keyboard dismissal (`Escape`) for its inline popover, while still keeping the popover open when its modal children (`FolderBrowserModal`, `ManageWorkspacesModal`) are active.

- README expectation: keep the first section as a concrete, chronological from-scratch quickstart for running this frontend against a real `openhands-agent-server` (clone, install prerequisites, optional `.env`, run `npm run dev`).
- Windows-specific command syntax (PowerShell) lives in `README.windows.md`. When changing install / Docker sandbox instructions in `README.md`, update `README.windows.md` in the same PR to keep them in sync.
- `scripts/dev-safe.mjs` uses `uvx` for temporary agent-server installation — no permanent `uv tool install` needed. Environment variables (highest precedence first):
  - `OH_AGENT_SERVER_LOCAL_PATH` — absolute path to a local `software-agent-sdk` checkout. Runs the local checkout via `uvx` with `--with-editable` for `openhands-sdk`/`openhands-tools`/`openhands-workspace` and `--reinstall` for `openhands-agent-server`, so SDK edits are picked up on restart. Highest precedence.
  - `OH_AGENT_SERVER_GIT_REF` — git commit SHA or branch name (takes precedence over version)
  - `OH_AGENT_SERVER_VERSION` — specific PyPI version (e.g., "1.48.0")
  - `OH_SECRET_KEY` — secret key for settings encryption; auto-generated and persisted to `~/.openhands/agent-canvas/secret-key.txt` on first run (same file Docker uses), ensuring dev mode and Docker share the same key when both mount the same `~/.openhands` directory. Override with the env var to pin a specific key.
  - `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0` / `VITE_SESSION_API_KEY` — session API key for agent-server authentication; auto-generated using `crypto.randomBytes(32)` if not set, passed to both agent-server (`OH_SESSION_API_KEYS_0`) and frontend (`VITE_SESSION_API_KEY`)
  - Default: released PyPI version `1.48.0` for agent-server SDK libraries

- Security: launchers generate and persist a 64-character session API key at `~/.openhands/agent-canvas/session-api-key.txt` unless overridden. The agent-server and automation backend share that session key. `OH_SECRET_KEY` protects settings encryption and is persisted separately at `~/.openhands/agent-canvas/secret-key.txt`.
- `scripts/dev-safe.mjs` should fail fast if `uvx` cannot be spawned (for example missing PATH entries).
- `tools/` holds Python modules the agent-server can import: `buildAgentServerEnv` exposes the directory through `OH_EXTRA_PYTHON_PATH` (Docker: `/opt/agent-canvas/tools`, see `docker/entrypoint.sh`). `tools/canvas_ui_tool.py` is imported at startup via `--import-modules canvas_ui_tool` (appended by `buildAgentServerCommand` and by both launch lines in `docker/entrypoint.sh`), not only lazily from persisted conversation metadata: besides the legacy `canvas_ui` registration it registers the SDK's builtin `FinishTool`, so `openhands-automation` ≥ 1.9.0 presets — which dispatch remote conversations with `finish_tool_response_schema=TaskOutcome` and advertise the tool as the non-self-registering `openhands.sdk.tool.builtins.finish` — do not fail every run with `ToolDefinition 'FinishTool' is not registered`. Remove that registration once the SDK registers builtins for remote conversations.
- `npm run dev` runs the full local stack via `uvx` (agent-server + automation backend + Vite dev server + ingress proxy) with no Docker dependency. `npm run dev:static` does the same but serves a production build of the frontend instead of the Vite dev server.
- `scripts/dev-with-automation.mjs` runs the full stack: agent-server, automation backend (both via uvx), frontend server, and ingress proxy. It defaults to Vite when run directly, supports `--static` for an existing build, and supports `--dynamic` so wrappers that default static can opt back into Vite. Uses a standalone ingress proxy (`scripts/ingress.mjs`) to route traffic:
  - Keep `SIGINT`, `SIGTERM`, and `SIGHUP` wired through the coordinated shutdown handler. Services run in detached process groups on POSIX, so cleanup must use `signalProcessTree()` rather than signaling only the direct child; regression coverage lives in `__tests__/scripts/dev-with-automation.test.ts`.
  - `/api/automation/*` → automation backend (:18001)
  - `/api/*`, `/sockets`, etc. → agent server (:18000)
  - `/*` (default) → frontend server (:3001), either Vite or static depending on launcher mode
  - Environment variables: `PORT` (ingress port, default from `config/defaults.json`), `OH_AUTOMATION_GIT_REF` (git ref, overrides default version), and `OH_AUTOMATION_VERSION` (defaults to `versions.automation` in `config/defaults.json`)
  - `scripts/check-sdk-version-sync.mjs` checks the released `openhands-automation` package against `versions.agentServer` in `config/defaults.json`; these must always match — if the automation package's SDK dependencies differ from `agentServer`, the check fails.
  - Access points: `http://localhost:8000/` (main UI), `http://localhost:8000/api/automation/docs` (API docs)
  - Security: the automation backend receives the same session key as the agent-server through `AUTOMATION_LOCAL_API_KEY`; the frontend does not bake a separate automation API key.
- `scripts/ingress.mjs` is a standalone HTTP reverse proxy that can be used independently to route traffic to multiple backends based on URL path prefix.
- `scripts/dev-safe.mjs` (now `npm run dev:minimal`) runs just agent-server + Vite without automation.
- Vite dev mode can black-screen on first load with `504 Outdated Optimize Dep` if core client-entry deps are not prebundled; keep `react`, `react/jsx-runtime`, `react-dom/client`, and `react-router/dom` in `optimizeDeps.include`.
- Vercel deployment note: React Router builds for this repo must keep `build/client` intact on actual Vercel builds and include `presets: [vercelPreset()]` from `@vercel/react-router/vite`; flattening `build/client` during a Vercel build produces deployments with empty outputs (`routes: null`, no static files) and a production 404.

- The repo should include a root `LICENSE` file to satisfy the incubator-program requirements.
- OpenHands repo bootstrap files live under `.openhands/`:
  - `.openhands/setup.sh` installs `uv` (via `curl -LsSf https://astral.sh/uv/install.sh | sh`) if not present, installs frontend dependencies with `npm ci` when needed, creates `.env` from `.env.sample` if missing, appends `VITE_WORKING_DIR` for this repo when unset, and generates `src/i18n/declaration.ts` via `npm run make-i18n`.
- The repo now includes `.agents/skills/custom-codereview-guide.md`, adapted from `OpenHands/software-agent-sdk`, to force PR reviews to always leave either an APPROVE or COMMENT review instead of silently finishing with no review object.

- HeroUI rollback / migration notes:
  - The attempted HeroUI v3 upgrade changed global theme wiring and homepage design tokens enough that the repo currently prefers `@heroui/react@2.8.10` until a broader visual validation pass is done.
  - Keep the v2 Tailwind integration active via `@plugin '../hero.ts'` in `src/tailwind.css` and source HeroUI classes from `node_modules/@heroui/theme/dist/**/*`.
  - The settings UI currently relies on the v2 `Autocomplete` + `AutocompleteItem`/`AutocompleteSection` APIs in `settings-dropdown-input.tsx` and `model-selector.tsx`; a future v3 retry will need to replace those controls again.
- Library i18n is now namespace-scoped under `openhands`: `src/i18n/index.ts` exports `OPENHANDS_I18N_NAMESPACE`, `translationResources`, and `waitForI18n()`, `scripts/make-i18n-translations.cjs` emits `public/locales/<lang>/openhands.json`, standalone `src/entry.client.tsx` explicitly awaits i18n init, and host apps can register bundles via the `@openhands/agent-canvas/i18n` subpath export.

- Route decoupling note: `src/components/` should stay free of direct `react-router` imports. Route state now flows through `src/context/navigation-context.tsx`, the standalone app bridges router state with `src/routes/react-router-navigation-provider.tsx`, and link-like UI should use `src/components/shared/navigation-link.tsx`.
- Test helper note: `test-utils.tsx` now wraps renders with a default `NavigationProvider` (`currentPath: "/"`, `conversationId: "test-conversation-id"`). Navigation-sensitive tests can override that via `renderWithProviders(..., { navigation: { ... } })`.
- CSS isolation for embeddable/hosted use now relies on a scoped wrapper attribute: all bundled CSS is prefixed under `[data-agent-server-ui]` via `postcss-prefix-selector` in `vite.config.ts`, with selector exceptions handled by `transformAgentServerUISelector()` in `src/styles/agent-server-ui-style-scope.ts`. That transform must remap global selectors like `:root`, `html`, and `body` directly onto the scoped shell instead of emitting impossible descendants such as `[data-agent-server-ui] :root`.
- Public embedding entry points should use `AgentServerUIProviders` (scoped root on by default) or `AgentServerUIRoot` for manual control. The standalone app already renders its own scoped root in `src/root.tsx`, so `src/entry.client.tsx` must pass `withStyleRoot={false}` to avoid nesting duplicate shells. Keep `AgentServerUIRoot` and the scoping constants re-exported from `src/lib/index.ts` so library consumers can customize the host wrapper without reaching into private paths.
- `AgentServerUIRoot`'s themed inner wrapper must set a default `color: var(--foreground)` in addition to the `dark` / `data-theme` markers; otherwise inherited text and `currentColor` SVG icons fall back to dark browser defaults after CSS scoping, causing dark-on-dark regressions on pages like the home screen.
- Theme/customization tokens for the embedded shell are exposed as `--oh-*` CSS variables. Override them through `styleOverrides`, `style`, or host CSS targeting `[data-agent-server-ui]`; Tailwind theme tokens in `src/tailwind.css` should continue to reference those variables with `@theme inline` so host apps can restyle the UI without reworking component class names.
- Regression coverage for the CSS isolation work lives in `__tests__/agent-server-ui-providers.test.tsx`, `__tests__/agent-server-ui-style-scope.test.ts`, and the browser-level CSS-isolation test in `tests/e2e/mock-llm/regressions/mock-llm-ui-regressions.spec.ts`.

- Conversation history is loaded lazily, REST-first then WebSocket:
  - `useConversationHistory` (in `src/hooks/query/use-conversation-history.ts`) fetches only the most recent `INITIAL_HISTORY_PAGE_SIZE` (default 50) events using `sort_order='TIMESTAMP_DESC'`, then reverses to chronological order. Older pages are paginated in via `useLoadOlderEvents` when the user scrolls near the top of the chat.
  - `EventService.searchEvents(conversationId, conversationUrl, sessionApiKey, options)` returns the raw `EventSearchPage` (`{ items, next_page_id }`); options support `limit`, `pageId`, `sortOrder`, `timestampGte`, `timestampLt`. Both the local and cloud-proxy code paths forward the new params.
  - The main `ConversationWebSocketProvider` waits for the REST query to settle before opening its socket, then connects with `resend_mode='since'` and `after_timestamp=<latest preloaded event ts>` (falling back to `'all'` when the REST result is empty or errored). The legacy `resend_all=true` flag is removed for the main connection; the planning-agent sub-conversation still uses `resend_all` until it is migrated to the same REST-then-WS pattern.
  - The event store gained a bulk `addEvents(events)` action (used for the initial REST seed and for "scroll-up" pagination) that re-sorts by timestamp once at the end so older pages can be merged in cheaply. Per-event dedup still works via the existing `eventIds` set.
  - `ChatInterface` wires `useLoadOlderEvents` into its scroll handler (threshold 80px from the top), shows a `data-testid="loading-older-events"` spinner during pagination, and preserves the visible scroll offset by storing the previous `scrollHeight` and adding the height delta after the older page renders.

  - `useLoadOlderEvents` intentionally distinguishes between "no anchor yet" (empty store before the initial REST seed, so `loadOlder()` should no-op) and "malformed oldest event" (store has an oldest event with no timestamp, so the hook throws, flips `hasMore` false, and `ChatInterface` surfaces the failure via the shared error banner instead of failing silently).

- Action grouping in the chat stream:
  - `src/components/conversation-events/chat/group-events.ts` folds runs of consecutive groupable events (regular `ActionEvent`/`ObservationEvent` cards, but not `FinishAction`, `ThinkAction`, `PlanningFileEditorObservation`, `TaskTrackerObservation`, hooks, errors, or message events) into single `RenderedItem` groups. The threshold lives in `EVENT_GROUP_MIN_SIZE` (currently 2, so even pairs of back-to-back actions get folded).
  - `EventGroup` (`src/components/conversation-events/chat/event-message-components/event-group.tsx`) is the collapsible header that wraps each run. Default state is collapsed; the header shows `EVENT_GROUP$ACTIONS_COMPLETED` (with a success check) when the group is done, or `EVENT_GROUP$ACTIONS_PROGRESS` plus the currently-running action's title (from `getEventContent`) while a member `ActionEvent` has not yet been replaced by its observation in the UI events array. Expanding renders the original `EventMessage`s verbatim so each card still expands the way it did before.
  - Agent thoughts attached to an `ActionEvent` (`event.thought`) are hoisted out of groups: `groupEvents` emits a third `RenderedItem` kind `"thought"` whenever a groupable event carries (or, for an observation, originates from) a non-empty thought, flushing the current run and starting a new one. `messages.tsx` renders that item via `ThoughtEventMessage` and passes `suppressThought` to `EventMessage` so the inline thought isn't duplicated inside the group's expanded content. `ThinkAction` is excluded from this hoisting because the thought IS its action body and is rendered through its own codepath.
  - `groupEvents` now de-duplicates hoisted thoughts by action ID so mixed UI arrays that temporarily contain both an action and its replacement observation do not emit the same thought twice; `minSize` is treated as a validated internal invariant (`>= 1`).
  - `EventGroup` should return `null` for an empty `events` array and wire the toggle button to the expanded body with `aria-controls` / `role="region"` / `aria-labelledby`.
  - `src/components/conversation-events/chat/messages.tsx` is the only consumer; the grouping is transparent to upstream code. Coverage lives in `__tests__/components/conversation-events/chat/group-events.test.ts` (pure logic, including thought hoisting) and `__tests__/components/conversation-events/chat/event-message-components/event-group.test.tsx` (rendering/interaction).

- Home page workspace UX (local backend):
  - `FolderBrowserModal`'s "Use this folder" button adds **only the currently navigated directory** as a single workspace (named by its basename). It no longer iterates `subdirs` and adds each child as a separate workspace.
  - The `WorkspaceDropdown` sticky footer now exposes both "+ Add Workspace" (opens the folder browser) and "Manage Workspaces" (opens `ManageWorkspacesModal`, which lets users remove individual workspaces via `useWorkspacesStore.removeWorkspace`). The Manage button is hidden when there are no workspaces yet.
  - The conversation-panel new-thread picker (`ConversationPanelNewThreadPicker` in `src/components/features/conversation-panel/conversation-panel-new-thread-picker.tsx`, mounted from `conversation-panel.tsx`) is a `FolderPlus` icon button in the conversation-list header, not a labelled "+ New Conversation" button, and it branches on `backendKind`. Local backends get `LocalNewConversationMenu`, whose popover is a **flat list**, not the home-screen combobox: a leading "No workspace" entry plus one entry per stored workspace, each clicking through to `useCreateConversation` immediately (no separate Launch button). It mirrors the dropdown footer actions/pattern (`+ Add Workspace`, `Manage Workspaces`) locally rather than embedding `WorkspaceDropdown` itself. Cloud backends get `CloudNewConversationMenu` instead, a searchable repository picker (`useGitRepositories` / `useSearchRepositories`) with no workspace entries and no footer actions. The sidebar rail's "+ New Chat" item is a nav link to `/conversations` (`src/components/features/sidebar/sidebar-rail-body.tsx`), not this popover.
  - `useResolvedWorkspaces()` now returns `isLoading` / `isError` for parent-directory scans; `WorkspaceSelectionForm` should surface that state (status text and disabling the empty dropdown while parent results are still loading) instead of assuming the merged list is immediately ready.
  - `ManageWorkspacesModal` should require a confirmation step before removing either a saved workspace or a workspace parent; parent removals should mention the child-workspace impact, and tests should assert both the confirmation flow and that removing the selected workspace clears the launch selection.
  - In `useWorkspacesStore`, keep `clearWorkspaces()` scoped to literal workspaces only; use explicit helpers like `clearWorkspaceParents()` / `clearAll()` for broader resets so future callers do not accidentally wipe parent registrations.

- Default LLM model — `DEFAULT_SETTINGS.llm_model` (`"openhands/glm-5.2"`, defined in `src/services/settings.ts`) is the canonical frontend default. `buildConfiguredOpenHandsAgentSettings` in `src/api/agent-server-adapter.ts` **always** sends this value explicitly when the resolved `llm.model` is absent, empty, or whitespace-only — the frontend never relies on the agent-server SDK's own default (`gpt-5.5`). If you change the default model, update `DEFAULT_SETTINGS.llm_model` in `src/services/settings.ts` **and** the checklist in `specs/llm-defaults.md`. Spec: `@spec LLD-001`.

- Custom secrets are NOT auto-attached by the agent-server. `POST /api/conversations` only persists what the client sends in `request.secrets`; the persisted secrets store (`/api/settings/secrets`) is never read at conversation start. `buildStartConversationRequestWithEncryptedSettings` enumerates `SecretsService.getSecrets()` and turns each entry into a `LookupSecret` whose `url` points back at `/api/settings/secrets/{name}` and whose `headers` carry `X-Session-API-Key` for auth.

- MCP page layout: MCP is a **top-level** nav entry at `/mcp` (rendered by `src/routes/mcp.tsx`), shown right below "Skills" in `src/components/features/sidebar/sidebar.tsx`. `src/routes/mcp-settings.tsx` re-exports the new page so the published `MCPSettings` library symbol (in `src/components/settings/index.ts`) keeps the same shape. The legacy `/settings/mcp` redirect was removed in issue #1337. Marketplace catalog data and MCP logo mappings live in the MCP-capable entries from `@openhands/extensions/integrations`; the Slack API catalog option should point at `https://github.com/zencoderai/slack-mcp-server` and use `@zencoderai/slack-mcp-server`. Deprecated marketplace entries removed upstream (for example GitLab / Google Maps / Postgres / Puppeteer / SQLite) should disappear from the marketplace grid. The Installed section still needs to render and search arbitrary non-catalog custom servers via the raw server `name` / `command` fallback in `src/utils/mcp-marketplace-utils.ts` + `InstalledServerCard`. Tavily is a regular stdio MCP entry (`tavily-mcp` + `TAVILY_API_KEY`), not a special built-in sentinel anymore. Components are colocated under `src/components/features/mcp-page/` and reuse the existing `MCPServerForm` for the "Add custom server" / edit flow.
- MCP catalog runtime patching: `getMcpMarketplaceCatalog()` in `src/utils/mcp-marketplace-utils.ts` pipes every catalog entry through patch functions before the UI sees it. Two patches exist: `patchLinearEntry` (rewrites the deprecated Linear SSE endpoint to streamable HTTP) and `patchGitHubEntry` (rewrites the `docker run` transport to the native `github-mcp-server stdio` binary, only when `getDeploymentMode() === "docker"`). The `getDeploymentMode()` helper is exported from `src/api/agent-server-adapter.ts` and reads the `mode` field from the runtime services info. The Docker image pre-installs the `github-mcp-server` Go binary at `/usr/local/bin/github-mcp-server` via a dedicated Dockerfile download stage (`github-mcp-download`). This avoids a Docker-in-Docker requirement — GitHub is the only catalog entry that uses `docker` as its stdio command; all others use `npx` or `uvx`. When adding similar patches for other entries, follow the same pattern: guard on `entry.id`, check environment conditionally, spread immutably.

- Library packaging notes:
  - Public npm entrypoints now come from `src/index.ts` → `src/lib/index.ts`, with domain barrels under `src/components/{conversation,terminal,browser,files,settings,sidebar}/index.ts`.
  - `npm run build` remains the standalone app build (`react-router build`), while `npm run build:lib` runs `vite build` in library mode plus `tsc -p tsconfig.lib.json` to emit `.d.ts` files into `dist/`.
  - The library build relies on `vite.config.ts` with `BUILD_LIB=true`, preserved modules in `dist/`, and package `exports` entries that map root/subpaths to `dist/**/*.js` plus matching declaration files.
  - Declaration emit needs `src/library-env.d.ts` and the narrowed `tsconfig.lib.json`; broad `src/**/*.tsx` declaration builds pulled in route-only files and missed `?react`/window globals.
- Bundle/dev-graph hygiene (Tier 1 cleanup landed):
  - `src/i18n/translation.json` (~1 MB) is imported only by `src/i18n/resources.ts`, which `src/i18n/index.ts` re-exports as `translationResources` for the `@openhands/agent-canvas/i18n` subpath. The re-export is a `export … from` plus `/* @__PURE__ */` annotation, so rollup drops the JSON from the app build (prod `custom-toast-handlers` chunk: 909 KB -> 74 KB; `conversation` chunk: 728 KB -> 392 KB). Do not move the JSON import back into `src/i18n/index.ts` — that immediately re-bundles all translations into every chunk that imports `i18n`.
  - The environment-switch overlay is split: lightweight store/triggers live in `components/features/backends/environment-switch-store.ts`; the React component lives in `environment-switch-overlay.tsx` (re-exports the store API for back-compat). Eagerly-mounted callers (e.g. `backend-selector.tsx`) MUST import trigger helpers from the store, not the overlay file. The overlay is `React.lazy`'d from `routes/root-layout.tsx`.
  - Conditional UI is `React.lazy`'d to keep eager graphs small. Current examples include `AlertBanner` and `CommandMenu` in `root-layout.tsx`, `SettingsModal` in the sidebar, and backend/onboarding modals in `root.tsx`. Tests that assert on these mounted nodes may need `await screen.findByTestId(...)` / `waitFor(...)` instead of synchronous `getByTestId(...)`.
  - The terminal tab (`components/features/terminal/terminal.tsx`) is `React.lazy`'d in `conversation-tab-content.tsx` alongside the other tabs, so xterm + addon-fit + xterm.css don't enter the conversation route's eager graph (they ship as a separate `terminal-*.js` chunk now).
  - Avoid importing app code through `#/components/conversation-events/chat` or its `event-message-components/index.ts` barrel — they exist for `lib/index.ts` (npm subpath) consumers only. Internal callers use deep paths (`./messages`, `./event-message-components/<name>`, `./event-content-helpers/should-render-event`) so Vite dev doesn't fan out the barrel.

- Backend dropdown connectivity indicator: `useBackendsHealth` (`src/hooks/query/use-backends-health.ts`) polls each registered backend every 10s. Local backends validate the configured session key through `SettingsClient` before calling `ServerClient.getServerInfo()` and enforcing the compatibility floor. Cloud API-key backends use `getCurrentCloudApiKey()`; cookie-auth Cloud backends use `getCloudOrganizations()`. Verdicts are surfaced as a colored dot rendered through `DropdownOption.prefix`; the trigger reads its prefix from the live `options` array (not downshift's frozen `selectedItem`) so the indicator updates without remounting. The same dot is rendered in each row of `ManageBackendsModal`, which opts into a one-shot re-probe for previously disabled backends. Tests live in `__tests__/hooks/query/use-backends-health.test.tsx`, the `connection indicator` block of `__tests__/components/backends/backend-selector.test.tsx`, and `__tests__/components/backends/manage-backends-modal.test.tsx`.

- Manage Backends modal: `src/components/features/backends/manage-backends-modal.tsx` lets users edit (host/name/api-key/kind) and remove existing backends, plus add new ones inline via a "+ Add Backend" footer button that opens a `BackendFormModal`. Both the dropdown footer's "Add backend" and the manage modal's "+ Add Backend" reuse `BackendFormModal` (see `backend-form-modal.tsx`), with `mode="add"` or `mode="edit"`; `AddBackendModal` is now a thin compatibility wrapper for `BackendFormModal mode="add"`. The modal is also auto-rendered (with a no-op `onClose`) by `src/root.tsx` when the active backend is unreachable, replacing the old full-screen `MissingAgentServerNotice` onboarding screen.

- Conversation right-panel regression note: `ConversationTabs` now owns the moved refresh/build buttons, so `__tests__/components/features/conversation/conversation-tabs.test.tsx` should cover that behavior directly. The drawer's open/closed state (`isRightPanelShown` / `hasRightPanelToggled`) is intentionally **session-only**: it always starts closed on app load (or on opening a fresh/existing conversation after a restart), but it survives in-app navigation because the Zustand `useConversationStore` stays alive across React Router transitions. The `ConversationState` localStorage blob (`conversation-state-{id}`) deliberately does **not** carry a `rightPanelShown` field — `useConversationLocalStorageState` does not expose a `setRightPanelShown` setter, `sanitizeStoredState` strips the legacy `rightPanelShown` key from older persisted blobs on read, and `RightPanelToggle` / `useSelectConversationTab` only mutate the in-memory store. In tests, seed the Zustand store directly for `selectedTab` / `isRightPanelShown` / `hasRightPanelToggled` (the component sync effect currently restores only `selectedTab` from localStorage, so localStorage alone will not make a tab read as active or the drawer read as open).

- Changes tab / `FileDiffViewer` deleted-file note: the agent-server's `/api/git/diff` endpoint calls `path.exists()` first (see `openhands-sdk/openhands/sdk/git/git_diff.py` → `get_git_diff`), so requesting a diff for a `D` (deleted) file returns `GitPathError` → HTTP 400 and trips the global QueryCache error toast. `useUnifiedGitDiff` disables the query when `type === "D"` and `FileDiffViewer` renders a localized "file deleted" placeholder (`DIFF_VIEWER$FILE_DELETED`, `data-testid="file-deleted-message"`) instead of the view-mode toolbar / Monaco editor for that case.

- Onboarding modal: `src/components/features/onboarding/onboarding-modal.tsx` is rendered by `<OnboardingHost />` and normally gated by the `openhands-onboarded` localStorage flag. `OnboardingHost` also suppresses the modal, without writing that flag, when any active Cloud backend's settings report a usable LLM (non-empty model plus API-key or subscription auth); this readiness exception must not apply to Local backends. It tracks logical phases (`backend`, `agent`, `setup`, `hello`) rather than fixed numeric steps; the backend phase may be omitted for an already configured backend. Agent choices are derived from `ACP_PROVIDERS` plus OpenHands. The setup phase renders `SetupLlmStep` for OpenHands or `SetupAcpSecretsStep` for ACP providers. Keep the phase-based navigation so adding or removing the backend slide cannot move users to the wrong step.

- Files tab diff-view default logic: keyed off `useHasAttachedSource()` (`src/hooks/use-has-attached-source.ts`), which is true when the user explicitly attached _either_ a repo (`conversation.selected_repository`) _or_ a local workspace (`getStoredConversationMetadata(id).selected_workspace`, persisted by `createConversation` when `workingDirOverride` is supplied). The agent-server pre-initialises every conversation workspace as a git worktree for its own change tracking, so do NOT use a filesystem probe (`git status` / `useUnifiedGetGitChanges`) as the attachment signal — that was tried in earlier iterations and made every fresh no-attachment conversation incorrectly default to diff view. The companion `useHasGitCommits` probe (`src/hooks/query/use-has-git-commits.ts`) then suppresses diff view for attached-but-empty cases (unborn HEAD, non-git workspace).

- Collapsible thinking: `ThinkAction` events and LLM extended reasoning (`reasoning_content` / `thinking_blocks` on `ActionEvent`) are rendered as collapsible sections via `CollapsibleThinking` (`src/components/conversation-events/chat/event-message-components/collapsible-thinking.tsx`). Collapsed by default to keep the chat compact — the thinking is often in English regardless of the user's conversation language. The `getReasoningContent()` helper in `event-thought-helpers.ts` extracts the content, preferring `reasoning_content` (plain string) and falling back to Anthropic `thinking_blocks`. i18n keys: `THINKING$TITLE`, `THINKING$EXPAND`, `THINKING$COLLAPSE`. Tests: `__tests__/components/conversation-events/chat/event-message-think-action.test.tsx`.

- Agent delegation settings: the `Settings > Agent` page (`src/routes/agent-settings.tsx`) is intentionally NOT a `SdkSectionPage` wrapper. It mirrors upstream OpenHands#14418 — it flatMaps every section of `agent_settings_schema` and finds the `enable_sub_agents` field by key, so it works regardless of which section the real backend exposes the field in. Don't refactor it back to `SdkSectionPage` unless you also know the real backend's section name and add a fallback for the live "SDK schema unavailable" path. The toggle persists via `agent_settings_diff`. Nav item lives in `OSS_NAV_ITEMS` (settings-nav.tsx) with the robot icon (`SETTINGS$NAV_AGENT`). The mock schema in `settings-handlers.ts` puts the field in a `general` section. **Client-side gate**: `getAgentTools()` in `agent-server-adapter.ts` only attaches `task_tool_set` to new conversations when `agent_settings.enable_sub_agents === true`. Without that gate the agent server would still receive the tool whenever it advertised it in `/api/server_info`, so the toggle had no effect on running conversations.

- Settings naming is backend-aware today: local `/settings` is profile-oriented (`use-settings-nav-items.ts` renames the first settings item/title/subtitle to `LLM Profiles` and `chat-input-model.tsx` / `chat-input-actions.tsx` link there as `LLM Profiles`), while cloud keeps the generic `LLM Settings` copy because cloud still edits raw settings rather than saved profiles. The local profile editor (`llm-settings-local-view.tsx`) should keep explicit create/edit profile headings plus helper text so users know they are saving a profile, not mutating the current conversation directly.

- ESLint config (flat, ESLint 9): the project uses `eslint.config.js` (not `.eslintrc`) and runs on `eslint@9.x`, not 10. The constraint pinning us below 10 is `eslint-plugin-react@7.37.x`, which still calls `context.getFilename()` at rule-load time — that API was removed in ESLint 10 and `@eslint/compat`'s `fixupPluginRules` does NOT shim it. Don't try to bump eslint past 9 until eslint-plugin-react ships a v10-compatible release. Import rules come from `eslint-plugin-import-x` (the maintained fork of `eslint-plugin-import`) but are registered under both `import-x/` and `import/` prefixes via `plugins: { import: importXPlugin, ... }` so existing `// eslint-disable-next-line import/...` directives keep working. `linterOptions.reportUnusedDisableDirectives` is set to `"warn"` (not "off") so stale airbnb-era disable comments still surface in lint output without failing CI. The TS-overrides block has an `ignores: ["src/hooks/query/query-keys.ts"]` so the `no-restricted-syntax` rule banning raw `["settings", ...]` query keys doesn't fire on the file that defines the helpers themselves. No `.npmrc` / `legacy-peer-deps` flag is needed — all our plugins declare ESLint 9 peer compatibility.

- **Centralized config**: `config/defaults.json` is the single source of truth for version pins (agent-server, automation, automation SDK), port defaults, persistence paths, and package names. All consumers read from this file:
  - JS scripts (`dev-safe.mjs`, `dev-with-automation.mjs`, `check-sdk-version-sync.mjs`) read it via `JSON.parse(readFileSync(...))`.
  - Docker: a `config-gen` build stage converts the JSON to `/opt/agent-canvas/defaults.env` (shell-sourceable); `entrypoint.sh` sources it at startup.
  - CI workflow: a `Read defaults from config/defaults.json` step uses `node -p` to extract values into `$GITHUB_OUTPUT`.
  - Dockerfile ARG defaults are kept as fallbacks for local `docker build` without the CI workflow; CI always passes `--build-arg` overrides from the JSON.
  - To bump a version, edit `config/defaults.json` only — the JS scripts, Docker build, and CI workflow all derive their values from it.
- Docker all-in-one image: `.github/workflows/docker.yml` builds and publishes `ghcr.io/openhands/agent-canvas` — a combined image that bundles the agent-server (from `ghcr.io/openhands/agent-server`), the automation server (`openhands-automation` via pip), and the agent-canvas frontend (static build). The Dockerfile lives at `docker/Dockerfile`, the entrypoint at `docker/entrypoint.sh`. The workflow structure mirrors the SDK repo's `server.yml`: a `build-and-push-image` matrix job (2 × arch: amd64 on `ubuntu-24.04`, arm64 on `ubuntu-24.04-arm`) pushes arch-suffixed tags, then `merge-manifests` creates multi-arch manifests via `docker buildx imagetools create`, then `consolidate-build-info` aggregates artifacts, and `update-pr-description` updates the PR body (using `<!-- AGENT_CANVAS_DOCKER_START -->` / `<!-- AGENT_CANVAS_DOCKER_END -->` markers). The workflow triggers on push to main, `v*` tags (releases), PRs, and `workflow_dispatch`. On release tags it also pushes semver tags (e.g. `1.2.3`, `1.2`, `1`, `latest`). Fork PRs are skipped (no GHCR auth). On PRs that link an `OpenHands/software-agent-sdk` PR in the description, the Docker workflow uses that SDK PR's published branch image (`ghcr.io/openhands/agent-server:<branch-with-slashes-as-dashes>-python`) as the agent-server base image unless a `workflow_dispatch` input explicitly overrides it. The image exposes port 8000 as a unified entry point: `/api/automation/*` → automation (:18001), `/api/*` → agent-server (:18000), `/*` → static frontend. The Dockerfile accepts the public `VITE_POSTHOG_API_KEY` build arg; CI passes staging for PR/main images and production for tagged releases. The npm release workflow passes the same production key to both the app and library builds. The entrypoint auto-generates **both** the session API key and `OH_SECRET_KEY` (persisted to `~/.openhands/agent-canvas/session-api-key.txt` and `secret-key.txt` respectively) when none is provided, so the image runs secure by default. Users can override either via env var (`OH_SECRET_KEY`, `SESSION_API_KEY` / `OH_SESSION_API_KEYS_0`). `scripts/dev-safe.mjs` uses the same `secret-key.txt` file, so dev mode and Docker share the same key when both use the same `~/.openhands` directory.

- Spec files live under `specs/`. Spec IDs are stable — never renumber. Mark deprecated specs with ~~strikethrough~~. Tag implementation code and tests with `// @spec BM-002 — Short title` comments so specs are grep-able across the codebase (`grep -rn '@spec BM-' src/ __tests__/`). Place the comment on the line immediately above the relevant code block or test. When multiple tests cover the same spec, use `it.each` if the test structure is identical.

- Release automation is trunk-based through release-please. Follow `.agents/skills/release.md` for the current process.

- Electron desktop app (`npm run desktop` for dev / `npm run build:desktop` for the binary) starts the same stack as `dev-with-automation.mjs` but inside an Electron BrowserWindow. Two gotchas live here:

  1. **Boot race vs. agent-server cold start**: `dev-with-automation.mjs` is a fire-and-forget launcher — `main()` previously returned as soon as `waitForService` saw the ingress proxy respond on `/api/health`, but ingress responds immediately while the agent-server behind it can still be downloading via `uvx` (first run pulls ~50 MB of Python + the SDK from PyPI, easily 30–90s). `electron/main.mjs` used to load the URL as soon as the proxy responded, so the React app booted, called `/server_info`, and got the "Request timeout" popup. Fix: `main()` now accepts an `agentServerReadyTimeoutMs` option and returns `{ config, agentServerReady }`; `electron/main.mjs` runs a two-stage wait — Stage 1 (`waitForUrl`) confirms the ingress proxy is up, Stage 2 (`waitForAgentServer`) hits `${ingress}/server_info` and only accepts `200` (or `401`, which proves the proxy reached a real agent-server) before the BrowserWindow loads. Don't shorten the agent-server timeout below ~3 min — uvx cold start on slow connections genuinely takes that long.

  2. **First-run feedback loop**: `dev-with-automation.mjs::setServiceLogListener(cb)` exposes a workspace-wide hook that fires `cb(name, line, level)` for every line of every child-process stdout/stderr/exit. `electron/main.mjs::handleServiceLog` filters for uvx install lines (`Downloading...`, `Resolved N packages...`, etc.) plus agent-server boot markers and forwards them to `loading.html` via `setLoadingStatus()` → `window.__setLoadingStatus()`. The hook is best-effort and swallows listener errors — a buggy embedder must not be able to take down the dev stack.

- Electron desktop packaging — `electron-builder.config.mjs` uses `directories.app: "electron"` so electron/package.json is the app manifest. Even though electron/package.json has zero `dependencies`, app-builder-lib's `collectNodeModulesWithLogging` walks UP from the app dir looking for the first npm workspace that resolves modules. The next dir in line is the project root, where `npm list --json` reports the full hoisted tree (~342 dirs, ~600 MB of Vite/React/Monaco/HeroUI), and electron-builder copies all of it into `Resources/app/node_modules/`. The walk is hardcoded in `app-builder-lib/out/util/appFileCopier.js::collectNodeModulesWithLogging` — there is no config knob to disable it. Creating an empty `electron/node_modules/` does NOT help because the collector falls through to project root when it sees zero deps. **The fix is the `afterPack` hook** (`stripBundledNodeModules` in `electron-builder.config.mjs`): after electron-builder copies everything, the hook `rm -rf`s `Resources/app/node_modules/` (handling macOS `.app` bundle layout and Linux/Windows flat resources/ layout), then copies back the dependency closure of `RUNTIME_PACKAGES` (`sirv` for static-server.mjs, `httpxy` for proxy-utils.mjs/ingress.mjs — ~200 KB total). Effect: `resources/app/` drops from ~598 MB to ~7 MB; total `linux-unpacked/` from ~1 GB to ~365 MB (the rest is Electron + Chromium + the bundled `uv` binary). If a spawned backend script gains a new bare npm import, add the package to `RUNTIME_PACKAGES` — otherwise that service crashes with `ERR_MODULE_NOT_FOUND` only in the installed app. **Testing trap:** an app launched from `dist-electron/` inside the repo resolves bare specifiers against the repo's own `node_modules` (Node ESM resolution walks up from the script file), so a missing runtime package is invisible there — verify packaged builds from a copy outside the repo tree (e.g. `/Applications`). Don't add real deps to electron/package.json — any real dep would survive the strip and would also have to be hand-installed inside electron/ since the project root is npm-hoisted. `build:desktop:universal` and `--linux/--win/--mac` variants all run the same hook.

- Electron desktop app name in dev (macOS) — `npm run desktop` shows the app as "Electron" in the Dock unless `scripts/brand-dev-electron.mjs` (wired as the `predesktop` hook) has run. There are **three independent name sources** and they must all be set; getting one wrong looks like the fix silently not working. (1) `app.name` — Electron-internal, drives the menu bar, About panel and `app.getPath("userData")`. It comes from `productName` in `electron/package.json`, read by Electron's `default_app` in dev and `lib/browser/init` when packaged. Note `default_app` only reads `<arg>/package.json`, so `npm run desktop` must point electron at the `electron/` **directory** — `electron electron/main.mjs` makes it probe `electron/main.mjs/package.json`, miss, and leave `app.name` at the host bundle default. (2) `CFBundleDisplayName` / `CFBundleName` in the running bundle's Info.plist — what `lsappinfo` and `NSRunningApplication.localizedName` report. (3) **The `.app` directory name — this is what the Dock tooltip actually shows.** macOS prefers the bundle's filesystem name over the plist keys; `/Applications/DBeaver.app` displays as "DBeaver" despite `CFBundleName = "DBeaver Community"`. So patching only the plist is NOT enough — the script also renames `node_modules/electron/dist/Electron.app` → `<productName>.app` and rewrites `node_modules/electron/path.txt` to match (`getElectronPath()` in `node_modules/electron/index.js` joins path.txt onto `dist/` and silently re-downloads Electron ~100 MB if it doesn't resolve, so the two must move together). `CFBundleExecutable` is deliberately left as `Electron` — `/Applications/Antigravity.app` ships that exact value and still displays correctly, so it only affects `ps`/Activity Monitor. Editing the plist does not break code signing: Electron's dist is ad-hoc *linker-signed* (`Info.plist=not bound`, `Sealed Resources=none`), so the signature covers only the Mach-O. `npm run build:desktop` is unaffected by the rename — electron-builder packages from `~/Library/Caches/electron/electron-v*.zip`, never from `node_modules/electron/dist`. The packaged app never had the problem: electron-builder emits `<productName>.app` with matching plist keys. Already-running instances keep the name they launched with, so quit and relaunch when verifying.

- Electron desktop `node` / `npm` / `npx` PATH bridging — when the packaged `.app` is launched from Finder/Spotlight on macOS, the OS gives it a minimal PATH (`/usr/bin:/bin`). Homebrew, nvm, asdf installs of Node.js are invisible to spawned subprocesses. Two breakages flow from that: (1) backend launcher scripts that do `spawn("node", ...)` can't find Node; (2) most stdio MCP marketplace entries (Slack, GitHub, Figma, etc.) use `command: "npx"`, and when the agent-server tries to spawn them the missing `npx` makes the spawn fail with ENOENT — the SDK reports it as an `error_kind: "connection"` MCP test failure, which the install modal renders as `MCP$TEST_ERROR_CONNECTION` ("Could not reach the server. Check the URL and server type."), a misleading error since no URL is involved. **First fix attempt — DOES NOT WORK for stdio MCPs:** wrap `node`/`npm`/`npx` with thin shell scripts that run Electron with `ELECTRON_RUN_AS_NODE=1` against the package's CLI JS. That bridges the ENOENT but stdio JSON-RPC servers spawned through the wrapper exit with `McpError: Connection closed` before completing the MCP handshake — Electron-as-Node has subtly different stdin/stdout pipe semantics from a vanilla `node` binary when used as a stdio child of a windowed process. **Working fix:** bundle the real Node.js distribution. `scripts/download-node.mjs` downloads the official `node-v<ver>-<platform>-<arch>` tarball from `https://nodejs.org/dist/v<ver>/` into `resources/node/` (gitignored), prunes `include/`, `share/`, docs, and `node_modules/corepack` to keep the size down (~130 MB on Linux x64, dominated by the Node binary itself). Default pin: `NODE_BUNDLE_VERSION = "22.12.0"` (the repo's `engines.node` floor; every 22.x build shares the Electron 42 ABI); override with `NODE_VERSION=`. `electron-builder.config.mjs` ships `resources/node/` as an extraResource → `<Resources>/node/`. `electron/main.mjs::injectBundledNode()` prepends the platform-appropriate bin dir to `PATH` (POSIX: `<Resources>/node/bin`; Windows: `<Resources>/node/`) so subsequent spawns of `node`/`npm`/`npx` resolve to real binaries with full stdio fidelity. It also `chmod +x`'s the binaries on POSIX because electron-builder doesn't always preserve the bit. `injectBundledNode()` is a no-op when `!app.isPackaged` (dev `npm run desktop` uses the developer's system node). `build:desktop` and `build:desktop:universal` both run `download-node.mjs` after `download-uv.mjs`. If the bundled dir is missing at runtime, `injectBundledNode()` logs a loud `[desktop]` warning instead of silently leaving PATH bare. **extraResources will not copy the distribution's root `node_modules`:** `app-builder-lib/src/util/filter.ts::createFilter` returns `false` for any entry whose path relative to the copy root is exactly `node_modules`, *before* the `filter` patterns are consulted, so no `filter` value can opt back in. The Windows Node zip puts npm at `<root>/node_modules/npm` and hits this exactly; POSIX tarballs put it at `<root>/lib/node_modules/npm` and are unaffected — which is why this only broke Windows. Shipped result: a working `node.exe` beside `npm.cmd`/`npx.cmd` shims pointing at a missing `node_modules\npm\bin\npx-cli.js`, so every `npx -y <pkg>` spawn dies with `MODULE_NOT_FOUND` *and* shadows the user's own npm, since the dir is PREPENDED to PATH. The `afterPack` hook (`restoreBundledNodeNpm`) copies that directory into the packed output and then hard-fails the build if `npm-cli.js` still isn't there, mirroring the check `download-node.mjs` already runs on the source tree.

- Cloud conversation resume gating: when a cloud conversation is closed from the UI (`pauseCloudSandbox` is called), the conversation's `conversation_url` is NOT cleared -- it still points to the old sandbox host. `WebSocketProviderWrapper` must suppress the URL (pass `null` to `ConversationWebSocketProvider`) while `sandbox_status === "PAUSED"`, otherwise the WebSocket immediately tries the stale URL before the sandbox wakes. Symmetrically, `useActiveConversation`'s refetch interval must fast-poll (3 s) on both `!conversation_url` AND `sandbox_status === "PAUSED"` -- checking only the missing URL would leave the hook on the 30 s interval while the sandbox is resuming. The resume sequence: navigate -> sandbox PAUSED detected -> `resumeCloudSandbox` called (in `conversation.tsx`) -> fast-poll detects RUNNING -> `conversationUrl` unblocked -> WebSocket connects.
