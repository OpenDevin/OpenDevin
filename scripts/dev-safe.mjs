import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import net from "node:net";
import { homedir } from "node:os";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  getProcessTreeSpawnOptions,
  isProcessRunning,
  signalProcessTree,
} from "./dev-process-utils.mjs";
// buildRuntimeServicesInfo moved to its own dependency-free module so the
// Docker entrypoint can run it as a CLI. Re-exported below for back-compat
// (dev-with-automation.mjs and tests still import it from here).
import { buildRuntimeServicesInfo } from "./runtime-services-info.mjs";
import { fileLog, stripAnsi } from "./logger.mjs";

// ── Centralized config (single source of truth for versions, ports, etc.) ───
const __dev_safe_dirname = path.dirname(fileURLToPath(import.meta.url));
const SHARED_DEFAULTS = JSON.parse(
  readFileSync(
    path.join(__dev_safe_dirname, "..", "config", "defaults.json"),
    "utf-8",
  ),
);

const DEFAULT_BACKEND_PORT = SHARED_DEFAULTS.ports.agentServer;
// Path prefix the bundled editor is served under. The same value has to reach
// agent-server (as OH_VSCODE_BASE_PATH, so openvscode-server is launched with
// --server-base-path and advertises the prefix) and the ingress route table,
// or the advertised URL and the route that serves it disagree.
export const VSCODE_BASE_PATH = SHARED_DEFAULTS.paths.vscodeBasePath;
const DEFAULT_VITE_PORT = 3001;
const DEFAULT_WAIT_TIMEOUT_MS = 30_000;
const DEFAULT_AGENT_SERVER_PACKAGE = SHARED_DEFAULTS.packages.agentServer;
const AGENT_SERVER_GIT_REPO = "https://github.com/OpenHands/software-agent-sdk";
const LOCAL_AGENT_SERVER_SUBDIRS = [
  "openhands-agent-server",
  "openhands-sdk",
  "openhands-tools",
  "openhands-workspace",
];
const DEFAULT_AGENT_SERVER_VERSION = SHARED_DEFAULTS.versions.agentServer;
// Temporary transitive-dep pin: openhands-sdk 1.40.1 leaves agent-client-protocol
// unbounded (>=0.10.1), but acp 0.11.0 reordered the ACP prompt() args and breaks
// the SDK's ACP client. Hold acp <0.11 until a fixed SDK ships. See config/defaults.json.
const AGENT_CLIENT_PROTOCOL_CONSTRAINT =
  SHARED_DEFAULTS.constraints?.agentClientProtocol;
const DEFAULT_AGENT_SERVER_TELEMETRY_POSTHOG_API_KEY =
  SHARED_DEFAULTS.telemetry.posthogApiKey;
const DEFAULT_AGENT_SERVER_TELEMETRY_POSTHOG_HOST =
  SHARED_DEFAULTS.telemetry.posthogHost;
const AGENT_SERVER_POSTHOG_CONSTRAINT = "posthog>=6,<7";
const FRONTEND_REQUIRED_BINS = ["cross-env", "react-router"];

/**
 * Generate a cryptographically secure random API key.
 * Returns a 64-character hex string (256-bit).
 */
export function generateRandomApiKey() {
  return randomBytes(32).toString("hex");
}

// Where the auto-generated API key is persisted so it stays stable across
// `npm run dev` restarts. Keeping the key stable means the value baked into
// the frontend (VITE_SESSION_API_KEY) and the persisted backend-registry entry
// (`openhands-backends` localStorage) stay in sync without users needing to
// set anything in `.env`.
//
// To rotate the key, delete this file. To pin a key explicitly, export
// LOCAL_BACKEND_API_KEY — it takes precedence over the persisted file.
export const DEFAULT_API_KEY_PATH = path.join(
  homedir(),
  ".openhands",
  "agent-canvas",
  "api-key.txt",
);

/** @deprecated Use DEFAULT_API_KEY_PATH */
export const DEFAULT_SESSION_API_KEY_PATH = DEFAULT_API_KEY_PATH;

// Where the OH_SECRET_KEY is persisted so dev mode and Docker mode share the
// same encryption key when both use ~/.openhands as their state directory.
// docker/entrypoint.sh reads and writes this same file, so whichever mode runs
// first generates the key and the other picks it up automatically.
//
// To rotate the key, delete this file and restart both modes. To pin a key
// explicitly, export OH_SECRET_KEY — that takes precedence over the file.
export const DEFAULT_SECRET_KEY_PATH = path.join(
  homedir(),
  ".openhands",
  "agent-canvas",
  "secret-key.txt",
);

// Cache so repeated lookups within a single process return the same key,
// keyed by file path so tests can use temp paths in isolation.
const persistedApiKeyCache = new Map();

/**
 * Load the persisted default API key, generating + persisting one if the file
 * doesn't exist yet.
 *
 * Best-effort: if the file can't be written (e.g. read-only home dir), we
 * fall back to an in-memory key for this process so dev still works -- the
 * key just won't survive a restart.
 *
 * @param {string} filePath - Where to read/write the key.
 * @returns {string} The (hex) API key.
 */
export function getOrCreatePersistedApiKeyFile(
  filePath = DEFAULT_API_KEY_PATH,
) {
  return getOrCreatePersistedApiKey(filePath, "session");
}

/** @deprecated Use getOrCreatePersistedApiKeyFile */
export function getOrCreatePersistedSessionApiKey(
  filePath = DEFAULT_API_KEY_PATH,
) {
  return getOrCreatePersistedApiKeyFile(filePath);
}

/**
 * Load a persisted default API key, generating + persisting one if the file
 * doesn't exist yet.
 *
 * Best-effort: if the file can't be written (e.g. read-only home dir), we
 * fall back to an in-memory key for this process so dev still works -- the
 * key just won't survive a restart.
 *
 * @param {string} filePath - Where to read/write the key.
 * @param {string} label - Human-readable key label for warning messages.
 * @returns {string} The (hex) API key.
 */
export function getOrCreatePersistedApiKey(filePath, label = "API") {
  const cached = persistedApiKeyCache.get(filePath);
  if (cached) return cached;

  // Try to read an existing key.
  try {
    const existing = readFileSync(filePath, "utf8").trim();
    if (existing) {
      persistedApiKeyCache.set(filePath, existing);
      return existing;
    }
    // File exists but is empty -- treat as if missing and regenerate.
  } catch (error) {
    if (!isEnoentError(error)) {
      console.warn(
        `Could not read persisted ${label} API key from ${filePath}: ${error.message}. Regenerating.`,
      );
    }
  }

  // Generate and persist a new key.
  const newKey = generateRandomApiKey();
  try {
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, `${newKey}\n`, { mode: 0o600 });
  } catch (error) {
    console.warn(
      `Could not persist ${label} API key to ${filePath}: ${error.message}. Falling back to in-memory key (will not survive restarts).`,
    );
  }
  persistedApiKeyCache.set(filePath, newKey);
  return newKey;
}

/**
 * Clear the in-memory cache used by {@link getOrCreatePersistedSessionApiKey}.
 * Intended for tests that swap the persisted file path between cases.
 */
export function resetPersistedSessionApiKeyCache() {
  persistedApiKeyCache.clear();
}

function isEnoentError(error) {
  return Boolean(
    (error &&
      typeof error === "object" &&
      "code" in error &&
      error.code === "ENOENT") ||
    /ENOENT/.test(String(error)),
  );
}

/**
 * Find a free port, preferring the specified port if available.
 *
 * Tries the preferred port first; if it's busy, falls back to letting
 * the OS assign any available port. This preserves predictable defaults
 * while gracefully handling port conflicts.
 *
 * **Note on race conditions:** There is a small window between when this
 * function checks port availability and when the calling service actually
 * binds to the port. During this window, another process could theoretically
 * grab the port. This is an accepted limitation of the "check-then-use"
 * approach. Callers (like agent-server) should handle EADDRINUSE gracefully.
 * For Vite, `strictPort: true` ensures a fast failure if this occurs.
 *
 * @param {number} preferredPort - The port to try first
 * @param {string} host - The host to bind to (default: "127.0.0.1")
 * @returns {Promise<number>} The actual port that was acquired
 */
export async function findFreePort(preferredPort, host = "127.0.0.1") {
  // If preferredPort is 0, skip the check and go straight to OS assignment
  if (preferredPort > 0) {
    const preferredAvailable = await tryPort(preferredPort, host);
    if (preferredAvailable) {
      return preferredPort;
    }
  }

  // Fall back to OS-assigned port
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, host, () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

/**
 * Check if a port is available by attempting to bind to it.
 *
 * @param {number} port - The port to check
 * @param {string} host - The host to bind to
 * @returns {Promise<boolean>} True if the port is available
 */
function tryPort(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.listen(port, host, () => {
      server.close(() => resolve(true));
    });
  });
}

/**
 * Assert that all listed ports are available, throwing a descriptive error if
 * any are already in use.
 *
 * Intended as a pre-flight check before spawning services so that a concurrent
 * agent-canvas instance is detected immediately rather than silently starting
 * on a different port.
 *
 * @param {Array<{name: string, port: number}>} portConfigs - Named port list
 * @param {string} [host]
 */
export async function assertPortsFree(portConfigs, host = "127.0.0.1") {
  const results = await Promise.all(
    portConfigs.map(async ({ name, port }) => ({
      name,
      port,
      free: await tryPort(port, host),
    })),
  );
  const busy = results.filter(({ free }) => !free);
  if (busy.length === 0) return;

  const lines = busy
    .map(({ name, port }) => `   • ${name}: port ${port}`)
    .join("\n");
  throw new Error(
    `Cannot start: the following ports are already in use:\n\n${lines}\n\n` +
      `Another agent-canvas instance may already be running.\n` +
      `Stop it first, or override the port via environment variables (e.g. PORT=<other>).`,
  );
}

/**
 * Find multiple free ports at once, each preferring its specified default.
 *
 * Allocates ports sequentially to avoid race conditions between checks.
 *
 * @param {Array<{name: string, preferred: number}>} portConfigs - Port configurations
 * @param {string} host - The host to bind to (default: "127.0.0.1")
 * @returns {Promise<Record<string, number>>} Map of name to actual port
 */
export async function findFreePorts(portConfigs, host = "127.0.0.1") {
  const result = {};
  const usedPorts = new Set();

  for (const { name, preferred } of portConfigs) {
    // Try preferred if not already taken by a previous allocation
    // Skip if preferred is 0 (means "any port") or already used
    if (preferred > 0 && !usedPorts.has(preferred)) {
      const available = await tryPort(preferred, host);
      if (available) {
        result[name] = preferred;
        usedPorts.add(preferred);
        continue;
      }
    }

    // Fall back to OS-assigned port, retrying if we get a collision
    let port;
    let attempts = 0;
    const maxAttempts = 100;
    do {
      port = await findFreePort(0, host);
      if (++attempts > maxAttempts) {
        throw new Error(
          `Could not allocate unique port for "${name}" after ${maxAttempts} attempts`,
        );
      }
    } while (usedPorts.has(port));

    result[name] = port;
    usedPorts.add(port);
  }

  return result;
}

export function formatMissingUvxGuidance(cwd = process.cwd()) {
  const readmePath = path.join(cwd, "README.md");

  return [
    "Failed to start uvx. Make sure uv is installed and on your PATH.",
    "",
    "To fix this:",
    "1. Install uv:",
    "   curl -LsSf https://astral.sh/uv/install.sh | sh",
    "2. Make sure the uv bin dir is on your PATH:",
    '   export PATH="$HOME/.local/bin:$PATH"',
    "   command -v uvx",
    "",
    "Need Windows or another install method? https://docs.astral.sh/uv/getting-started/installation/",
    `See the local Quickstart for details: ${readmePath}`,
    "",
    "Other options:",
    "- npm run dev:frontend   # use an already running backend",
    "- npm run dev:mock       # run the frontend with mock APIs",
  ].join("\n");
}

function npmBinCandidates(binName, platform = process.platform) {
  const candidates = [binName];
  if (platform === "win32") {
    candidates.push(`${binName}.cmd`, `${binName}.ps1`);
  }
  return candidates;
}

export function getMissingFrontendDependencyBins(
  cwd = process.cwd(),
  platform = process.platform,
) {
  const binDir = path.join(cwd, "node_modules", ".bin");
  return FRONTEND_REQUIRED_BINS.filter(
    (binName) =>
      !npmBinCandidates(binName, platform).some((candidate) =>
        existsSync(path.join(binDir, candidate)),
      ),
  );
}

export function formatMissingFrontendDependenciesGuidance(
  missingBins,
  cwd = process.cwd(),
) {
  const missingList = missingBins.join(", ");
  return [
    "Frontend dependencies are not installed or are incomplete.",
    "",
    `Missing npm binaries: ${missingList}`,
    "",
    "Run this from the repository root:",
    "  npm ci",
    "",
    `Repository root: ${cwd}`,
  ].join("\n");
}

export function validateFrontendDependencies(
  cwd = process.cwd(),
  platform = process.platform,
) {
  const missingBins = getMissingFrontendDependencyBins(cwd, platform);
  if (missingBins.length > 0) {
    throw new Error(
      formatMissingFrontendDependenciesGuidance(missingBins, cwd),
    );
  }
}

/**
 * Modules the agent-server imports at startup (`--import-modules`). They are
 * resolved from `tools/`, which `buildAgentServerEnv` exposes through
 * OH_EXTRA_PYTHON_PATH. Importing `canvas_ui_tool` eagerly registers the SDK's
 * builtin FinishTool so automation presets (openhands-automation >= 1.9.0) can
 * resolve it on the remote conversations they dispatch — see the note at the
 * bottom of tools/canvas_ui_tool.py.
 */
export const AGENT_SERVER_IMPORT_MODULES = "canvas_ui_tool";

/**
 * Build the uvx command and arguments for running agent-server.
 *
 * Environment variables (highest precedence first):
 * - OH_AGENT_SERVER_LOCAL_PATH: Absolute path to a software-agent-sdk checkout.
 *   Runs the local checkout via uvx with editable installs of the workspace
 *   packages (openhands-sdk, openhands-tools, openhands-workspace) so source
 *   edits are picked up without a manual reinstall. The agent-server itself
 *   is rebuilt from local source on each invocation (--reinstall).
 * - OH_AGENT_SERVER_GIT_REF: Git commit SHA or branch name
 * - OH_AGENT_SERVER_VERSION: Specific PyPI version (e.g., "1.48.0")
 *
 * If none are set, defaults to the released version specified by
 * DEFAULT_AGENT_SERVER_VERSION. Set OH_AGENT_SERVER_GIT_REF to use a
 * git branch or commit instead.
 *
 * @param {Record<string, string | undefined>} env
 * @returns {{ command: string, args: string[], source: string }}
 */
export function buildAgentServerCommand(env = process.env) {
  const localPath = env.OH_AGENT_SERVER_LOCAL_PATH;
  const gitRef = env.OH_AGENT_SERVER_GIT_REF;
  const version = env.OH_AGENT_SERVER_VERSION;

  const uvxArgs = [];
  let source = "";

  if (localPath) {
    if (!path.isAbsolute(localPath)) {
      throw new Error(
        `OH_AGENT_SERVER_LOCAL_PATH must be an absolute path, got: ${localPath}`,
      );
    }
    uvxArgs.push(
      "--reinstall",
      "--from",
      path.join(localPath, "openhands-agent-server"),
      "--with-editable",
      path.join(localPath, "openhands-sdk"),
      "--with-editable",
      path.join(localPath, "openhands-tools"),
      "--with-editable",
      path.join(localPath, "openhands-workspace"),
      "--with",
      AGENT_SERVER_POSTHOG_CONSTRAINT,
      "agent-server",
    );
    source = `local (${localPath})`;
  } else if (gitRef) {
    // Use git ref with subdirectory syntax for uv workspace monorepo.
    // The software-agent-sdk repo has packages in subdirectories:
    // openhands-agent-server/, openhands-sdk/, openhands-tools/, openhands-workspace/
    // All four must come from the same ref so inter-package APIs stay in sync.
    //
    // --reinstall is required because the git branch may carry the same version
    // string as the current PyPI release (e.g. both "1.26.0"). Without it, uv
    // silently reuses the cached PyPI wheels and the git ref is never actually
    // used, even though it was explicitly requested.
    const baseGitUrl = `git+${AGENT_SERVER_GIT_REPO}@${gitRef}`;
    uvxArgs.push(
      "--reinstall",
      "--from",
      `${baseGitUrl}#subdirectory=openhands-agent-server`,
      "--with",
      `${baseGitUrl}#subdirectory=openhands-sdk`,
      "--with",
      `${baseGitUrl}#subdirectory=openhands-tools`,
      "--with",
      `${baseGitUrl}#subdirectory=openhands-workspace`,
      "--with",
      AGENT_SERVER_POSTHOG_CONSTRAINT,
      "agent-server",
    );
    source = `git (${gitRef})`;
  } else if (version) {
    // Use specific PyPI version: uvx --from openhands-agent-server==version agent-server
    // The package name differs from the executable name, so we need --from syntax
    // Pin all SDK packages to the same version for consistency
    uvxArgs.push(
      "--from",
      `${DEFAULT_AGENT_SERVER_PACKAGE}==${version}`,
      "--with",
      `openhands-sdk==${version}`,
      "--with",
      `openhands-tools==${version}`,
      "--with",
      `openhands-workspace==${version}`,
    );
    if (AGENT_CLIENT_PROTOCOL_CONSTRAINT) {
      uvxArgs.push("--with", AGENT_CLIENT_PROTOCOL_CONSTRAINT);
    }
    uvxArgs.push("--with", AGENT_SERVER_POSTHOG_CONSTRAINT);
    uvxArgs.push("agent-server");
    source = `PyPI (${version})`;
  } else {
    // Default to released PyPI version
    // Pin all SDK packages to the same version for consistency
    uvxArgs.push(
      "--from",
      `${DEFAULT_AGENT_SERVER_PACKAGE}==${DEFAULT_AGENT_SERVER_VERSION}`,
      "--with",
      `openhands-sdk==${DEFAULT_AGENT_SERVER_VERSION}`,
      "--with",
      `openhands-tools==${DEFAULT_AGENT_SERVER_VERSION}`,
      "--with",
      `openhands-workspace==${DEFAULT_AGENT_SERVER_VERSION}`,
    );
    if (AGENT_CLIENT_PROTOCOL_CONSTRAINT) {
      uvxArgs.push("--with", AGENT_CLIENT_PROTOCOL_CONSTRAINT);
    }
    uvxArgs.push("--with", AGENT_SERVER_POSTHOG_CONSTRAINT);
    uvxArgs.push("agent-server");
    source = `PyPI (${DEFAULT_AGENT_SERVER_VERSION}, default)`;
  }

  // Everything after the executable name is an agent-server CLI argument.
  // Import the registration module before any conversation is created.
  uvxArgs.push("--import-modules", AGENT_SERVER_IMPORT_MODULES);

  return {
    command: "uvx",
    args: uvxArgs,
    source,
  };
}

function parsePort(value, fallback) {
  if (value == null || value === "") {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid port: ${value}`);
  }

  return parsed;
}

/**
 * Build safe dev configuration (synchronous version).
 *
 * Uses the port values from environment variables or defaults WITHOUT checking
 * port availability. Use this when:
 * - You need synchronous config (e.g., for test setup, config inspection)
 * - Ports are already known to be available (e.g., specified via env vars)
 * - You're building config objects for downstream use, not starting services
 *
 * For scripts that actually start services (dev-safe.mjs main, dev-with-automation.mjs),
 * use {@link buildSafeDevConfigAsync} instead to handle port conflicts gracefully.
 *
 * @param {string} cwd - Current working directory
 * @param {Record<string, string | undefined>} env - Environment variables
 * @returns {SafeDevConfig} Configuration object
 */
export function buildSafeDevConfig(cwd = process.cwd(), env = process.env) {
  const backendPort = parsePort(
    env.OH_CANVAS_SAFE_BACKEND_PORT,
    DEFAULT_BACKEND_PORT,
  );
  const vscodePort = parsePort(env.OH_CANVAS_SAFE_VSCODE_PORT, backendPort + 1);

  return buildConfigFromPorts({ backendPort, vscodePort }, cwd, env);
}

/**
 * Build safe dev configuration with dynamic port allocation.
 *
 * Tries preferred ports first; if busy, finds available alternatives.
 * This is the recommended entry point for scripts that start services.
 *
 * @param {string} cwd - Current working directory
 * @param {Record<string, string | undefined>} env - Environment variables
 * @returns {Promise<SafeDevConfig>} Configuration object with allocated ports
 */
export async function buildSafeDevConfigAsync(
  cwd = process.cwd(),
  env = process.env,
) {
  // Get preferred ports from env or defaults
  const preferredBackendPort = parsePort(
    env.OH_CANVAS_SAFE_BACKEND_PORT,
    DEFAULT_BACKEND_PORT,
  );
  const preferredVscodePort = parsePort(
    env.OH_CANVAS_SAFE_VSCODE_PORT,
    preferredBackendPort + 1,
  );

  // Fail fast if any required port is already in use.
  await assertPortsFree([
    { name: "agent-server", port: preferredBackendPort },
    { name: "vscode", port: preferredVscodePort },
  ]);

  return buildConfigFromPorts(
    { backendPort: preferredBackendPort, vscodePort: preferredVscodePort },
    cwd,
    env,
  );
}

/**
 * @typedef {object} SafeDevConfig
 * @property {string} cwd
 * @property {number} backendPort
 * @property {number} vscodePort
 * @property {string} vscodeBasePath
 * @property {string} stateDir
 * @property {string} tmuxTmpDir
 * @property {string} conversationsPath
 * @property {string} workspacesPath
 * @property {string} bashEventsDir
 * @property {string} backendBaseUrl
 * @property {string} backendHost
 * @property {string} workingDir
 * @property {string} secretKey
 * @property {string} sessionApiKey
 * @property {string} canvasToolsDir
 */

/**
 * Internal helper to build config from already-resolved ports.
 * @param {{backendPort: number, vscodePort: number}} ports
 * @param {string} cwd
 * @param {Record<string, string | undefined>} env
 * @returns {SafeDevConfig}
 */
function buildConfigFromPorts(ports, cwd, env) {
  const { backendPort, vscodePort } = ports;
  const stateDir = path.resolve(
    cwd,
    env.OH_CANVAS_SAFE_STATE_DIR ||
      path.join(homedir(), ".openhands", "agent-canvas"),
  );
  const conversationsPath = path.join(stateDir, "dev_conversations");
  const workspacesPath = path.join(stateDir, "workspaces");
  // Use provided secret key, or read/generate one persisted to
  // ~/.openhands/agent-canvas/secret-key.txt. Persisting ensures dev mode
  // and Docker mode share the same encryption key when they mount the same
  // ~/.openhands directory (docker/entrypoint.sh reads/writes the same file).
  const secretKeyPath = env.OH_SECRET_KEY_PATH || DEFAULT_SECRET_KEY_PATH;
  const secretKey =
    env.OH_SECRET_KEY || getOrCreatePersistedApiKey(secretKeyPath, "secret");
  // Use the user-provided LOCAL_BACKEND_API_KEY or fall back to a key
  // persisted to ~/.openhands/agent-canvas/api-key.txt. Persisting on disk
  // keeps the agent-server, the Vite-baked VITE_SESSION_API_KEY, and any
  // `openhands-backends` localStorage entries the frontend has cached all
  // pointing at the same value across dev restarts.
  //
  // LOCAL_BACKEND_API_KEY is the single user-facing env var for the API key.
  // OH_SESSION_API_KEY_PATH overrides the persisted file path (used by tests).
  const persistedKeyPath = env.OH_SESSION_API_KEY_PATH || DEFAULT_API_KEY_PATH;
  const sessionApiKey =
    env.LOCAL_BACKEND_API_KEY ||
    getOrCreatePersistedApiKeyFile(persistedKeyPath);

  // Host directory containing the legacy canvas_ui Python module. Persisted
  // conversations created before the client_tools migration still reference
  // its module qualname, so the agent-server can import it when resuming them.
  const canvasToolsDir = fileURLToPath(new URL("../tools", import.meta.url));

  return {
    cwd,
    backendPort,
    vscodePort,
    vscodeBasePath: VSCODE_BASE_PATH,
    stateDir,
    // tmux socket directory. Defaults to <stateDir>/tmux (under
    // ~/.openhands/agent-canvas), matching where the rest of dev state lives
    // and persisting across restarts.
    //
    // Do NOT use os.tmpdir() here: on macOS it resolves to the per-user
    // $TMPDIR (/var/folders/.../T), which the OS periodically reaps
    // (com.apple.bsd.dirhelper deletes entries untouched for a few days).
    // Reaping deletes the live tmux socket while the server process keeps
    // running, orphaning it — every later new-window then fails with
    // "error connecting to .../openhands (No such file or directory)".
    //
    // The only hosts where <stateDir>/tmux can't hold the socket are those
    // whose $HOME is a network/overlay mount without Unix-domain-socket
    // support (some devcontainers, NFS/CIFS homes). Those rare cases can point
    // tmux at a local, socket-capable path with the standard TMUX_TMPDIR env
    // var (e.g. TMUX_TMPDIR=/tmp), which we honor and pass through below.
    tmuxTmpDir: env.TMUX_TMPDIR || path.join(stateDir, "tmux"),
    conversationsPath,
    workspacesPath,
    bashEventsDir: path.join(stateDir, "bash_events"),
    backendBaseUrl: `http://127.0.0.1:${backendPort}`,
    backendHost: `127.0.0.1:${backendPort}`,
    workingDir: env.VITE_WORKING_DIR || workspacesPath,
    secretKey,
    sessionApiKey,
    canvasToolsDir,
  };
}

/**
 * Telemetry-related env vars for the agent-server process.
 *
 * Split out from `buildAgentServerEnv` so callers that assemble their own
 * agent-server environment can reuse the same mapping.
 *
 * @param {Record<string, string | undefined>} [env] - Source environment.
 * @returns {Record<string, string>} Telemetry env vars for agent-server
 */
export function buildAgentServerTelemetryEnv(env = process.env) {
  const telemetryDisabled =
    env.VITE_DO_NOT_TRACK === "1" || env.DO_NOT_TRACK === "1";
  const result = {};

  for (const key of [
    "OH_TELEMETRY_EXPORTER",
    "OH_TELEMETRY_POSTHOG_API_KEY",
    "OH_TELEMETRY_POSTHOG_HOST",
    "OH_TELEMETRY_HTTP_ENDPOINT",
    "OH_TELEMETRY_HTTP_TOKEN",
    "OH_TELEMETRY_CONSENT",
    "OH_TELEMETRY_CONSENT_MODE",
    "OH_TELEMETRY_SALT",
  ]) {
    if (env[key]) result[key] = env[key];
  }

  if (telemetryDisabled) {
    result.DO_NOT_TRACK = "1";
  }

  const apiKey =
    env.OH_TELEMETRY_POSTHOG_API_KEY ||
    env.VITE_POSTHOG_API_KEY ||
    (telemetryDisabled ? "" : DEFAULT_AGENT_SERVER_TELEMETRY_POSTHOG_API_KEY);
  const exporter = env.OH_TELEMETRY_EXPORTER || (apiKey ? "posthog" : "");

  if (exporter) {
    result.OH_TELEMETRY_EXPORTER = exporter;
  }

  if (exporter === "posthog" && apiKey) {
    result.OH_TELEMETRY_POSTHOG_API_KEY = apiKey;
    result.OH_TELEMETRY_POSTHOG_HOST =
      env.OH_TELEMETRY_POSTHOG_HOST ||
      env.VITE_POSTHOG_HOST ||
      DEFAULT_AGENT_SERVER_TELEMETRY_POSTHOG_HOST;
  }

  return result;
}

/**
 * Build the environment variables object for spawning the agent-server process.
 *
 * This is exported so downstream consumers (e.g., automation service) can use
 * the same env vars without duplicating the mapping logic.
 *
 * `vscodeBasePath` is an explicit opt-in rather than a field read off `config`,
 * and that is deliberate. Setting it changes the URL `/api/vscode/url`
 * advertises: agent-server appends the prefix to the browser origin the
 * frontend sends, so the editor is only reachable if the same origin also
 * routes that prefix to the editor port. A launcher that sets it without
 * registering the route advertises `<origin>/vscode/…`, which serves the
 * canvas SPA shell instead of the editor.
 *
 * Requiring the caller to name it makes the pairing greppable: every call site
 * that passes `vscodeBasePath` must also register a matching route, and
 * `__tests__/scripts/vscode-base-path-opt-in.test.ts` asserts that no launcher
 * opts in without one.
 *
 * @param {ReturnType<typeof buildSafeDevConfig>} config - Config from buildSafeDevConfig
 * @param {{vscodeBasePath?: string | null, env?: Record<string, string | undefined>}} [options]
 * @param {string | null} [options.vscodeBasePath] - Opt into prefix-mode by
 *   passing the path prefix the caller also routes to `config.vscodePort`.
 * @param {Record<string, string | undefined>} [options.env] - Source
 *   environment for the telemetry mapping (defaults to `process.env`).
 * @returns {Record<string, string>} Environment variables for agent-server
 */
export function buildAgentServerEnv(config, options = {}) {
  const { vscodeBasePath = null, env = process.env } = options;
  return {
    ...buildAgentServerTelemetryEnv(env),
    // Force Python to use UTF-8 for all file I/O and streams.
    //
    // On Windows, Python defaults to the system ANSI codepage (e.g. cp1252).
    // The agent-server writes conversation metadata JSON that can contain
    // emoji (e.g. ✅ U+2705) which cp1252 cannot encode, producing:
    //   UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
    // Setting PYTHONUTF8=1 enables Python's UTF-8 mode (PEP 540) for the
    // entire agent-server process, matching the behaviour on Linux/macOS
    // where the locale is already UTF-8.
    // This is a no-op on Linux/macOS where the locale is already UTF-8.
    PYTHONUTF8: "1",
    TMUX_TMPDIR: config.tmuxTmpDir,
    // Parent of stateDir (= ~/.openhands) so settings/secrets match Docker.
    OH_PERSISTENCE_DIR: path.dirname(config.stateDir),
    OH_CONVERSATIONS_PATH: config.conversationsPath,
    OH_BASH_EVENTS_DIR: config.bashEventsDir,
    OH_VSCODE_PORT: String(config.vscodePort),
    // Serve the editor under a path prefix on the canvas origin rather than on
    // its own published port. agent-server passes this to openvscode-server as
    // --server-base-path and includes it in the URL from /api/vscode/url, which
    // matches the ingress route the caller registers for the same prefix.
    //
    // Omitted unless the caller opts in — see the note on this function.
    ...(vscodeBasePath ? { OH_VSCODE_BASE_PATH: vscodeBasePath } : {}),
    OH_SECRET_KEY: config.secretKey,
    // Use OH_SESSION_API_KEYS_0 for agent-server V1 config format
    OH_SESSION_API_KEYS_0: config.sessionApiKey,
    // Alias for the agent-server's own URL. The agent-server itself sets
    // OH_INTERNAL_SERVER_URL at startup, but downstream consumers (the
    // OpenHands SDK boilerplate emitted by automation prompt/plugin
    // presets) read AGENT_SERVER_URL — the canonical SDK name. Mirror it
    // here so automation runs work without each tarball having to know
    // about the OH_-prefixed variant.
    //
    // We deliberately do NOT set a SESSION_API_KEY alias: the SDK's
    // sanitized_env() would strip it from bash subprocesses anyway, and
    // a follow-up change to the automation preset reads
    // OH_SESSION_API_KEYS_0 directly (which is already in env).
    AGENT_SERVER_URL: config.backendBaseUrl,
    // Let the agent-server resolve canvas_ui_tool when old persisted metadata
    // requests that compatibility module during startup.
    OH_EXTRA_PYTHON_PATH: config.canvasToolsDir,
  };
}

// Re-export so existing importers (dev-with-automation.mjs, tests) keep
// resolving `buildRuntimeServicesInfo` from this module. The implementation
// now lives in ./runtime-services-info.mjs (imported at the top of this file).
export { buildRuntimeServicesInfo };

export function buildNpmScriptCommand(
  scriptName,
  platform = process.platform,
  env = process.env,
  nodeExecPath = process.execPath,
) {
  // On Windows, always use cmd.exe regardless of whether npm_execpath is set.
  // npm_execpath points to a path like
  // "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" which contains
  // spaces. When that path is passed as an argument with shell:true in
  // spawnService, cmd.exe splits on the space and tries to run "C:\Program"
  // as a command, producing "not recognized as an internal or external command".
  // Using "npm" via cmd.exe avoids the problem entirely.
  if (platform === "win32") {
    return {
      command: env.ComSpec || "cmd.exe",
      args: ["/d", "/s", "/c", "npm", "run", scriptName],
    };
  }

  if (env.npm_execpath) {
    return {
      command: env.npm_node_execpath || nodeExecPath,
      args: [env.npm_execpath, "run", scriptName],
    };
  }

  return {
    command: "npm",
    args: ["run", scriptName],
  };
}

export function validateLocalAgentServerPath(localPath) {
  if (!path.isAbsolute(localPath)) {
    throw new Error(
      `OH_AGENT_SERVER_LOCAL_PATH must be an absolute path, got: ${localPath}`,
    );
  }
  if (!existsSync(localPath)) {
    throw new Error(`OH_AGENT_SERVER_LOCAL_PATH does not exist: ${localPath}`);
  }
  for (const subdir of LOCAL_AGENT_SERVER_SUBDIRS) {
    const subdirPath = path.join(localPath, subdir);
    if (!existsSync(subdirPath)) {
      throw new Error(
        `OH_AGENT_SERVER_LOCAL_PATH is missing expected workspace package '${subdir}': ${subdirPath}`,
      );
    }
  }
}

async function waitForServer(url, timeoutMs = DEFAULT_WAIT_TIMEOUT_MS) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // Keep polling until timeout.
    }

    await delay(500);
  }

  throw new Error(`Timed out waiting for agent-server at ${url}`);
}

function spawnProcess(command, args, options = {}) {
  const child = spawn(
    command,
    args,
    getProcessTreeSpawnOptions({
      stdio: "inherit",
      ...options,
    }),
  );

  child.once("error", (error) => {
    if (isEnoentError(error) && command === "uvx") {
      const msg = formatMissingUvxGuidance(options?.cwd);
      console.error(msg);
      fileLog("error", stripAnsi(msg));
    } else if (isEnoentError(error)) {
      const msg = `Failed to start ${command}. Make sure it is installed and on your PATH.`;
      console.error(msg);
      fileLog("error", msg);
    } else {
      console.error(`Failed to start ${command}:`, error);
      fileLog("error", `Failed to start ${command}: ${error.message}`);
    }
  });

  return child;
}

async function main() {
  console.log("Starting isolated agent-server + frontend dev stack...");
  fileLog("info", "Starting isolated agent-server + frontend dev stack...");
  validateFrontendDependencies();
  console.log("Frontend dependencies found.");
  fileLog("info", "Frontend dependencies found.");
  console.log("Allocating ports...");
  fileLog("info", "Allocating ports...");

  // Use async config builder with dynamic port allocation
  const config = await buildSafeDevConfigAsync();

  if (process.env.OH_AGENT_SERVER_LOCAL_PATH) {
    validateLocalAgentServerPath(process.env.OH_AGENT_SERVER_LOCAL_PATH);
  }

  for (const dir of [
    config.stateDir,
    config.tmuxTmpDir,
    config.conversationsPath,
    config.workspacesPath,
    config.bashEventsDir,
  ]) {
    mkdirSync(dir, { recursive: true });
  }

  const agentServerCmd = buildAgentServerCommand();

  const secretKeySource = process.env.OH_SECRET_KEY
    ? "custom (from OH_SECRET_KEY)"
    : `persisted (${process.env.OH_SECRET_KEY_PATH || DEFAULT_SECRET_KEY_PATH})`;

  const sessionKeySource = process.env.LOCAL_BACKEND_API_KEY
    ? "custom (from LOCAL_BACKEND_API_KEY)"
    : `persisted (${
        process.env.OH_SESSION_API_KEY_PATH || DEFAULT_API_KEY_PATH
      })`;

  console.log(`- agent-server: ${agentServerCmd.source}`);
  console.log(`- backend: ${config.backendBaseUrl}`);
  console.log(`- vscode port: ${config.vscodePort}`);
  console.log(`- working dir: ${config.workingDir}`);
  console.log(`- isolated state dir: ${config.stateDir}`);
  console.log(`- secret key: ${secretKeySource}`);
  console.log(`- session API key: ${sessionKeySource}`);
  console.log("");
  fileLog(
    "info",
    [
      "Agent-server stack config:",
      `  agent-server: ${agentServerCmd.source}`,
      `  backend:      ${config.backendBaseUrl}`,
      `  working dir:  ${config.workingDir}`,
      `  state dir:    ${config.stateDir}`,
    ].join("\n"),
  );

  const backend = spawnProcess(
    agentServerCmd.command,
    [
      ...agentServerCmd.args,
      "--host",
      "127.0.0.1",
      "--port",
      String(config.backendPort),
    ],
    {
      cwd: config.cwd,
      env: {
        ...process.env,
        // Opt into prefix-mode: the Vite dev server proxies the same prefix to
        // `config.vscodePort` (see VITE_VSCODE_TARGET below), so the advertised
        // URL resolves on the frontend origin the browser is actually on.
        ...buildAgentServerEnv(config, {
          vscodeBasePath: config.vscodeBasePath,
        }),
      },
    },
  );

  let shuttingDown = false;
  let frontend = null;

  const shutdown = (signal = "SIGTERM") => {
    if (shuttingDown) {
      return;
    }

    shuttingDown = true;
    if (frontend) {
      signalProcessTree(frontend, signal);
    }
    signalProcessTree(backend, signal);

    setTimeout(() => {
      if (frontend && isProcessRunning(frontend)) {
        signalProcessTree(frontend, "SIGKILL");
      }
      if (isProcessRunning(backend)) {
        signalProcessTree(backend, "SIGKILL");
      }
      process.exit(process.exitCode ?? 0);
    }, 3000);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  // Services are spawned detached, so a SIGHUP that kills this launcher (terminal
  // or multiplexer death) would otherwise leave the whole tree running. Forward
  // SIGTERM rather than SIGHUP: uvicorn only handles SIGINT/SIGTERM, so a
  // forwarded SIGHUP would terminate the agent-server by default action instead
  // of shutting it down gracefully.
  process.on("SIGHUP", () => shutdown("SIGTERM"));

  const backendErrored = new Promise((_, reject) => {
    backend.once("error", (error) => reject(error));
  });
  const backendExited = new Promise((_, reject) => {
    backend.once("exit", (code, signal) => {
      if (!shuttingDown) {
        reject(
          new Error(
            `agent-server exited before startup completed (code=${code ?? "null"}, signal=${signal ?? "null"})`,
          ),
        );
      }
    });
  });

  try {
    await Promise.race([
      waitForServer(`${config.backendBaseUrl}/server_info`),
      backendErrored,
      backendExited,
    ]);
  } catch (error) {
    shutdown();
    throw error;
  }

  const frontendCommand = buildNpmScriptCommand("dev:frontend");
  frontend = spawnProcess(frontendCommand.command, frontendCommand.args, {
    cwd: config.cwd,
    env: {
      ...process.env,
      VITE_BACKEND_HOST: config.backendHost,
      VITE_BACKEND_BASE_URL: config.backendBaseUrl,
      VITE_WORKING_DIR: config.workingDir,
      // Pass session API key so frontend can authenticate with agent-server
      VITE_SESSION_API_KEY: config.sessionApiKey,
      // This mode has no static server or ingress in front of Vite, so Vite's
      // own proxy is the only thing that can serve the editor prefix on the
      // frontend origin. The editor is a separate process on a port of its
      // own, so it needs its own proxy target rather than VITE_BACKEND_HOST.
      VITE_VSCODE_BASE_PATH: config.vscodeBasePath,
      VITE_VSCODE_TARGET: `http://127.0.0.1:${config.vscodePort}`,
      // dev:minimal deliberately does NOT supply runtime-services info (the
      // frontend here talks straight to the agent-server over
      // VITE_BACKEND_BASE_URL — there is no ingress or static-server in front
      // of it to append `runtime_services` to `/server_info`, and the
      // frontend's own VITE_RUNTIME_SERVICES_INFO env var is no longer read).
      // It is a bare agent-server + Vite stack with no companion services to
      // advertise, so `fetchBackendRuntimeServicesInfo()` correctly returns
      // null and conversations simply omit the <RUNTIME_SERVICES> block.
      // Stacks with automation/ingress/frontend services should use
      // `npm run dev` / `dev:static`, which pass runtime-services info through
      // ingress/static-server instead.
    },
  });

  frontend.once("exit", (code) => {
    shutdown();
    process.exitCode = code ?? 0;
  });

  backend.once("exit", (code) => {
    if (!shuttingDown) {
      const msg = `agent-server exited unexpectedly with code ${code ?? 0}`;
      console.error(msg);
      fileLog("error", msg);
      shutdown();
      process.exitCode = code ?? 1;
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversation lease cleanup
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns true if `host:port` accepts a TCP connection within `timeoutMs`.
 * Used to detect a live agent-server we shouldn't disturb.
 */
export function isPortBusy(port, host = "127.0.0.1", timeoutMs = 500) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const finish = (busy) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(busy);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    socket.connect(port, host);
  });
}

/**
 * Remove stale `owner_lease.json` files under `conversationsDir` so a
 * freshly spawned agent-server can claim ownership and re-load every
 * existing conversation.
 *
 * Why this is needed: each conversation directory carries an
 * `owner_lease.json` that locks it to a single agent-server's
 * `owner_instance_id` for a 45 s TTL refreshed by heartbeat. On
 * graceful shutdown the agent-server unlinks its leases; on a hard
 * kill (or a fast restart, well under 45 s) the leases linger. A new
 * agent-server with a fresh `owner_instance_id` will then raise
 * `ConversationLeaseHeldError` for each conversation at startup load
 * and skip it entirely — `/api/conversations/search` returns `[]`
 * even though the meta files are right there on disk.
 *
 * The caller MUST verify (e.g. with `isPortBusy`) that no agent-server
 * is currently bound to the backend port before calling this — there
 * is no other reliable way to tell a stale lease from an actively
 * renewed one.
 *
 * Returns the number of lease files unlinked.
 */
export function releaseStaleConversationLeases(conversationsDir) {
  if (!existsSync(conversationsDir)) return 0;

  let removed = 0;
  for (const name of readdirSync(conversationsDir)) {
    const convDir = path.join(conversationsDir, name);
    let isDir = false;
    try {
      isDir = statSync(convDir).isDirectory();
    } catch {
      continue;
    }
    if (!isDir) continue;

    const leasePath = path.join(convDir, "owner_lease.json");
    if (!existsSync(leasePath)) continue;
    try {
      unlinkSync(leasePath);
      removed += 1;
    } catch {
      // Best-effort: the new agent-server will simply skip this
      // conversation as before. Don't fail the whole start.
    }
  }
  return removed;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(msg);
    fileLog("error", `Fatal error: ${msg}`);
    if (error instanceof Error && error.stack) {
      fileLog("error", error.stack);
    }
    process.exit(1);
  });
}
