// @vitest-environment node
// These tests load `scripts/dev-safe.mjs`, which constructs file:// URLs
// relative to its own location via `new URL("../tools", import.meta.url)`.
// jsdom's URL constructor ignores file:// base URLs (it falls back to its
// document base, e.g. http://localhost:3000/), breaking that resolution;
// the Node environment has the standard WHATWG URL behavior that honors
// the file:// base.
import net from "node:net";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { homedir } from "node:os";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { describe, expect, it, afterEach } from "vitest";
import {
  assertPortsFree,
  buildSafeDevConfig,
  buildSafeDevConfigAsync,
  buildNpmScriptCommand,
  buildAgentServerCommand,
  buildAgentServerEnv,
  buildAgentServerTelemetryEnv,
  buildRuntimeServicesInfo,
  formatMissingUvxGuidance,
  formatMissingFrontendDependenciesGuidance,
  getMissingFrontendDependencyBins,
  validateFrontendDependencies,
  validateLocalAgentServerPath,
  findFreePort,
  findFreePorts,
  getOrCreatePersistedSessionApiKey,
  resetPersistedSessionApiKeyCache,
} from "../../scripts/dev-safe.mjs";
import {
  chmodSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

describe("findFreePort", () => {
  const servers: net.Server[] = [];

  afterEach(() => {
    // Clean up any servers we created
    for (const server of servers) {
      server.close();
    }
    servers.length = 0;
  });

  it("returns preferred port when available", async () => {
    // Port 9999 should be free (unlikely to be in use during tests)
    const port = await findFreePort(9999, "127.0.0.1");
    expect(port).toBe(9999);
  });

  it("falls back to OS-assigned port when preferred is busy", async () => {
    // Create a server that holds a port
    const busyPort = await new Promise<number>((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, "127.0.0.1", () => {
        const addr = server.address();
        if (addr && typeof addr === "object") {
          servers.push(server);
          resolve(addr.port);
        } else {
          server.close();
          reject(new Error("Failed to get server address"));
        }
      });
    });

    // Now try to get that busy port
    const allocatedPort = await findFreePort(busyPort, "127.0.0.1");

    // Should get a different port since busyPort is taken
    expect(allocatedPort).not.toBe(busyPort);
    expect(typeof allocatedPort).toBe("number");
    expect(allocatedPort).toBeGreaterThan(0);
  });

  it("returns OS-assigned port when preferredPort is 0", async () => {
    const port = await findFreePort(0, "127.0.0.1");
    expect(typeof port).toBe("number");
    expect(port).toBeGreaterThan(0);
  });
});

describe("findFreePorts", () => {
  const servers: net.Server[] = [];

  afterEach(() => {
    for (const server of servers) {
      server.close();
    }
    servers.length = 0;
  });

  it("allocates all requested ports when all preferred are available", async () => {
    // Use high ports unlikely to be in use
    const result = await findFreePorts([
      { name: "portA", preferred: 19891 },
      { name: "portB", preferred: 19892 },
    ]);

    // Check ports are valid - they may be the preferred or fallbacks
    expect(typeof result.portA).toBe("number");
    expect(result.portA).toBeGreaterThan(0);
    expect(typeof result.portB).toBe("number");
    expect(result.portB).toBeGreaterThan(0);
    // Ports should be different
    expect(result.portA).not.toBe(result.portB);
  });

  it("returns unique ports for each name", async () => {
    // Use preferred: 0 to get OS-assigned ports
    const result = await findFreePorts([
      { name: "port1", preferred: 0 },
      { name: "port2", preferred: 0 },
      { name: "port3", preferred: 0 },
    ]);

    const ports = [result.port1, result.port2, result.port3];
    const uniquePorts = new Set(ports);

    expect(uniquePorts.size).toBe(3);
    for (const port of ports) {
      expect(typeof port).toBe("number");
      expect(port).toBeGreaterThan(0);
    }
  });

  it("falls back when preferred port is busy", async () => {
    // Create a server that holds a port
    const busyPort = await new Promise<number>((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, "127.0.0.1", () => {
        const addr = server.address();
        if (addr && typeof addr === "object") {
          servers.push(server);
          resolve(addr.port);
        } else {
          server.close();
          reject(new Error("Failed to get server address"));
        }
      });
    });

    const result = await findFreePorts([
      { name: "busy", preferred: busyPort },
      { name: "free", preferred: 19800 }, // high port unlikely to be busy
    ]);

    // "busy" should get a different port since it's taken
    expect(result.busy).not.toBe(busyPort);
    expect(typeof result.busy).toBe("number");
    expect(result.busy).toBeGreaterThan(0);

    // "free" should get the requested port if available
    // (or a fallback if 19800 happens to be busy)
    expect(typeof result.free).toBe("number");
    expect(result.free).toBeGreaterThan(0);
  });
});

describe("buildSafeDevConfigAsync", () => {
  const servers: net.Server[] = [];
  let keyTmp: string | null = null;

  afterEach(() => {
    for (const server of servers) {
      server.close();
    }
    servers.length = 0;
    if (keyTmp) {
      rmSync(keyTmp, { recursive: true, force: true });
      keyTmp = null;
    }
    resetPersistedSessionApiKeyCache();
  });

  function tempKeyPath(): string {
    keyTmp = mkdtempSync(path.join(tmpdir(), "dev-safe-async-key-"));
    return path.join(keyTmp, "session-api-key.txt");
  }

  it("returns config with dynamically allocated ports", async () => {
    // Use a high port so the assertPortsFree check passes even when a real
    // dev stack is running on the default port (18000).
    const config = await buildSafeDevConfigAsync(repoRoot, {
      OH_CANVAS_SAFE_BACKEND_PORT: "19800",
      OH_SESSION_API_KEY_PATH: tempKeyPath(),
    });

    expect(typeof config.backendPort).toBe("number");
    expect(config.backendPort).toBeGreaterThan(0);
    expect(typeof config.vscodePort).toBe("number");
    expect(config.vscodePort).toBeGreaterThan(0);
    // Ports should be different
    expect(config.backendPort).not.toBe(config.vscodePort);
  });

  it("throws when preferred port is busy", async () => {
    // Block a specific high port we'll request
    const busyPort = 19600;
    const server = net.createServer();
    await new Promise<void>((resolve, reject) => {
      server.listen(busyPort, "127.0.0.1", () => {
        servers.push(server);
        resolve();
      });
      server.on("error", reject);
    });

    // Request the busy port via env var — should throw instead of falling back
    await expect(
      buildSafeDevConfigAsync(repoRoot, {
        OH_CANVAS_SAFE_BACKEND_PORT: busyPort.toString(),
        OH_SESSION_API_KEY_PATH: tempKeyPath(),
      }),
    ).rejects.toThrow(/agent-server.*port 19600/i);
  });
});

describe("assertPortsFree", () => {
  const servers: net.Server[] = [];

  afterEach(() => {
    for (const server of servers) {
      server.close();
    }
    servers.length = 0;
  });

  it("resolves when all ports are free", async () => {
    // High ports unlikely to be in use
    await expect(
      assertPortsFree([
        { name: "svc-a", port: 19700 },
        { name: "svc-b", port: 19701 },
      ]),
    ).resolves.toBeUndefined();
  });

  it("throws when a single port is busy", async () => {
    const busyPort = await new Promise<number>((resolve, reject) => {
      const server = net.createServer();
      server.listen(0, "127.0.0.1", () => {
        const addr = server.address();
        if (addr && typeof addr === "object") {
          servers.push(server);
          resolve(addr.port);
        } else {
          server.close();
          reject(new Error("Failed to get address"));
        }
      });
    });

    await expect(
      assertPortsFree([{ name: "agent-server", port: busyPort }]),
    ).rejects.toThrow(/agent-server.*port/i);
  });

  it("names all busy ports in the error message", async () => {
    const [portA, portB] = await Promise.all(
      [0, 0].map(
        () =>
          new Promise<number>((resolve, reject) => {
            const server = net.createServer();
            server.listen(0, "127.0.0.1", () => {
              const addr = server.address();
              if (addr && typeof addr === "object") {
                servers.push(server);
                resolve(addr.port);
              } else {
                server.close();
                reject(new Error("Failed to get address"));
              }
            });
          }),
      ),
    );

    await expect(
      assertPortsFree([
        { name: "ingress", port: portA },
        { name: "vite", port: portB },
      ]),
    ).rejects.toThrow(/ingress.*vite|vite.*ingress/is);
  });
});

describe("frontend dependency preflight", () => {
  let tempRoot: string | null = null;

  afterEach(() => {
    if (tempRoot) {
      rmSync(tempRoot, { recursive: true, force: true });
      tempRoot = null;
    }
  });

  function makeTempRoot(): string {
    tempRoot = mkdtempSync(path.join(tmpdir(), "frontend-deps-"));
    return tempRoot;
  }

  function writeBin(root: string, name: string): void {
    const binDir = path.join(root, "node_modules", ".bin");
    mkdirSync(binDir, { recursive: true });
    writeFileSync(path.join(binDir, name), "#!/bin/sh\n");
  }

  it("reports required npm binaries when node_modules has not been installed", () => {
    const root = makeTempRoot();

    expect(getMissingFrontendDependencyBins(root, "linux")).toEqual([
      "cross-env",
      "react-router",
    ]);
  });

  it("passes when required npm binary shims exist", () => {
    const root = makeTempRoot();
    writeBin(root, "cross-env");
    writeBin(root, "react-router");

    expect(getMissingFrontendDependencyBins(root, "linux")).toEqual([]);
    expect(() => validateFrontendDependencies(root, "linux")).not.toThrow();
  });

  it("accepts Windows command shims", () => {
    const root = makeTempRoot();
    writeBin(root, "cross-env.cmd");
    writeBin(root, "react-router.cmd");

    expect(getMissingFrontendDependencyBins(root, "win32")).toEqual([]);
  });

  it("formats an actionable npm ci message", () => {
    const guidance = formatMissingFrontendDependenciesGuidance(
      ["cross-env"],
      "/workspace/project/agent-canvas",
    );

    expect(guidance).toContain("Frontend dependencies are not installed");
    expect(guidance).toContain("Missing npm binaries: cross-env");
    expect(guidance).toContain("npm ci");
    expect(guidance).toContain("/workspace/project/agent-canvas");
  });
});

describe("formatMissingUvxGuidance", () => {
  it("includes install, PATH, README, and fallback workflow hints", () => {
    const guidance = formatMissingUvxGuidance(
      "/workspace/project/agent-canvas",
    );

    expect(guidance).toContain(
      "curl -LsSf https://astral.sh/uv/install.sh | sh",
    );
    expect(guidance).toContain('export PATH="$HOME/.local/bin:$PATH"');
    expect(guidance).toContain("command -v uvx");
    expect(guidance).toContain(
      path.join("/workspace/project/agent-canvas", "README.md"),
    );
    expect(guidance).toContain(
      "https://docs.astral.sh/uv/getting-started/installation/",
    );
    expect(guidance).toContain("npm run dev:frontend");
    expect(guidance).toContain("npm run dev:mock");
  });
});

describe("buildAgentServerTelemetryEnv", () => {
  it("configures PostHog telemetry by default without seeding consent", () => {
    expect(buildAgentServerTelemetryEnv({})).toEqual({
      OH_TELEMETRY_EXPORTER: "posthog",
      OH_TELEMETRY_POSTHOG_API_KEY:
        "phc_kBtz5nKmxVRRQ7HtPwr2QX9eMC5j65zE86QKocVNwb4U",
      OH_TELEMETRY_POSTHOG_HOST: "https://us.i.posthog.com",
    });
  });

  it("prefers explicit agent-server telemetry settings", () => {
    expect(
      buildAgentServerTelemetryEnv({
        OH_TELEMETRY_EXPORTER: "http",
        OH_TELEMETRY_CONSENT: "denied",
        OH_TELEMETRY_POSTHOG_API_KEY: "phc_agent",
        OH_TELEMETRY_POSTHOG_HOST: "https://agent.example",
        VITE_POSTHOG_API_KEY: "phc_frontend",
        VITE_POSTHOG_HOST: "https://frontend.example",
      }),
    ).toEqual({
      OH_TELEMETRY_EXPORTER: "http",
      OH_TELEMETRY_CONSENT: "denied",
      OH_TELEMETRY_POSTHOG_API_KEY: "phc_agent",
      OH_TELEMETRY_POSTHOG_HOST: "https://agent.example",
    });
  });

  it("uses frontend telemetry settings when agent-server settings are absent", () => {
    expect(
      buildAgentServerTelemetryEnv({
        VITE_POSTHOG_API_KEY: "phc_frontend",
        VITE_POSTHOG_HOST: "https://frontend.example",
      }),
    ).toEqual({
      OH_TELEMETRY_EXPORTER: "posthog",
      OH_TELEMETRY_POSTHOG_API_KEY: "phc_frontend",
      OH_TELEMETRY_POSTHOG_HOST: "https://frontend.example",
    });
  });

  it("maps frontend do-not-track to the agent-server kill switch", () => {
    expect(buildAgentServerTelemetryEnv({ VITE_DO_NOT_TRACK: "1" })).toEqual({
      DO_NOT_TRACK: "1",
    });
  });

  it("includes telemetry defaults in the full agent-server environment", () => {
    const env = buildAgentServerEnv(
      {
        cwd: "/tmp/cwd",
        backendPort: 18000,
        tmuxTmpDir: "/tmp/tmux",
        stateDir: "/tmp/state",
        conversationsPath: "/tmp/conversations",
        workspacesPath: "/tmp/workspaces",
        bashEventsDir: "/tmp/bash-events",
        vscodePort: 19000,
        vscodeBasePath: "/vscode",
        secretKey: "secret",
        sessionApiKey: "session",
        backendBaseUrl: "http://127.0.0.1:18000",
        backendHost: "127.0.0.1:18000",
        workingDir: "/tmp/workspaces",
        canvasToolsDir: "/tmp/tools",
      },
      { env: {} },
    );

    expect(env).toMatchObject({
      OH_TELEMETRY_EXPORTER: "posthog",
      OH_SESSION_API_KEYS_0: "session",
    });
  });
});

describe("buildAgentServerCommand", () => {
  it("uses released PyPI version by default with all packages pinned", () => {
    const cmd = buildAgentServerCommand({});

    expect(cmd.command).toBe("uvx");
    // Defaults to the released PyPI version with all SDK packages pinned to same version
    expect(cmd.args).toEqual([
      "--from",
      "openhands-agent-server==1.48.0",
      "--with",
      "openhands-sdk==1.48.0",
      "--with",
      "openhands-tools==1.48.0",
      "--with",
      "openhands-workspace==1.48.0",
      "--with",
      "agent-client-protocol<0.11",
      "--with",
      "posthog>=6,<7",
      "agent-server",
      "--import-modules",
      "canvas_ui_tool",
    ]);
    expect(cmd.source).toBe("PyPI (1.48.0, default)");
  });

  it("uses specific PyPI version when OH_AGENT_SERVER_VERSION is set with all packages pinned", () => {
    const cmd = buildAgentServerCommand({ OH_AGENT_SERVER_VERSION: "1.18.0" });

    expect(cmd.command).toBe("uvx");
    // Uses --from syntax because executable name (agent-server) differs from package name (openhands-agent-server)
    // All SDK packages are pinned to the same version
    expect(cmd.args).toEqual([
      "--from",
      "openhands-agent-server==1.18.0",
      "--with",
      "openhands-sdk==1.18.0",
      "--with",
      "openhands-tools==1.18.0",
      "--with",
      "openhands-workspace==1.18.0",
      "--with",
      "agent-client-protocol<0.11",
      "--with",
      "posthog>=6,<7",
      "agent-server",
      "--import-modules",
      "canvas_ui_tool",
    ]);
    expect(cmd.source).toBe("PyPI (1.18.0)");
  });

  it("uses git ref with subdirectory syntax for monorepo", () => {
    const cmd = buildAgentServerCommand({
      OH_AGENT_SERVER_GIT_REF: "feature-branch",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toEqual([
      "--reinstall",
      "--from",
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-agent-server",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-sdk",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-tools",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-workspace",
      "--with",
      "posthog>=6,<7",
      "agent-server",
      "--import-modules",
      "canvas_ui_tool",
    ]);
    expect(cmd.source).toBe("git (feature-branch)");
  });

  it("uses git ref for commit SHA", () => {
    const cmd = buildAgentServerCommand({ OH_AGENT_SERVER_GIT_REF: "abc1234" });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toEqual([
      "--reinstall",
      "--from",
      "git+https://github.com/OpenHands/software-agent-sdk@abc1234#subdirectory=openhands-agent-server",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@abc1234#subdirectory=openhands-sdk",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@abc1234#subdirectory=openhands-tools",
      "--with",
      "git+https://github.com/OpenHands/software-agent-sdk@abc1234#subdirectory=openhands-workspace",
      "--with",
      "posthog>=6,<7",
      "agent-server",
      "--import-modules",
      "canvas_ui_tool",
    ]);
    expect(cmd.source).toBe("git (abc1234)");
  });

  it("git ref takes precedence over version", () => {
    const cmd = buildAgentServerCommand({
      OH_AGENT_SERVER_VERSION: "1.18.0",
      OH_AGENT_SERVER_GIT_REF: "feature-branch",
    });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toContain("--from");
    expect(cmd.args).toContain(
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-agent-server",
    );
    expect(cmd.args).not.toContain("openhands-agent-server==1.18.0");
  });

  it("uses local path with editable workspace packages when OH_AGENT_SERVER_LOCAL_PATH is set", () => {
    const sdk = "/abs/path/to/software-agent-sdk";
    const cmd = buildAgentServerCommand({ OH_AGENT_SERVER_LOCAL_PATH: sdk });

    expect(cmd.command).toBe("uvx");
    expect(cmd.args).toEqual([
      "--reinstall",
      "--from",
      path.join(sdk, "openhands-agent-server"),
      "--with-editable",
      path.join(sdk, "openhands-sdk"),
      "--with-editable",
      path.join(sdk, "openhands-tools"),
      "--with-editable",
      path.join(sdk, "openhands-workspace"),
      "--with",
      "posthog>=6,<7",
      "agent-server",
      "--import-modules",
      "canvas_ui_tool",
    ]);
    expect(cmd.source).toBe(`local (${sdk})`);
  });

  it("local path takes precedence over git ref and version", () => {
    const sdk = "/abs/path/to/software-agent-sdk";
    const cmd = buildAgentServerCommand({
      OH_AGENT_SERVER_LOCAL_PATH: sdk,
      OH_AGENT_SERVER_GIT_REF: "feature-branch",
      OH_AGENT_SERVER_VERSION: "1.18.0",
    });

    expect(cmd.source).toBe(`local (${sdk})`);
    expect(cmd.args).toContain(path.join(sdk, "openhands-agent-server"));
    expect(cmd.args).not.toContain(
      "git+https://github.com/OpenHands/software-agent-sdk@feature-branch#subdirectory=openhands-agent-server",
    );
    expect(cmd.args).not.toContain("openhands-agent-server==1.18.0");
  });

  it("passes --import-modules to the agent-server, after the executable, in every source mode", () => {
    // The flag must sit after "agent-server" so uvx hands it to the server
    // instead of parsing it itself. tools/canvas_ui_tool.py documents why the
    // module has to be imported before any conversation is created.
    const variants = [
      {},
      { OH_AGENT_SERVER_VERSION: "1.18.0" },
      { OH_AGENT_SERVER_GIT_REF: "feature-branch" },
      { OH_AGENT_SERVER_LOCAL_PATH: "/abs/path/to/software-agent-sdk" },
    ];
    for (const env of variants) {
      const { args } = buildAgentServerCommand(env);
      const executable = args.indexOf("agent-server");
      expect(executable).toBeGreaterThan(-1);
      expect(args.slice(executable + 1)).toEqual([
        "--import-modules",
        "canvas_ui_tool",
      ]);
    }
  });

  it("rejects relative OH_AGENT_SERVER_LOCAL_PATH", () => {
    expect(() =>
      buildAgentServerCommand({
        OH_AGENT_SERVER_LOCAL_PATH: "./software-agent-sdk",
      }),
    ).toThrow(/must be an absolute path/);
  });
});

describe("validateLocalAgentServerPath", () => {
  it("passes when all four workspace packages exist", () => {
    const tmp = mkdtempSync(path.join(tmpdir(), "sdk-"));
    try {
      for (const subdir of [
        "openhands-agent-server",
        "openhands-sdk",
        "openhands-tools",
        "openhands-workspace",
      ]) {
        mkdirSync(path.join(tmp, subdir));
      }
      expect(() => validateLocalAgentServerPath(tmp)).not.toThrow();
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("throws when the path does not exist", () => {
    expect(() =>
      validateLocalAgentServerPath("/definitely/does/not/exist/sdk"),
    ).toThrow(/does not exist/);
  });

  it("throws when a workspace package subdirectory is missing", () => {
    const tmp = mkdtempSync(path.join(tmpdir(), "sdk-"));
    try {
      mkdirSync(path.join(tmp, "openhands-agent-server"));
      mkdirSync(path.join(tmp, "openhands-sdk"));
      mkdirSync(path.join(tmp, "openhands-tools"));
      // openhands-workspace is intentionally absent
      expect(() => validateLocalAgentServerPath(tmp)).toThrow(
        /openhands-workspace/,
      );
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("throws when given a relative path", () => {
    expect(() => validateLocalAgentServerPath("./sdk")).toThrow(
      /must be an absolute path/,
    );
  });
});

describe("buildSafeDevConfig", () => {
  let keyTmp: string | null = null;

  afterEach(() => {
    if (keyTmp) {
      rmSync(keyTmp, { recursive: true, force: true });
      keyTmp = null;
    }
    resetPersistedSessionApiKeyCache();
  });

  function tempKeyPath(): string {
    keyTmp = mkdtempSync(path.join(tmpdir(), "dev-safe-key-"));
    return path.join(keyTmp, "session-api-key.txt");
  }

  it("builds isolated default paths and ports", () => {
    const cwd = "/workspace/project/agent-canvas";

    const config = buildSafeDevConfig(cwd, {
      OH_SESSION_API_KEY_PATH: tempKeyPath(),
    });

    expect(config.backendPort).toBe(18000);
    expect(config.vscodePort).toBe(18001);
    expect(config.backendBaseUrl).toBe("http://127.0.0.1:18000");
    expect(config.backendHost).toBe("127.0.0.1:18000");
    expect(config.workingDir).toBe(config.workspacesPath);
    expect(config.stateDir).toBe(
      path.join(homedir(), ".openhands", "agent-canvas"),
    );
    expect(config.tmuxTmpDir).toBe(path.join(config.stateDir, "tmux"));
    expect(config.conversationsPath).toBe(
      path.join(config.stateDir, "dev_conversations"),
    );
    expect(config.workspacesPath).toBe(
      path.join(config.stateDir, "workspaces"),
    );
    expect(config.bashEventsDir).toBe(
      path.join(config.stateDir, "bash_events"),
    );
  });

  it("honors environment overrides", () => {
    const cwd = "/workspace/project/agent-canvas";

    const config = buildSafeDevConfig(cwd, {
      OH_CANVAS_SAFE_BACKEND_PORT: "19000",
      OH_CANVAS_SAFE_VSCODE_PORT: "19010",
      OH_CANVAS_SAFE_STATE_DIR: ".tmp/dev-safe",
      VITE_WORKING_DIR: "/workspace/custom-repo",
      OH_SESSION_API_KEY_PATH: tempKeyPath(),
    });

    expect(config.backendPort).toBe(19000);
    expect(config.vscodePort).toBe(19010);
    expect(config.backendBaseUrl).toBe("http://127.0.0.1:19000");
    expect(config.backendHost).toBe("127.0.0.1:19000");
    expect(config.stateDir).toBe(path.resolve(cwd, ".tmp", "dev-safe"));
    expect(config.workingDir).toBe("/workspace/custom-repo");
    // tmux socket dir defaults to <stateDir>/tmux.
    expect(config.tmuxTmpDir).toBe(path.join(config.stateDir, "tmux"));
  });

  it("honors TMUX_TMPDIR for hosts without socket-capable homes", () => {
    const config = buildSafeDevConfig("/workspace/project/agent-canvas", {
      TMUX_TMPDIR: "/tmp",
      OH_SESSION_API_KEY_PATH: tempKeyPath(),
    });

    expect(config.tmuxTmpDir).toBe("/tmp");
  });

  it("falls back to the persisted session key file when no env override is set", () => {
    const keyPath = tempKeyPath();
    const config = buildSafeDevConfig("/workspace/project/agent-canvas", {
      OH_SESSION_API_KEY_PATH: keyPath,
    });

    // A fresh hex key was generated and persisted.
    expect(config.sessionApiKey).toMatch(/^[a-f0-9]{64}$/);
    expect(readFileSync(keyPath, "utf8").trim()).toBe(config.sessionApiKey);
  });

  it("reuses the same key across config builds, simulating restarts", () => {
    const keyPath = tempKeyPath();

    const first = buildSafeDevConfig("/workspace/project/agent-canvas", {
      OH_SESSION_API_KEY_PATH: keyPath,
    });

    // Simulate a fresh process by clearing the in-memory cache; the file
    // on disk is what should make the key stable.
    resetPersistedSessionApiKeyCache();

    const second = buildSafeDevConfig("/workspace/project/agent-canvas", {
      OH_SESSION_API_KEY_PATH: keyPath,
    });

    expect(second.sessionApiKey).toBe(first.sessionApiKey);
  });

  it("LOCAL_BACKEND_API_KEY takes precedence over the persisted file", () => {
    const keyPath = tempKeyPath();
    // Pre-seed the file with one key.
    mkdirSync(path.dirname(keyPath), { recursive: true });
    writeFileSync(keyPath, "persisted-key-value\n");

    const config = buildSafeDevConfig("/workspace/project/agent-canvas", {
      LOCAL_BACKEND_API_KEY: "env-key-wins",
      OH_SESSION_API_KEY_PATH: keyPath,
    });

    expect(config.sessionApiKey).toBe("env-key-wins");
    // The file is left untouched.
    expect(readFileSync(keyPath, "utf8").trim()).toBe("persisted-key-value");
  });
});

describe("getOrCreatePersistedSessionApiKey", () => {
  let dir: string | null = null;

  afterEach(() => {
    if (dir) {
      rmSync(dir, { recursive: true, force: true });
      dir = null;
    }
    resetPersistedSessionApiKeyCache();
  });

  function tempPath(): string {
    dir = mkdtempSync(path.join(tmpdir(), "session-key-"));
    return path.join(dir, "nested", "session-api-key.txt");
  }

  it("creates the file (and parent dirs) with a hex key on first call", () => {
    const filePath = tempPath();
    const key = getOrCreatePersistedSessionApiKey(filePath);

    expect(key).toMatch(/^[a-f0-9]{64}$/);
    expect(readFileSync(filePath, "utf8").trim()).toBe(key);
  });

  it("returns the existing key on subsequent calls (after cache reset)", () => {
    const filePath = tempPath();
    const first = getOrCreatePersistedSessionApiKey(filePath);

    resetPersistedSessionApiKeyCache();

    const second = getOrCreatePersistedSessionApiKey(filePath);
    expect(second).toBe(first);
  });

  it("trims surrounding whitespace from the persisted file", () => {
    const filePath = tempPath();
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, "  abcdef1234  \n");

    const key = getOrCreatePersistedSessionApiKey(filePath);
    expect(key).toBe("abcdef1234");
  });

  it("regenerates and overwrites when the file is empty", () => {
    const filePath = tempPath();
    mkdirSync(path.dirname(filePath), { recursive: true });
    writeFileSync(filePath, "   \n");

    const key = getOrCreatePersistedSessionApiKey(filePath);
    expect(key).toMatch(/^[a-f0-9]{64}$/);
    expect(readFileSync(filePath, "utf8").trim()).toBe(key);
  });
});

describe("buildNpmScriptCommand", () => {
  it("runs npm through cmd.exe on Windows even when npm_execpath is set", () => {
    // npm_execpath points to a path with spaces like
    // "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js".
    // spawnService uses shell:true on Windows, so passing that path as an
    // argument causes cmd.exe to split on the space and fail with
    // "'C:\Program' is not recognized as an internal or external command".
    // The win32 branch must fire BEFORE the npm_execpath branch.
    const command = buildNpmScriptCommand(
      "dev:frontend",
      "win32",
      {
        ComSpec: "C:\\Windows\\System32\\cmd.exe",
        npm_execpath:
          "C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js",
        npm_node_execpath: "C:\\Program Files\\nodejs\\node.exe",
      },
      "C:\\Program Files\\nodejs\\node.exe",
    );

    expect(command).toEqual({
      command: "C:\\Windows\\System32\\cmd.exe",
      args: ["/d", "/s", "/c", "npm", "run", "dev:frontend"],
    });
  });

  it("reuses npm's own CLI path when available on POSIX", () => {
    const command = buildNpmScriptCommand(
      "dev:frontend",
      "linux",
      {
        npm_execpath: "/usr/lib/node_modules/npm/bin/npm-cli.js",
        npm_node_execpath: "/usr/bin/node",
      },
      "/fallback/node",
    );

    expect(command).toEqual({
      command: "/usr/bin/node",
      args: ["/usr/lib/node_modules/npm/bin/npm-cli.js", "run", "dev:frontend"],
    });
  });

  it("runs npm directly on POSIX platforms", () => {
    const command = buildNpmScriptCommand("dev:frontend", "linux", {});

    expect(command).toEqual({
      command: "npm",
      args: ["run", "dev:frontend"],
    });
  });

  it("runs npm through cmd.exe on Windows", () => {
    const command = buildNpmScriptCommand("dev:frontend", "win32", {
      ComSpec: "C:\\Windows\\System32\\cmd.exe",
    });

    expect(command).toEqual({
      command: "C:\\Windows\\System32\\cmd.exe",
      args: ["/d", "/s", "/c", "npm", "run", "dev:frontend"],
    });
  });

  it("falls back to cmd.exe when ComSpec is unavailable on Windows", () => {
    const command = buildNpmScriptCommand("dev:frontend", "win32", {});

    expect(command).toEqual({
      command: "cmd.exe",
      args: ["/d", "/s", "/c", "npm", "run", "dev:frontend"],
    });
  });
});

describe("dev-safe CLI startup", () => {
  it("exits promptly when uvx is missing", async () => {
    // Skip this test if uvx is globally installed via /usr/local/bin symlink
    // that may still be accessible even with a stripped PATH
    const child = spawn(process.execPath, ["scripts/dev-safe.mjs"], {
      cwd: repoRoot,
      env: {
        // Use empty PATH to ensure uvx is not found.
        PATH: "",
        // Redirect the agent-server port to a high free port so the
        // assertPortsFree pre-flight check passes when a real dev stack is
        // running on the default port (18000) — the test is about uvx, not
        // port detection.
        OH_CANVAS_SAFE_BACKEND_PORT: "19810",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      output += chunk.toString();
    });

    const exitResult = await Promise.race([
      once(child, "exit").then(([code, signal]) => ({
        code,
        signal,
        timedOut: false,
      })),
      delay(4_000).then(() => ({ code: null, signal: null, timedOut: true })),
    ]);

    if (exitResult.timedOut) {
      child.kill("SIGKILL");
    }

    expect(exitResult.timedOut).toBe(false);
    expect(exitResult.code).toBe(1);
    expect(output).toContain("Failed to start uvx");
    expect(output).toContain("curl -LsSf https://astral.sh/uv/install.sh | sh");
    expect(output).toContain(
      "https://docs.astral.sh/uv/getting-started/installation/",
    );
    expect(output).toContain("README.md");
    expect(output).toContain("npm run dev:mock");
    expect(output).toContain("spawn uvx ENOENT");
  });

  it.skipIf(process.platform === "win32")(
    "cleans up the detached agent-server when the launcher receives SIGHUP",
    async () => {
      // Services are spawned detached (getProcessTreeSpawnOptions), so killing
      // the launcher does not kill them. Drive the real launcher with a stub
      // agent-server, then SIGHUP the launcher and assert the stub's port is
      // released rather than held by a survivor.
      const stubDir = mkdtempSync(path.join(tmpdir(), "dev-safe-sighup-"));
      const stubJs = path.join(stubDir, "stub-agent-server.mjs");
      const uvxStub = path.join(stubDir, "uvx");

      writeFileSync(
        stubJs,
        [
          'import net from "node:net";',
          'const portIndex = process.argv.indexOf("--port");',
          "const port = Number(process.argv[portIndex + 1]);",
          "const server = net.createServer(() => {});",
          'server.listen(port, "127.0.0.1", () => {',
          '  console.log("STUB_LISTENING", process.pid, port);',
          "});",
          "setInterval(() => {}, 1_000);",
        ].join("\n"),
      );
      writeFileSync(
        uvxStub,
        `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} ${JSON.stringify(stubJs)} "$@"\n`,
      );
      chmodSync(uvxStub, 0o755);

      const backendPort = await findFreePort(0);
      const isPortListening = async () =>
        new Promise<boolean>((resolve) => {
          const socket = net
            .connect(backendPort, "127.0.0.1")
            .on("connect", () => {
              socket.destroy();
              resolve(true);
            })
            .on("error", () => resolve(false));
        });

      const launcher = spawn(process.execPath, ["scripts/dev-safe.mjs"], {
        cwd: repoRoot,
        env: {
          ...process.env,
          PATH: `${stubDir}${path.delimiter}${process.env.PATH ?? ""}`,
          OH_CANVAS_SAFE_BACKEND_PORT: String(backendPort),
        },
        stdio: ["ignore", "pipe", "pipe"],
      });

      let output = "";
      let stubPid: number | undefined;
      const capture = (chunk: Buffer) => {
        output += chunk.toString();
        const stubMatch = output.match(/STUB_LISTENING (\d+)/);
        if (stubMatch) stubPid = Number(stubMatch[1]);
      };
      launcher.stdout.on("data", capture);
      launcher.stderr.on("data", capture);

      try {
        const readyDeadline = Date.now() + 20_000;
        let listening = false;
        while (!listening && Date.now() < readyDeadline) {
          if (launcher.exitCode !== null) break;
          listening = await isPortListening();
          if (!listening) await delay(100);
        }
        expect(listening, output).toBe(true);

        launcher.kill("SIGHUP");
        await Promise.race([once(launcher, "exit"), delay(10_000)]);

        // shutdown() forwards SIGTERM, then SIGKILLs after 3s.
        const freeDeadline = Date.now() + 12_000;
        let stillListening = true;
        while (stillListening && Date.now() < freeDeadline) {
          stillListening = await isPortListening();
          if (stillListening) await delay(200);
        }
        expect(stillListening, output).toBe(false);
      } finally {
        if (launcher.exitCode === null) launcher.kill("SIGKILL");
        // The stub is a detached process-group leader, so killing the launcher
        // does not reap it. Without this, the regression path this test exists
        // to catch would itself leave the stub holding its port indefinitely.
        if (stubPid !== undefined) {
          try {
            process.kill(-stubPid, "SIGKILL");
          } catch {
            // Already gone, which is the passing path.
          }
        }
        rmSync(stubDir, { recursive: true, force: true });
      }
    },
    45_000,
  );
});

interface RuntimeServiceEntry {
  kind?: string;
  description?: string;
  url_from_agent?: string;
  api_prefix?: string;
  docs_url?: string;
  openapi_url?: string;
  auth_env_var?: string;
}
interface RuntimeServicesInfoShape {
  mode: string;
  agent_host_alias: string;
  services: {
    agent_server?: RuntimeServiceEntry;
    ingress?: RuntimeServiceEntry;
    frontend?: RuntimeServiceEntry;
    automation?: RuntimeServiceEntry;
  };
}

describe("buildRuntimeServicesInfo", () => {
  it("describes only the agent-server in a minimal dev-safe stack", () => {
    const info = buildRuntimeServicesInfo({
      mode: "dev:safe",
      agentServerPort: 18000,
    }) as RuntimeServicesInfoShape;
    expect(info).toEqual({
      mode: "dev:safe",
      agent_host_alias: "localhost",
      services: {
        agent_server: {
          description: expect.any(String),
          url_from_agent: "http://localhost:18000",
        },
      },
    });
  });

  it("includes ingress, frontend (vite), and automation entries when ports are provided", () => {
    const info = buildRuntimeServicesInfo({
      mode: "dev:automation",
      agentServerPort: 18000,
      ingressPort: 8000,
      frontendPort: 3001,
      automation: { port: 18001 },
    }) as RuntimeServicesInfoShape;
    expect(info.services.ingress?.url_from_agent).toBe("http://localhost:8000");
    expect(info.services.frontend).toMatchObject({
      kind: "vite",
      url_from_agent: "http://localhost:3001",
    });
    expect(info.services.frontend?.description).toMatch(/Vite dev server/i);
    expect(info.services.automation).toMatchObject({
      url_from_agent: "http://localhost:18001",
      api_prefix: "/api/automation",
      docs_url: "http://localhost:18001/api/automation/docs",
      openapi_url: "http://localhost:18001/api/automation/openapi.json",
      auth_env_var: "OPENHANDS_AUTOMATION_API_KEY",
    });
  });

  it("supports a custom agent host alias for remote setups", () => {
    const info = buildRuntimeServicesInfo({
      mode: "custom",
      agentHostAlias: "custom-host",
      agentServerPort: 8000,
      ingressPort: 8000,
      frontendPort: 3001,
      frontendKind: "static",
      automation: { port: 18001 },
    }) as RuntimeServicesInfoShape;
    // Agent-server URL is always localhost (the agent is *inside* it).
    expect(info.services.agent_server?.url_from_agent).toBe(
      "http://localhost:8000",
    );
    // Host-side services use the custom alias.
    expect(info.services.ingress?.url_from_agent).toBe(
      "http://custom-host:8000",
    );
    expect(info.services.frontend?.url_from_agent).toBe(
      "http://custom-host:3001",
    );
    // Static-mode description, not "Vite dev server".
    expect(info.services.frontend?.kind).toBe("static");
    expect(info.services.frontend?.description).toMatch(/Static-file server/i);
    expect(info.services.frontend?.description).not.toMatch(/Vite/i);
    expect(info.services.automation?.url_from_agent).toBe(
      "http://custom-host:18001",
    );
  });

  it("allows overriding the api prefix and auth env var", () => {
    const info = buildRuntimeServicesInfo({
      mode: "dev:custom",
      agentServerPort: 18000,
      automation: {
        port: 9000,
        apiPrefix: "/v2/auto",
        authEnvVar: "MY_KEY",
      },
    }) as RuntimeServicesInfoShape;
    expect(info.services.automation).toMatchObject({
      api_prefix: "/v2/auto",
      docs_url: "http://localhost:9000/v2/auto/docs",
      openapi_url: "http://localhost:9000/v2/auto/openapi.json",
      auth_env_var: "MY_KEY",
    });
  });

  it("omits the automation entry when none is provided", () => {
    const info = buildRuntimeServicesInfo({
      mode: "dev:safe",
      agentServerPort: 18000,
      ingressPort: 8000,
    }) as RuntimeServicesInfoShape;
    expect(info.services.automation).toBeUndefined();
  });

  it("omits the automation entry when the object lacks a port", () => {
    // A bare `{}` previously slipped through and produced
    // `http://localhost:undefined`; require the port explicitly.
    const info = buildRuntimeServicesInfo({
      mode: "dev:safe",
      agentServerPort: 18000,
      automation: {},
    }) as RuntimeServicesInfoShape;
    expect(info.services.automation).toBeUndefined();
  });

  it("throws when neither agentServerPort nor agentServerUrl is given", () => {
    expect(() =>
      buildRuntimeServicesInfo({
        mode: "dev:safe",
      }),
    ).toThrow(/agentServerPort or agentServerUrl is required/);
  });

  it("accepts the legacy vitePort alias for frontendPort", () => {
    // dev-safe.mjs's `main()` and some external callers still pass the
    // older option name; keep them working for one release.
    const info = buildRuntimeServicesInfo({
      mode: "dev:safe",
      agentServerPort: 18000,
      vitePort: 3001,
    }) as RuntimeServicesInfoShape;
    expect(info.services.frontend?.url_from_agent).toBe(
      "http://localhost:3001",
    );
  });
});
