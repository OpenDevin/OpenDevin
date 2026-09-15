#!/usr/bin/env node

/**
 * Check SDK Version Sync
 *
 * Verifies two things against versions.agentServer in config/defaults.json:
 *
 * 1. The local @openhands/typescript-client pin in package.json. Canvas renders
 *    the ACP provider picker from that generated registry mirror but launches
 *    the adapter through the agent-server image, so a skew ships a picker
 *    offering models and launch commands agent-server does not implement.
 *
 * 2. That the released automation package (openhands-automation on PyPI)
 *    uses the SDK version expected for that automation release for all agent SDK libraries:
 *   - openhands-sdk
 *   - openhands-tools
 *   - openhands-workspace
 *   - openhands-agent-server
 *
 * This script checks the RELEASED PyPI version of openhands-automation (as specified
 * by versions.automation in config/defaults.json), not the main branch.
 * The expected SDK dependency version is versions.agentServer — the two must
 * always match, so this script catches any drift.
 *
 * This script is run in CI to catch version drift between projects.
 *
 * Usage:
 *   node scripts/check-sdk-version-sync.mjs
 *   EXPECTED_SDK_VERSION=1.48.0 node scripts/check-sdk-version-sync.mjs
 *   node scripts/check-sdk-version-sync.mjs --check-pypi
 *
 * Environment variables:
 *   EXPECTED_SDK_VERSION      - Override the expected version (instead of reading from config/defaults.json)
 *   AUTOMATION_PACKAGE_NAME   - Override the automation package name (default: openhands-automation)
 *   AUTOMATION_PACKAGE_VERSION - Override the automation package version (instead of reading from config/defaults.json)
 *
 * Options:
 *   --check-pypi    Also check the latest SDK version on PyPI
 *   --help          Show help
 *
 * Exit codes:
 *   0 - All SDK versions match
 *   1 - Version mismatch detected or error occurred
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

// Parse command line arguments
const args = process.argv.slice(2);
const checkPyPI = args.includes("--check-pypi");
const showHelp = args.includes("--help") || args.includes("-h");

if (showHelp) {
  console.log(`
SDK Version Sync Check

Verifies that the released openhands-automation package on PyPI uses the
SDK version expected for that automation release.

The automation version is read from config/defaults.json (versions.automation).
The expected SDK dependency version is read from versions.agentServer.

Usage:
  node scripts/check-sdk-version-sync.mjs [options]

Options:
  --check-pypi    Also check the latest SDK version on PyPI
  --help, -h      Show this help

Environment variables:
  EXPECTED_SDK_VERSION        Override the expected SDK version (instead of reading from config/defaults.json)
  AUTOMATION_PACKAGE_NAME     Override the automation package name (default: openhands-automation)
  AUTOMATION_PACKAGE_VERSION  Override the automation package version (instead of reading from config/defaults.json)

Triggering from other repos:
  The automation repo or SDK repo can trigger this check via GitHub repository_dispatch:

  curl -X POST \\
    -H "Authorization: token \$GITHUB_TOKEN" \\
    -H "Accept: application/vnd.github.v3+json" \\
    https://api.github.com/repos/OpenHands/OpenHands/dispatches \\
    -d '{"event_type": "sdk-version-check", "client_payload": {"version": "1.48.0"}}'
`);
  process.exit(0);
}

// ANSI color codes for terminal output
const colors = {
  reset: "\x1b[0m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
  dim: "\x1b[2m",
};

// SDK packages that must have matching versions
const SDK_PACKAGES = [
  "openhands-sdk",
  "openhands-tools",
  "openhands-workspace",
  "openhands-agent-server",
];

// Mirrors the SDK's ACP provider registry. Must track versions.agentServer:
// the picker is rendered from this pin but the adapter is launched by that
// image, so a skew advertises models the running agent-server cannot run.
const CLIENT_PACKAGE_NAME = "@openhands/typescript-client";

// Configurable automation package (can be overridden via env)
const AUTOMATION_PACKAGE_NAME = process.env.AUTOMATION_PACKAGE_NAME || "openhands-automation";

// Default retry configuration
const RETRY_COUNT = 3;
const RETRY_DELAY_MS = 1000;

/**
 * Normalize a version string for comparison.
 * Handles variations like "1.22" vs "1.22.0" by ensuring consistent format.
 */
function normalizeVersion(version) {
  if (!version) return null;
  
  // Remove any pre-release or build metadata for base comparison
  const baseVersion = version.split(/[-+]/)[0];
  
  // Split into parts and pad to 3 parts (major.minor.patch)
  const parts = baseVersion.split(".").map((p) => parseInt(p, 10) || 0);
  while (parts.length < 3) {
    parts.push(0);
  }
  
  return parts.slice(0, 3).join(".");
}

/**
 * Compare two versions for equality (handles semantic equivalence)
 */
function versionsEqual(v1, v2) {
  return normalizeVersion(v1) === normalizeVersion(v2);
}

/**
 * Sleep for a given number of milliseconds
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Centralized config ──────────────────────────────────────────────────────
let SHARED_DEFAULTS;
try {
  SHARED_DEFAULTS = JSON.parse(
    readFileSync(join(projectRoot, "config", "defaults.json"), "utf-8"),
  );
  if (!SHARED_DEFAULTS.versions?.agentServer) {
    throw new Error("missing required field: versions.agentServer");
  }
} catch (err) {
  console.error(`${colors.red}Failed to load config/defaults.json: ${err.message}${colors.reset}`);
  console.error("Ensure the file exists and contains valid JSON with required fields.");
  process.exit(1);
}

/**
 * Read the expected automation SDK dependency version from environment
 * or config/defaults.json.
 */
function getExpectedVersion() {
  // Allow override via environment variable (useful for CI triggers).
  const envVersion = process.env.EXPECTED_SDK_VERSION;
  if (envVersion && envVersion.trim()) {
    return { version: envVersion.trim(), source: "EXPECTED_SDK_VERSION env var" };
  }

  return {
    version: SHARED_DEFAULTS.versions.agentServer,
    source: "config/defaults.json (versions.agentServer)",
  };
}

/**
 * Compare the local typescript-client pin against the expected SDK version.
 * Returns a mismatch descriptor, or null when they agree.
 */
function findClientPinMismatch(pinnedVersion, expectedVersion) {
  if (!pinnedVersion) {
    return { package: CLIENT_PACKAGE_NAME, expected: expectedVersion, actual: null };
  }
  // A range would reintroduce the skew this check exists to catch.
  if (!/^[0-9]/.test(pinnedVersion)) {
    return { package: CLIENT_PACKAGE_NAME, expected: expectedVersion, actual: pinnedVersion };
  }
  if (versionsEqual(pinnedVersion, expectedVersion)) {
    return null;
  }
  return { package: CLIENT_PACKAGE_NAME, expected: expectedVersion, actual: pinnedVersion };
}

/**
 * Read the typescript-client pin from package.json.
 */
function readClientPin() {
  const pkg = JSON.parse(
    readFileSync(join(projectRoot, "package.json"), "utf-8"),
  );
  return pkg.dependencies?.[CLIENT_PACKAGE_NAME] ?? null;
}

/**
 * Fetch the latest version of a package from PyPI
 */
async function fetchPyPIVersion(packageName) {
  const url = `https://pypi.org/pypi/${packageName}/json`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    return data.info?.version || null;
  } catch {
    return null;
  }
}

/**
 * Read the automation version from env var or config/defaults.json
 */
function getAutomationVersion() {
  // Allow override via environment variable
  const envVersion = process.env.AUTOMATION_PACKAGE_VERSION;
  if (envVersion && envVersion.trim()) {
    return { version: envVersion.trim(), source: "AUTOMATION_PACKAGE_VERSION env var" };
  }

  return {
    version: SHARED_DEFAULTS.versions.automation,
    source: "config/defaults.json (versions.automation)",
  };
}

/**
 * Fetch package metadata from PyPI and extract dependencies (with retry)
 */
async function fetchPyPIDependencies(packageName, version) {
  const url = `https://pypi.org/pypi/${packageName}/${version}/json`;

  console.log(`${colors.dim}Fetching ${url}${colors.reset}`);

  let lastError;
  for (let attempt = 0; attempt < RETRY_COUNT; attempt++) {
    try {
      const response = await fetch(url);
      
      // 404 is a config issue, don't retry
      if (response.status === 404) {
        throw new Error(
          `Package ${packageName}==${version} not found on PyPI (404). Check the package name and version.`,
        );
      }
      
      if (!response.ok) {
        throw new Error(
          `Failed to fetch ${packageName}==${version} from PyPI: ${response.status} ${response.statusText}`,
        );
      }
      
      const data = await response.json();
      return data.info?.requires_dist || [];
    } catch (err) {
      lastError = err;
      
      // Don't retry on 404 (config issue)
      if (err.message.includes("not found on PyPI (404)")) {
        throw err;
      }
      
      // Retry on other errors (network issues, 5xx, etc.)
      if (attempt < RETRY_COUNT - 1) {
        const delay = RETRY_DELAY_MS * (attempt + 1);
        console.log(
          `${colors.yellow}Retry ${attempt + 1}/${RETRY_COUNT - 1} after ${delay}ms...${colors.reset}`,
        );
        await sleep(delay);
      }
    }
  }
  
  throw lastError;
}

/**
 * Parse PyPI requires_dist array and extract SDK package versions
 *
 * PyPI returns dependencies in PEP 508 format like:
 *   "openhands-sdk>=1.48.0,<2.0.0"
 *   "openhands-tools==1.48.0"
 *   "openhands-workspace (>=1.48.0)"
 */
function parseSdkVersionsFromRequiresDist(requiresDist) {
  const versions = {};

  for (const pkg of SDK_PACKAGES) {
    for (const dep of requiresDist) {
      // Check if the dependency starts with our package name
      // The package name may be followed by whitespace, operators, or parentheses
      if (!dep.toLowerCase().startsWith(pkg.toLowerCase())) {
        continue;
      }

      // Extract the version number - look for patterns like:
      // ">=1.48.0", "==1.48.0", "(>=1.48.0)", "~=1.48.0"
      // After the package name and before any comma or closing paren
      const versionPattern = /[><=~!]+\s*([0-9]+(?:\.[0-9]+)*)/;
      const match = dep.match(versionPattern);
      if (match) {
        versions[pkg] = match[1];
        break;
      }
    }
  }

  return versions;
}

/**
 * Main entry point
 */
async function main() {
  console.log("");
  console.log(
    `${colors.cyan}SDK Version Sync Check${colors.reset}`,
  );
  console.log("─".repeat(50));
  console.log("");

  try {
    // Get expected version from env var or config/defaults.json
    const { version: expectedVersion, source: versionSource } = getExpectedVersion();
    console.log(
      `Expected automation SDK version: ${colors.green}${expectedVersion}${colors.reset} (from ${versionSource})`,
    );

    // Offline, so it runs first and fails fast without the PyPI round trip.
    const clientMismatch = findClientPinMismatch(readClientPin(), expectedVersion);
    if (clientMismatch) {
      console.log("");
      console.log(
        `  ${CLIENT_PACKAGE_NAME.padEnd(30)} ${colors.red}✗ ${clientMismatch.actual ?? "(absent)"} (expected ${expectedVersion})${colors.reset}`,
      );
      console.log("");
      console.log(`${colors.red}Version mismatch detected!${colors.reset}`);
      console.log("");
      console.log(
        `${CLIENT_PACKAGE_NAME} mirrors the SDK's ACP provider registry that Canvas renders the`,
      );
      console.log(
        `ACP picker from, but the adapter is launched by agent-server ${expectedVersion}. A skew ships a`,
      );
      console.log("picker offering models and launch commands that agent-server does not implement.");
      console.log("");
      console.log("To fix, update one of the following:");
      console.log(`  1. Pin ${CLIENT_PACKAGE_NAME} to ${expectedVersion} in package.json`);
      console.log("  2. Update versions.agentServer in config/defaults.json");
      console.log("");
      process.exit(1);
    }
    console.log(
      `Client registry pin: ${colors.green}${CLIENT_PACKAGE_NAME}@${expectedVersion}${colors.reset} (matches versions.agentServer)`,
    );

    // Get automation version from env var or config/defaults.json
    const { version: automationVersion, source: automationSource } = getAutomationVersion();
    console.log(
      `Automation package: ${colors.cyan}${AUTOMATION_PACKAGE_NAME}==${automationVersion}${colors.reset} (from ${automationSource})`,
    );

    // Optionally check PyPI for the latest SDK version
    if (checkPyPI) {
      console.log("");
      console.log("Checking latest SDK versions on PyPI:");
      for (const pkg of SDK_PACKAGES) {
        const pypiVersion = await fetchPyPIVersion(pkg);
        if (pypiVersion) {
          const status = versionsEqual(pypiVersion, expectedVersion)
            ? colors.green
            : colors.yellow;
          console.log(`  ${pkg.padEnd(25)} ${status}${pypiVersion}${colors.reset}`);
        } else {
          console.log(`  ${pkg.padEnd(25)} ${colors.dim}(not found on PyPI)${colors.reset}`);
        }
      }
    }

    console.log("");

    // Fetch automation package dependencies from PyPI
    const requiresDist = await fetchPyPIDependencies(AUTOMATION_PACKAGE_NAME, automationVersion);
    const automationVersions = parseSdkVersionsFromRequiresDist(requiresDist);

    // Check each SDK package
    let hasErrors = false;
    let foundAny = false;
    const mismatches = [];

    console.log(`Checking ${AUTOMATION_PACKAGE_NAME}==${automationVersion} SDK dependencies:`);
    console.log("");

    for (const pkg of SDK_PACKAGES) {
      const actualVersion = automationVersions[pkg];

      if (actualVersion) {
        foundAny = true;
        if (versionsEqual(actualVersion, expectedVersion)) {
          console.log(
            `  ${pkg.padEnd(25)} ${colors.green}✓ ${actualVersion}${colors.reset}`,
          );
        } else {
          hasErrors = true;
          console.log(
            `  ${pkg.padEnd(25)} ${colors.red}✗ ${actualVersion} (expected ${expectedVersion})${colors.reset}`,
          );
          mismatches.push({
            package: pkg,
            expected: expectedVersion,
            actual: actualVersion,
          });
        }
      } else {
        // Package not found - might be a transitive dependency, not an error
        console.log(
          `  ${pkg.padEnd(25)} ${colors.dim}- not a direct dependency${colors.reset}`,
        );
      }
    }

    console.log("");

    if (!foundAny) {
      console.log(
        `${colors.yellow}Warning: No SDK packages found in ${AUTOMATION_PACKAGE_NAME}==${automationVersion} dependencies${colors.reset}`,
      );
      console.log("This might indicate a parsing issue or the package is not yet published.");
      console.log("");
      process.exit(1);
    }

    if (hasErrors) {
      console.log(
        `${colors.red}Version mismatch detected!${colors.reset}`,
      );
      console.log("");
      console.log(`The released ${AUTOMATION_PACKAGE_NAME}==${automationVersion} uses different SDK versions than expected for that automation release.`);
      console.log("");
      console.log("Mismatched packages:");
      for (const m of mismatches) {
        console.log(`  - ${m.package}: ${m.actual} (expected ${m.expected})`);
      }
      console.log("");
      console.log("To fix, update one of the following:");
      console.log(
        `  1. Release a new version of ${AUTOMATION_PACKAGE_NAME} with SDK dependencies pinned to ${expectedVersion}`,
      );
      console.log(
        `  2. Update versions.automation in config/defaults.json to a newer release`,
      );
      console.log("");
      process.exit(1);
    }

    console.log(
      `${colors.green}All SDK versions are in sync!${colors.reset}`,
    );
    console.log("");
  } catch (error) {
    console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
    process.exit(1);
  }
}

// Export for testing
export {
  normalizeVersion,
  versionsEqual,
  parseSdkVersionsFromRequiresDist,
  findClientPinMismatch,
  readClientPin,
  SDK_PACKAGES,
  CLIENT_PACKAGE_NAME,
  AUTOMATION_PACKAGE_NAME,
};

main();
