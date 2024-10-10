import { ClientActionFunctionArgs, json } from "@remix-run/react";
import {
  getDefaultSettings,
  LATEST_SETTINGS_VERSION,
  maybeMigrateSettings,
  saveSettings,
  Settings,
  settingsAreUpToDate,
} from "#/services/settings";

const requestedToEndSession = (formData: FormData) =>
  formData.get("end-session")?.toString() === "true";

const removeSessionTokenAndSelectedRepo = () => {
  const token = localStorage.getItem("token");
  const repo = localStorage.getItem("repo");

  if (token) localStorage.removeItem("token");
  if (repo) localStorage.removeItem("repo");
};

// This is the route for saving settings. It only exports the action function.
export const clientAction = async ({ request }: ClientActionFunctionArgs) => {
  const formData = await request.formData();
  const intent = formData.get("intent")?.toString();

  if (intent === "account") {
    const LANGUAGE = formData.get("language")?.toString();
    if (LANGUAGE) saveSettings({ LANGUAGE });

    return json({ success: true });
  }

  if (intent === "reset") {
    saveSettings(getDefaultSettings());
    if (requestedToEndSession(formData)) removeSessionTokenAndSelectedRepo();

    return json({ success: true });
  }

  const keys = Array.from(formData.keys());
  const isUsingAdvancedOptions = keys.includes("use-advanced-options");

  let customModel: string | undefined;
  let baseUrl: string | undefined;
  let confirmationMode = false;
  let securityAnalyzer: string | undefined;

  if (isUsingAdvancedOptions) {
    customModel = formData.get("custom-model")?.toString();
    baseUrl = formData.get("base-url")?.toString();
    confirmationMode = keys.includes("confirmation-mode");
    if (confirmationMode) {
      // only set securityAnalyzer if confirmationMode is enabled
      securityAnalyzer = formData.get("security-analyzer")?.toString();
    }
  }

  const provider = formData.get("llm-provider")?.toString();
  const model = formData.get("llm-model")?.toString();

  const LLM_MODEL = customModel || `${provider}/${model}`.toLowerCase();
  const LLM_API_KEY = formData.get("api-key")?.toString();
  const AGENT = formData.get("agent")?.toString();
  const LANGUAGE = formData.get("language")?.toString();
  const LLM_BASE_URL = baseUrl;
  const CONFIRMATION_MODE = confirmationMode;
  const SECURITY_ANALYZER = securityAnalyzer;

  const settings: Partial<Settings> = {
    LLM_MODEL,
    LLM_API_KEY,
    AGENT,
    LANGUAGE,
    LLM_BASE_URL,
    CONFIRMATION_MODE,
    SECURITY_ANALYZER,
  };

  saveSettings(settings);
  // store for settings view
  localStorage.setItem(
    "use-advanced-options",
    isUsingAdvancedOptions ? "true" : "false",
  );

  // If the settings version is different from the current version, update it.
  if (!settingsAreUpToDate()) {
    maybeMigrateSettings();
    localStorage.setItem(
      "SETTINGS_VERSION",
      LATEST_SETTINGS_VERSION.toString(),
    );
  }

  if (requestedToEndSession(formData)) removeSessionTokenAndSelectedRepo();
  return json({ success: true });
};
