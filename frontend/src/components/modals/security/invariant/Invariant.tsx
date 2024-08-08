import React, { useState, useRef, useCallback, useEffect } from "react";
import { useSelector } from "react-redux";
import { IoAlertCircle } from "react-icons/io5";
import { useTranslation } from "react-i18next";
import { Editor, Monaco } from "@monaco-editor/react";
import { editor } from "monaco-editor";
import { Button, Select, SelectItem } from "@nextui-org/react";
import { RootState } from "#/store";
import {
  ActionSecurityRisk,
  SecurityAnalyzerLog,
} from "#/state/securityAnalyzerSlice";
import { useScrollToBottom } from "#/hooks/useScrollToBottom";
import { I18nKey } from "#/i18n/declaration";
import { request } from "#/services/api";
import toast from "#/utils/toast";
import InvariantLogoIcon from "./assets/logo";

type SectionType = "logs" | "policy" | "settings";

function SecurityInvariant(): JSX.Element {
  const { t } = useTranslation();
  const { logs } = useSelector((state: RootState) => state.securityAnalyzer);
  const [activeSection, setActiveSection] = useState("logs");

  const logsRef = useRef<HTMLDivElement>(null);
  const [policy, setPolicy] = useState<string>("");
  const [selectedRisk, setSelectedRisk] = useState(ActionSecurityRisk.MEDIUM);

  useEffect(() => {
    const fetchPolicy = async () => {
      const data = await request(`/api/security/policy`);
      setPolicy(data.policy);
    };
    const fetchRiskSeverity = async () => {
      const data = await request(`/api/security/settings`);
      setSelectedRisk(
        data.RISK_SEVERITY === 0
          ? ActionSecurityRisk.LOW
          : data.RISK_SEVERITY || ActionSecurityRisk.MEDIUM,
      );
    };

    fetchPolicy();
    fetchRiskSeverity();
  }, []);

  useScrollToBottom(logsRef);

  const getRiskColor = useCallback((risk: ActionSecurityRisk) => {
    switch (risk) {
      case ActionSecurityRisk.LOW:
        return "text-green-500";
      case ActionSecurityRisk.MEDIUM:
        return "text-yellow-500";
      case ActionSecurityRisk.HIGH:
        return "text-red-500";
      case ActionSecurityRisk.UNKNOWN:
      default:
        return "text-gray-500";
    }
  }, []);

  const getRiskText = useCallback(
    (risk: ActionSecurityRisk) => {
      switch (risk) {
        case ActionSecurityRisk.LOW:
          return t(I18nKey.SECURITY_ANALYZER$LOW_RISK);
        case ActionSecurityRisk.MEDIUM:
          return t(I18nKey.SECURITY_ANALYZER$MEDIUM_RISK);
        case ActionSecurityRisk.HIGH:
          return t(I18nKey.SECURITY_ANALYZER$HIGH_RISK);
        case ActionSecurityRisk.UNKNOWN:
        default:
          return t(I18nKey.SECURITY_ANALYZER$UNKNOWN_RISK);
      }
    },
    [t],
  );

  const handleEditorDidMount = useCallback(
    (_: editor.IStandaloneCodeEditor, monaco: Monaco): void => {
      monaco.editor.defineTheme("my-theme", {
        base: "vs-dark",
        inherit: true,
        rules: [],
        colors: {
          "editor.background": "#171717",
        },
      });

      monaco.editor.setTheme("my-theme");
    },
    [],
  );

  const getFormattedDateTime = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    const hour = String(now.getHours()).padStart(2, "0");
    const minute = String(now.getMinutes()).padStart(2, "0");
    const second = String(now.getSeconds()).padStart(2, "0");

    return `${year}-${month}-${day}-${hour}-${minute}-${second}`;
  };

  // Function to download JSON data as a file
  const downloadJSON = (data: object, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  async function exportTraces(): Promise<void> {
    const data = await request(`/api/security/export-trace`);
    toast.info("Trace exported");

    const filename = `opendevin-trace-${getFormattedDateTime()}.json`;
    downloadJSON(data, filename);
  }

  async function updatePolicy(): Promise<void> {
    await request(`/api/security/policy`, {
      method: "POST",
      body: JSON.stringify({ policy }),
    });
    toast.info("Policy updated");
  }

  async function updateSettings(): Promise<void> {
    const payload = { RISK_SEVERITY: selectedRisk };
    await request(`/api/security/settings`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    toast.info("Settings updated");
  }

  const handleExportTraces = useCallback(() => {
    exportTraces();
  }, [exportTraces]);

  const handleUpdatePolicy = useCallback(() => {
    updatePolicy();
  }, [updatePolicy]);

  const handleUpdateSettings = useCallback(() => {
    updateSettings();
  }, [updateSettings]);

  const sections: { [key in SectionType]: JSX.Element } = {
    logs: (
      <>
        <div className="flex justify-between items-center border-b border-neutral-600 mb-4 p-4">
          <h2 className="text-2xl">Logs</h2>
          <Button onClick={handleExportTraces} className="bg-neutral-700">
            Export Trace
          </Button>
        </div>
        <div className="flex-1 p-4 max-h-screen overflow-y-auto" ref={logsRef}>
          {logs.map((log: SecurityAnalyzerLog, index: number) => (
            <div
              key={index}
              className={`mb-2 p-2 rounded-lg ${log.confirmed_changed && log.is_confirmed === "confirmed" ? "border-green-800" : "border-red-800"}`}
              style={{
                backgroundColor: "rgba(128, 128, 128, 0.2)",
                borderWidth: log.confirmed_changed ? "2px" : "0",
              }}
            >
              <p className="text-sm relative break-words">
                {log.content}
                {(log.is_confirmed === "awaiting_confirmation" ||
                  log.confirmed_changed) && (
                  <IoAlertCircle className="absolute top-0 right-0" />
                )}
              </p>
              <p className={`text-xs ${getRiskColor(log.security_risk)}`}>
                {getRiskText(log.security_risk)}
              </p>
            </div>
          ))}
        </div>
      </>
    ),
    policy: (
      <>
        <div className="flex justify-between items-center border-b border-neutral-600 mb-4 p-4">
          <h2 className="text-2xl">Policy</h2>
          <Button className="bg-neutral-700" onClick={handleUpdatePolicy}>
            Update Policy
          </Button>
        </div>
        <div className="flex grow items-center justify-center">
          <Editor
            path="policy.py"
            height="100%"
            onMount={handleEditorDidMount}
            value={policy}
            onChange={(value) => setPolicy(`${value}`)}
          />
        </div>
      </>
    ),
    settings: (
      <>
        <div className="flex justify-between items-center border-b border-neutral-600 mb-4 p-4">
          <h2 className="text-2xl">Settings</h2>
          <Button className="bg-neutral-700" onClick={handleUpdateSettings}>
            Update Settings
          </Button>
        </div>
        <div className="flex grow p-4">
          <div className="flex flex-col w-full">
            <p className="mb-2">Ask for user confirmation on risk severity:</p>
            <Select
              placeholder="Select risk severity"
              value={selectedRisk}
              onChange={(e) =>
                setSelectedRisk(Number(e.target.value) as ActionSecurityRisk)
              }
              className={getRiskColor(selectedRisk)}
              selectedKeys={new Set([selectedRisk.toString()])}
              aria-label="Select risk severity"
            >
              <SelectItem
                key={ActionSecurityRisk.UNKNOWN}
                aria-label="Unknown Risk"
                className={getRiskColor(ActionSecurityRisk.UNKNOWN)}
              >
                {getRiskText(ActionSecurityRisk.UNKNOWN)}
              </SelectItem>
              <SelectItem
                key={ActionSecurityRisk.LOW}
                aria-label="Low Risk"
                className={getRiskColor(ActionSecurityRisk.LOW)}
              >
                {getRiskText(ActionSecurityRisk.LOW)}
              </SelectItem>
              <SelectItem
                key={ActionSecurityRisk.MEDIUM}
                aria-label="Medium Risk"
                className={getRiskColor(ActionSecurityRisk.MEDIUM)}
              >
                {getRiskText(ActionSecurityRisk.MEDIUM)}
              </SelectItem>
              <SelectItem
                key={ActionSecurityRisk.HIGH}
                aria-label="High Risk"
                className={getRiskColor(ActionSecurityRisk.HIGH)}
              >
                {getRiskText(ActionSecurityRisk.HIGH)}
              </SelectItem>
              <SelectItem
                key={ActionSecurityRisk.HIGH + 1}
                aria-label="Don't ask for confirmation"
              >
                Don&apos;t ask for confirmation
              </SelectItem>
            </Select>
          </div>
        </div>
      </>
    ),
  };

  return (
    <div className="flex flex-1 w-full h-full">
      <div className="w-60 bg-neutral-800 border-r border-r-neutral-600 p-4 flex-shrink-0">
        <div className="text-center mb-2">
          <InvariantLogoIcon className="mx-auto mb-1" />
          <b>Invariant Analyzer</b>
        </div>
        <p className="text-[0.6rem]">
          Invariant Analyzer continuously monitors your OpenDevin agent for
          security issues.{" "}
          <a
            className="underline"
            href="https://github.com/invariantlabs-ai/invariant"
            target="_blank"
            rel="noreferrer"
          >
            Click to learn more
          </a>
        </p>
        <hr className="border-t border-neutral-600 my-2" />
        <ul className="space-y-2">
          <div
            className={`cursor-pointer p-2 rounded ${activeSection === "logs" && "bg-neutral-600"}`}
            onClick={() => setActiveSection("logs")}
          >
            Logs
          </div>
          <div
            className={`cursor-pointer p-2 rounded ${activeSection === "policy" && "bg-neutral-600"}`}
            onClick={() => setActiveSection("policy")}
          >
            Policy
          </div>
          <div
            className={`cursor-pointer p-2 rounded ${activeSection === "settings" && "bg-neutral-600"}`}
            onClick={() => setActiveSection("settings")}
          >
            Settings
          </div>
        </ul>
      </div>
      <div className="flex flex-col min-h-0 w-full overflow-y-auto bg-neutral-900">
        {sections[activeSection as SectionType]}
      </div>
    </div>
  );
}

export default SecurityInvariant;
