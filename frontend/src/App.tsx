import React, { useState } from "react";
import "./App.css";
import ChatInterface from "./components/ChatInterface";
import Terminal from "./components/Terminal";
import Planner from "./components/Planner";
import CodeEditor from "./components/CodeEditor";
import Browser from "./components/Browser";
import Errors from "./components/Errors";
import BannerSettings from "./components/BannerSettings";

const TAB_OPTIONS = ["terminal", "planner", "code", "browser"] as const;
type TabOption = (typeof TAB_OPTIONS)[number];

type TabProps = {
  name: string;
  active: boolean;
  onClick: () => void;
};
function Tab({ name, active, onClick }: TabProps): JSX.Element {
  return (
    <div
      className={`tab ${active ? "tab-active" : ""}`}
      onClick={() => onClick()}
    >
      <p className="font-bold">{name}</p>
    </div>
  );
}

const tabData = {
  terminal: {
    name: "Terminal",
    component: null,
  },
  planner: {
    name: "Planner",
    component: <Planner key="planner" />,
  },
  code: {
    name: "Code Editor",
    component: <CodeEditor key="code" />,
  },
  browser: {
    name: "Browser",
    component: <Browser key="browser" />,
  },
};

function App(): JSX.Element {
  const [activeTab, setActiveTab] = useState<TabOption>("terminal");

  return (
    <div className="app flex">
      <Errors />
      <div className="left-pane">
        <ChatInterface />
      </div>
      <div className="right-pane">
        <div className="workspace-heading">
          <p>OpenDevin Workspace</p>
          <BannerSettings />
        </div>
        <div role="tablist" className="tabs tabs-bordered tabs-lg">
          {TAB_OPTIONS.map((tab) => (
            <Tab
              key={tab}
              name={tabData[tab].name}
              active={activeTab === tab}
              onClick={() => setActiveTab(tab)}
            />
          ))}
        </div>
        {/* Keep terminal permanently open - see component for more details */}
        <Terminal key="terminal" hidden={activeTab !== "terminal"} />
        {tabData[activeTab].component}
      </div>
    </div>
  );
}

export default App;
