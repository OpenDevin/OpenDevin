import React from "react";
import "./Browser.css";
import { useSelector } from "react-redux";
import { RootState } from "../store";

function Browser(): JSX.Element {
  const url = useSelector((state: RootState) => state.browser.url);
  return (
    <div className="mockup-browser">
      <div className="mockup-browser-toolbar">
        <div className="input">{url}</div>
      </div>
      <iframe
        title="Devin's browser"
        width="100%"
        height="100%"
        src={url}/>
    </div>
  );
}

export default Browser;
