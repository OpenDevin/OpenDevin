import React, { ChangeEvent, useEffect, useRef, useState } from "react";
import { useSelector } from "react-redux";
import assistantAvatar from "../assets/assistant-avatar.png";
import userAvatar from "../assets/user-avatar.png";
import { sendChatMessage } from "../services/chatService";
import { RootState } from "../store";
import "./ChatInterface.css";

function MessageList(): JSX.Element {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages } = useSelector((state: RootState) => state.chat);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="message-list">
      {messages.map((msg, index) => (
        <div key={index} className="message-layout">
          <div
            className={`${msg.sender === "user" ? "user-message" : "message"}`}
          >
            <img
              src={msg.sender === "user" ? userAvatar : assistantAvatar}
              alt={`${msg.sender} avatar`}
              className="avatar"
            />
            <div className="chat chat-bubble">{msg.content}</div>
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

function InitializingStatus(): JSX.Element {
  return (
    <div className="initializing-status">
      <img src={assistantAvatar} alt="assistant avatar" className="avatar" />
      <div>Initializing agent (may take up to 10 seconds)...</div>
    </div>
  );
}

function ChatInterface(): JSX.Element {
  const { initialized } = useSelector((state: RootState) => state.task);
  const [inputMessage, setInputMessage] = useState("");
  const [selectedDirectory, setSelectedDirectory] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSendMessage = () => {
    if (inputMessage.trim() !== "") {
      sendChatMessage(inputMessage);
      setInputMessage("");
    }
  };

  const handleDirectorySelected = (event: ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target;
    if (files && files.length > 0) {
      const directory = files[0].webkitRelativePath.split("/")[0];
      setSelectedDirectory(directory);
    }
  };

  const handleEditDirectory = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Clear the file input value
      fileInputRef.current.click(); // Trigger the file picker dialog
    }
  };

  return (
    <div className="chat-interface">
      <label
        htmlFor="directoryInput"
        className="custom-file-input"
        style={{ display: selectedDirectory ? "none" : "block" }}
      >
        Choose Directory
        <input
          id="directoryInput"
          type="file"
          capture="directory"
          webkitdirectory=""
          onChange={handleDirectorySelected}
          ref={fileInputRef}
          style={{ display: "none" }}
        />
      </label>
      {selectedDirectory && (
        <div className="selected-directory">
          Selected Directory: {selectedDirectory}
          <button type="button" onClick={handleEditDirectory}>
            Edit
          </button>
        </div>
      )}
      {initialized ? <MessageList /> : <InitializingStatus />}
      <div className="input-container">
        <div className="input-box">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Send a message (won't interrupt the Assistant)"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSendMessage();
              }
            }}
          />
          <button
            type="button"
            onClick={handleSendMessage}
            disabled={!initialized}
          >
            <span className="button-text">Send</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
