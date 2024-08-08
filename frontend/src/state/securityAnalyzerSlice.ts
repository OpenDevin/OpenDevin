import { createSlice } from "@reduxjs/toolkit";

export enum ActionSecurityRisk {
  UNKNOWN = -1,
  LOW = 0,
  MEDIUM = 1,
  HIGH = 2,
}

export type SecurityAnalyzerLog = {
  id: number;
  content: string;
  security_risk: ActionSecurityRisk;
  is_confirmed?: "awaiting_confirmation" | "confirmed" | "rejected";
  confirmed_changed: boolean;
};

const initialLogs: SecurityAnalyzerLog[] = [];

export const securityAnalyzerSlice = createSlice({
  name: "securityAnalyzer",
  initialState: {
    logs: initialLogs,
  },
  reducers: {
    appendSecurityAnalyzerInput: (state, action) => {
      const log = {
        id: action.payload.id,
        content:
          action.payload.args.command ||
          action.payload.args.code ||
          action.payload.args.content ||
          action.payload.message,
        security_risk: action.payload.args.security_risk as ActionSecurityRisk,
        is_confirmed: action.payload.args.is_confirmed,
        confirmed_changed: false,
      };

      const existingLog = state.logs.find(
        (stateLog) =>
          stateLog.id === log.id ||
          (stateLog.is_confirmed === "awaiting_confirmation" &&
            stateLog.content === log.content),
      );

      if (existingLog) {
        if (existingLog.is_confirmed !== log.is_confirmed) {
          existingLog.is_confirmed = log.is_confirmed;
          existingLog.confirmed_changed = true;
        }
      } else {
        state.logs.push(log);
      }
    },
  },
});

export const { appendSecurityAnalyzerInput } = securityAnalyzerSlice.actions;

export default securityAnalyzerSlice.reducer;
