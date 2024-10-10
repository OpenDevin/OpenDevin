import { combineReducers, configureStore } from "@reduxjs/toolkit";
import agentReducer from "./state/agentSlice";
import browserReducer from "./state/browserSlice";
import chatReducer from "./state/chatSlice";
import codeReducer from "./state/codeSlice";
import fileStateReducer from "./state/file-state-slice";
import initialQueryReducer from "./state/initial-query-slice";
import commandReducer from "./state/commandSlice";
import jupyterReducer from "./state/jupyterSlice";
import securityAnalyzerReducer from "./state/securityAnalyzerSlice";
import statusReducer from "./state/statusSlice";

export const rootReducer = combineReducers({
  fileState: fileStateReducer,
  initalQuery: initialQueryReducer,
  browser: browserReducer,
  chat: chatReducer,
  code: codeReducer,
  cmd: commandReducer,
  agent: agentReducer,
  jupyter: jupyterReducer,
  securityAnalyzer: securityAnalyzerReducer,
  status: statusReducer,
});

const store = configureStore({
  reducer: rootReducer,
});

export type RootState = ReturnType<typeof store.getState>;
export type AppStore = typeof store;
export type AppDispatch = typeof store.dispatch;

export default store;
