import { combineReducers, configureStore } from "@reduxjs/toolkit";
import agentReducer from "./state/agentSlice";
import browserReducer from "./state/browserSlice";
import chatReducer from "./state/chatSlice";
import codeReducer from "./state/codeSlice";
import commandReducer from "./state/commandSlice";
import errorsReducer from "./state/errorsSlice";
import taskReducer from "./state/taskSlice";
import jupyterReducer from "./state/jupyterSlice";
import securityAnalyzerReducer from "./state/securityAnalyzerSlice";

export const rootReducer = combineReducers({
  browser: browserReducer,
  chat: chatReducer,
  code: codeReducer,
  cmd: commandReducer,
  errors: errorsReducer,
  task: taskReducer,
  agent: agentReducer,
  jupyter: jupyterReducer,
  securityAnalyzer: securityAnalyzerReducer,
});

const store = configureStore({
  reducer: rootReducer,
});

export type RootState = ReturnType<typeof store.getState>;
export type AppStore = typeof store;
export type AppDispatch = typeof store.dispatch;

export default store;
