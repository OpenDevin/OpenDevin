/* eslint-disable react/react-in-jsx-scope */
/**
 * By default, Remix will handle hydrating your app on the client for you.
 * You are free to delete this file if you'd like to, but if you ever want it revealed again, you can run `npx remix reveal` ✨
 * For more information, see https://remix.run/file-conventions/entry.client
 */

import { RemixBrowser } from "@remix-run/react";
import { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { SocketProvider } from "./context/socket";
import "./i18n";
import store from "./store";

async function prepareApp() {
  if (false && process.env.NODE_ENV === "development") {
    const { worker } = await import("./mocks/browser");

    await worker.start({
      onUnhandledRequest: "bypass",
    });
  }
}

prepareApp().then(() =>
  startTransition(() => {
    hydrateRoot(
      document,
      <StrictMode>
        <SocketProvider>
          <Provider store={store}>
            <RemixBrowser />
          </Provider>
        </SocketProvider>
      </StrictMode>,
    );
  }),
);