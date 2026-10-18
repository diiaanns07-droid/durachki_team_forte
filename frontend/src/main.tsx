import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource-variable/manrope";
import "./index.css";
import App from "./App";
import { loadUiConfig } from "./lib/uiConfig";

// Рантайм-флаги (/ui-config.json) — до первого рендера, не дольше 1,5 с.
void loadUiConfig().finally(() => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
