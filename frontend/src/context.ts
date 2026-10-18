import { createContext, useContext } from "react";
import type { Backend } from "./api/client";
import type { JobResult, SessionInfo } from "./api/types";

export interface AppContextValue {
  backend: Backend;
  demo: boolean;
  session: SessionInfo | null;
  /** null — ещё проверяем; false — сервер недоступен. */
  serverOk: boolean | null;
  /** Есть сессия (cookie) или ключ в памяти вкладки. */
  authenticated: boolean;
  login: (key: string) => Promise<void>;
  logout: () => Promise<void>;
  copilotEnabled: boolean;
  disableCopilot: () => void;
  /** Перечитать сессию (например, после 401 — сессия истекла). */
  refreshSession: () => Promise<void>;
  toast: (text: string, tone?: "info" | "error" | "success") => void;
  openSettings: () => void;
  /** Последний полученный результат (для «О системе» → модели). */
  lastResult: JobResult | null;
  setLastResult: (r: JobResult | null) => void;
}

export const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const v = useContext(AppContext);
  if (!v) throw new Error("AppContext не инициализирован");
  return v;
}
