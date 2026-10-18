import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, CircleCheck, Info, X } from "lucide-react";
import { HttpBackend, type Backend } from "./api/client";
import { DemoBackend, isDemoMode } from "./api/demo";
import type { JobResult, SessionInfo } from "./api/types";
import { AppContext, type AppContextValue } from "./context";
import { setBatchBackend } from "./lib/batch";
import { parseHash, type Route } from "./lib/router";
import Header from "./components/Header";
import UploadScreen from "./components/UploadScreen";
import JobScreen from "./components/JobScreen";
import AboutScreen from "./components/AboutScreen";
import ReplayScreen from "./components/ReplayScreen";
import SettingsDialog from "./components/SettingsDialog";

/** Постоянная пометка: синтетическое демо не подменяет реальную загрузку/результат. */
function DemoRibbon() {
  return (
    <div className="ks-ribbon-demo print:static" role="note">
      <span>Синтетическое демо</span>
      <span className="hidden font-medium normal-case tracking-normal sm:inline">данные сгенерированы в браузере, сервер анализа не используется</span>
      <a className="rounded bg-amber-950/10 px-2 py-0.5 font-semibold normal-case tracking-normal underline-offset-2 hover:underline" href="./#/">
        выйти из демо
      </a>
    </div>
  );
}

interface Toast {
  id: number;
  text: string;
  tone: "info" | "error" | "success";
}

// error — событие/ошибка (красный); success — только подтверждённое действие (зелёный); info — нейтрально
const TOAST_ICON = { info: Info, error: CircleAlert, success: CircleCheck } as const;
const TOAST_MS = { info: 4200, success: 4200, error: 6500 } as const; // ошибку дольше: её надо успеть прочитать

export default function App() {
  const demo = useMemo(() => isDemoMode(), []);
  const backend: Backend = useMemo(() => {
    const b = demo ? new DemoBackend() : new HttpBackend();
    setBatchBackend(b);
    return b;
  }, [demo]);
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [serverOk, setServerOk] = useState<boolean | null>(null);
  const [copilotEnabled, setCopilotEnabled] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [lastResult, setLastResult] = useState<JobResult | null>(null);
  const toastSeq = useRef(0);

  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const applySession = useCallback(
    (s: SessionInfo) => {
      setSession(s);
      setServerOk(demo || s.reachable);
      setCopilotEnabled(s.features.copilot !== false);
    },
    [demo],
  );

  const initSession = useCallback(async () => {
    applySession(await backend.init());
  }, [backend, applySession]);

  const login = useCallback(
    async (key: string) => {
      applySession(await backend.login(key));
    },
    [backend, applySession],
  );

  const logout = useCallback(async () => {
    await backend.logout();
    await initSession();
  }, [backend, initSession]);

  useEffect(() => {
    void initSession();
  }, [initSession]);

  const disableCopilot = useCallback(() => setCopilotEnabled(false), []);

  const dismissToast = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const toast = useCallback(
    (text: string, tone: Toast["tone"] = "info") => {
      const id = ++toastSeq.current;
      setToasts((t) => [...t.slice(-2), { id, text, tone }]); // на экране не больше трёх
      setTimeout(() => dismissToast(id), TOAST_MS[tone] ?? TOAST_MS.info);
    },
    [dismissToast],
  );

  const ctx: AppContextValue = {
    backend,
    demo,
    session,
    serverOk,
    authenticated: demo || !!session?.authenticated,
    login,
    logout,
    copilotEnabled,
    disableCopilot,
    refreshSession: initSession,
    toast,
    openSettings: () => setSettingsOpen(true),
    lastResult,
    setLastResult,
  };

  return (
    <AppContext.Provider value={ctx}>
      {/* Рабочая область: токены и стили U01 действуют только внутри .ks-ws (см. src/index.css). */}
      <div className="ks-ws ks-ws-page flex min-h-screen flex-col">
        <button className="ks-skip-link" onClick={() => document.getElementById("main")?.focus()}>
          Перейти к содержимому
        </button>
        <div className="sticky top-0 z-40">
          {demo && <DemoRibbon />}
          <Header route={route} />
        </div>
        <main id="main" tabIndex={-1} className="ks-container ks-main flex-1 outline-none">
          {route.name === "upload" && <UploadScreen />}
          {route.name === "job" && <JobScreen key={route.id} jobId={route.id} />}
          {route.name === "about" && <AboutScreen />}
          {route.name === "replay" && <ReplayScreen />}
        </main>
        <footer className="ks-footer no-print py-6 text-center text-xs">
          <div className="ks-container">
            Kassa Sentinel · результат — подсказка для проверяющего, не юридическое заключение · видео не покидает локальный контур
          </div>
        </footer>
        <SettingsDialog
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onSaved={() => {
            setSettingsOpen(false);
            void initSession();
            toast("Настройки сохранены", "success");
          }}
        />
        <div className="ks-toasts no-print" aria-label="Уведомления">
          {toasts.map((t) => {
            const Icon = TOAST_ICON[t.tone] ?? Info;
            return (
              <div key={t.id} className="ks-toast" data-tone={t.tone} role={t.tone === "error" ? "alert" : "status"}>
                <Icon aria-hidden />
                <span className="min-w-0 flex-1">{t.text}</span>
                <button className="-mr-1 -mt-0.5 rounded-md p-0.5 text-ks-text-3 hover:text-ks-text" onClick={() => dismissToast(t.id)} aria-label="Закрыть уведомление">
                  <X className="h-4 w-4" />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </AppContext.Provider>
  );
}
