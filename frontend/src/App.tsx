import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { cx } from "./components/ui";

/** Постоянная пометка: синтетическое демо не подменяет реальную загрузку/результат. */
function DemoRibbon() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 bg-amber-400 px-4 py-1.5 text-center text-xs font-bold uppercase tracking-wider text-amber-950 print:static">
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

  const toast = useCallback((text: string, tone: Toast["tone"] = "info") => {
    const id = ++toastSeq.current;
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4200);
  }, []);

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
      <div className="flex min-h-screen flex-col">
        <div className="sticky top-0 z-40">
          {demo && <DemoRibbon />}
          <Header route={route} />
        </div>
        <main className="mx-auto w-full max-w-[1320px] flex-1 px-4 pb-16 pt-6 sm:px-6 lg:px-8">
          {route.name === "upload" && <UploadScreen />}
          {route.name === "job" && <JobScreen key={route.id} jobId={route.id} />}
          {route.name === "about" && <AboutScreen />}
          {route.name === "replay" && <ReplayScreen />}
        </main>
        <footer className="no-print border-t border-slate-200/70 py-6 text-center text-xs text-slate-500 dark:border-white/5 dark:text-slate-500">
          Kassa Sentinel · результат — подсказка для проверяющего, не юридическое заключение · видео не покидает локальный контур
        </footer>
      </div>
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={() => {
          setSettingsOpen(false);
          void initSession();
          toast("Настройки сохранены", "success");
        }}
      />
      <div className="no-print pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex flex-col items-center gap-2 px-4">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cx(
              "pointer-events-auto animate-fade-in rounded-xl px-4 py-2.5 text-sm font-medium shadow-lift",
              t.tone === "error" && "bg-rose-600 text-white",
              t.tone === "success" && "bg-emerald-600 text-white",
              t.tone === "info" && "bg-ink-900 text-white dark:bg-white dark:text-ink-900",
            )}
            role="status"
          >
            {t.text}
          </div>
        ))}
      </div>
    </AppContext.Provider>
  );
}
