import { useEffect, useState } from "react";
import { Info, Monitor, Moon, ScanSearch, Settings, Sun } from "lucide-react";
import { useApp } from "../context";
import type { Route } from "../lib/router";
import { applyTheme, getThemePref, type ThemePref } from "../lib/theme";
import { cx, LocalBadge } from "./ui";

export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <defs>
        <linearGradient id="ks-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#b8436e" />
          <stop offset="1" stopColor="#2f3c5b" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#ks-logo)" />
      <path d="M32 12 48 18v12c0 11-7 19-16 22-9-3-16-11-16-22V18z" fill="none" stroke="#fff" strokeWidth="4" strokeLinejoin="round" />
      <circle cx="32" cy="30" r="6" fill="#fff" />
    </svg>
  );
}

const THEME_ORDER: ThemePref[] = ["system", "light", "dark"];

export default function Header({ route }: { route: Route }) {
  const { demo, openSettings, serverOk } = useApp();
  const [theme, setTheme] = useState<ThemePref>(() => getThemePref());

  useEffect(() => {
    applyTheme(theme);
    if (theme !== "system") return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => applyTheme("system");
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, [theme]);

  const nextTheme = () => setTheme(THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length]);
  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
  const themeTitle = theme === "dark" ? "Тёмная тема" : theme === "light" ? "Светлая тема" : "Тема как в системе";

  const link = (href: string, active: boolean, label: string, Icon: typeof Info) => (
    <a
      href={href}
      className={cx(
        "inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-medium transition",
        active
          ? "bg-slate-900/5 text-slate-900 dark:bg-white/10 dark:text-white"
          : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white",
      )}
    >
      <Icon className="h-4 w-4" />
      <span className="hidden sm:inline">{label}</span>
    </a>
  );

  return (
    <header className="no-print border-b border-slate-200/70 bg-white/80 backdrop-blur-xl dark:border-white/[.06] dark:bg-ink-950/75">
      <div className="mx-auto flex h-16 max-w-[1320px] items-center gap-3 px-4 sm:px-6 lg:px-8">
        <a href="#/" className="flex items-center gap-2.5" aria-label="Kassa Sentinel — на главную">
          <Logo className="h-9 w-9 shrink-0" />
          <div className="leading-tight">
            <div className="font-display text-[15px] font-extrabold tracking-tight">Kassa Sentinel</div>
            <div className="hidden text-[11px] text-slate-500 dark:text-slate-400 sm:block">AI-контроль кассовых операций</div>
          </div>
        </a>
        {demo && (
          <span className="chip hidden border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-300 sm:inline-flex">
            СИНТЕТИЧЕСКОЕ ДЕМО
          </span>
        )}
        <nav className="ml-2 flex items-center gap-1">
          {link("#/", route.name !== "about", "Проверка", ScanSearch)}
          {link("#/about", route.name === "about", "О системе", Info)}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden md:inline-flex">
            <LocalBadge />
          </span>
          {serverOk === false && !demo && (
            <span className="chip hidden border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-400/20 dark:bg-rose-400/10 dark:text-rose-300 sm:inline-flex">
              нет связи с сервером
            </span>
          )}
          <button className="rounded-xl p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-white/10" onClick={nextTheme} title={themeTitle} aria-label={themeTitle}>
            <ThemeIcon className="h-5 w-5" />
          </button>
          <button className="rounded-xl p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-white/10" onClick={openSettings} title="Настройки" aria-label="Настройки">
            <Settings className="h-5 w-5" />
          </button>
        </div>
      </div>
    </header>
  );
}
