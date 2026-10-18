import { useEffect, useState } from "react";
import { CircleAlert, Info, Monitor, Moon, ScanSearch, Settings, Sun } from "lucide-react";
import { useApp } from "../context";
import type { Route } from "../lib/router";
import { applyTheme, getThemePref, type ThemePref } from "../lib/theme";
import { LocalBadge } from "./ui";

/** Знак продукта. Малиновый brand — только здесь (идентичность), не в кнопках и статусах. */
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

/**
 * Шапка рабочей области. Порядок: знак → разделы → (справа) статусы → служебные кнопки.
 * Статусы по смыслу: нет связи — событие/ошибка (красный), демо — предупреждение (янтарный),
 * локальная обработка — справка (нейтральный). Зелёного «всё хорошо» здесь нет: шапка не знает,
 * что анализ подтверждён.
 */
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
    <a href={href} className="ks-nav-link" aria-current={active ? "page" : undefined} title={label}>
      <Icon className="h-4 w-4" aria-hidden />
      <span className="hidden sm:inline">{label}</span>
    </a>
  );

  return (
    <header className="ks-header no-print">
      <div className="ks-container flex h-16 items-center gap-2 sm:gap-3">
        <a href="#/" className="flex shrink-0 items-center gap-2.5 rounded-xl" aria-label="Kassa Sentinel — на главную">
          <Logo className="h-9 w-9 shrink-0" />
          <div className="hidden leading-tight min-[420px]:block">
            <div className="font-display text-[15px] font-extrabold tracking-tight text-ks-text">Kassa Sentinel</div>
            <div className="hidden text-[11px] text-ks-text-3 sm:block">AI-контроль кассовых операций</div>
          </div>
        </a>
        <nav className="ml-1 flex items-center gap-1 sm:ml-4" aria-label="Разделы">
          {link("#/", route.name !== "about", "Проверка", ScanSearch)}
          {link("#/about", route.name === "about", "О системе", Info)}
        </nav>
        <div className="ml-auto flex items-center gap-1.5 sm:gap-2">
          {serverOk === false && !demo && (
            <span className="chip chip-event" role="status">
              <CircleAlert className="h-3.5 w-3.5" aria-hidden />
              <span className="hidden sm:inline">нет связи с сервером</span>
              <span className="sm:hidden">нет связи</span>
            </span>
          )}
          {demo && <span className="chip chip-warning hidden uppercase tracking-wide sm:inline-flex">Синтетическое демо</span>}
          <span className="hidden lg:inline-flex">
            <LocalBadge />
          </span>
          <span className="mx-1 hidden h-6 w-px bg-[var(--ks-border)] sm:block" aria-hidden />
          <button className="ks-icon-btn" onClick={nextTheme} title={themeTitle} aria-label={themeTitle}>
            <ThemeIcon className="h-[18px] w-[18px]" />
          </button>
          <button className="ks-icon-btn" onClick={openSettings} title="Настройки" aria-label="Настройки">
            <Settings className="h-[18px] w-[18px]" />
          </button>
        </div>
      </div>
    </header>
  );
}
