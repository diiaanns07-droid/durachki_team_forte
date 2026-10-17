import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { confidenceLevel, formatPercent } from "../lib/format";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

export function ConfidenceBadge({ value, className }: { value?: number; className?: string }) {
  const lvl = confidenceLevel(value);
  const tone = {
    high: "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-400/20 dark:bg-rose-400/10 dark:text-rose-300",
    medium: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-300",
    low: "border-slate-200 bg-slate-50 text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300",
    unknown: "border-slate-200 bg-slate-50 text-slate-500 dark:border-white/10 dark:bg-white/5 dark:text-slate-400",
  }[lvl.tone];
  return (
    <span className={cx("chip tnum", tone, className)} title="Уверенность системы в эпизоде (0–100 %)">
      <span
        className={cx(
          "h-1.5 w-1.5 rounded-full",
          lvl.tone === "high" ? "bg-rose-500" : lvl.tone === "medium" ? "bg-amber-500" : "bg-slate-400",
        )}
      />
      уверенность {lvl.label}
      {value !== undefined && <span className="opacity-70">· {formatPercent(value)}</span>}
    </span>
  );
}

export function ProgressBar({ value, className, tone = "brand" }: { value: number; className?: string; tone?: "brand" | "ink" }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 1000) / 10;
  return (
    <div
      className={cx("h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-white/[.06]", className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
    >
      <div
        className={cx(
          "h-full rounded-full transition-[width] duration-500 ease-out",
          tone === "brand" ? "bg-gradient-to-r from-brand-500 to-brand-700" : "bg-ink-800 dark:bg-white",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function SectionTitle({ icon, title, aside }: { icon?: ReactNode; title: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="flex items-center gap-2 text-base font-bold text-slate-900 dark:text-white">
        {icon}
        {title}
      </h2>
      {aside}
    </div>
  );
}

export function Modal({
  open,
  onClose,
  title,
  children,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open || typeof document === "undefined") return null;
  return createPortal(
    <div className="no-print fixed inset-0 z-50 flex items-end justify-center bg-ink-950/60 p-0 backdrop-blur-sm sm:items-center sm:p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        className={cx(
          "card max-h-[92vh] w-full animate-fade-in overflow-y-auto rounded-b-none p-5 sm:rounded-2xl sm:p-6",
          wide ? "sm:max-w-3xl" : "sm:max-w-lg",
        )}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 className="text-lg font-bold">{title}</h2>
          <button className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-white/10" onClick={onClose} aria-label="Закрыть">
            <X className="h-5 w-5" />
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cx("animate-spin", className ?? "h-4 w-4")} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity=".2" strokeWidth="3" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function LocalBadge({ compact }: { compact?: boolean }) {
  return (
    <span
      className="chip border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-300"
      title="Видео обрабатывается на этом сервере; облачные AI-сервисы не используются"
    >
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
      </span>
      {compact ? "Локально" : "Работает полностью локально"}
    </span>
  );
}
