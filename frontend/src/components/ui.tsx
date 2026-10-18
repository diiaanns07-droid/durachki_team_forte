import { useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { CircleAlert, CircleCheck, Info, Lock, TriangleAlert, X } from "lucide-react";
import { confidenceLevel, formatPercent } from "../lib/format";

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/** Смысловой тон: event — события и ошибки (красный), success — ТОЛЬКО подтверждённый успех (зелёный),
 *  warning — предупреждение/демо, info — справка, muted — второстепенное. */
export type Tone = "event" | "success" | "warning" | "info" | "muted";

// Полные имена классов (не шаблон): Tailwind оставляет в сборке только встреченные в исходниках классы.
const CHIP_TONE: Record<Tone, string> = {
  event: "chip-event",
  success: "chip-success",
  warning: "chip-warning",
  info: "chip-info",
  muted: "chip-muted",
};

export function StatusChip({
  tone = "info",
  icon,
  children,
  className,
  title,
}: {
  tone?: Tone;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span className={cx("chip", CHIP_TONE[tone], className)} title={title}>
      {icon}
      {children}
    </span>
  );
}

const NOTICE_ICON = { info: Info, event: CircleAlert, success: CircleCheck, warning: TriangleAlert } as const;

/** Встроенное уведомление в потоке страницы (ошибка, предупреждение, справка, подтверждение). */
export function Notice({
  tone = "info",
  title,
  children,
  className,
}: {
  tone?: "info" | "event" | "success" | "warning";
  title?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const Icon = NOTICE_ICON[tone];
  return (
    <div className={cx("ks-notice", className)} data-tone={tone} role={tone === "event" ? "alert" : "status"}>
      <Icon aria-hidden />
      <div className="min-w-0">
        {title && <div className="font-semibold">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function ConfidenceBadge({ value, className }: { value?: number; className?: string }) {
  const lvl = confidenceLevel(value);
  // уверенность относится к событию: высокая — цвет события, средняя — предупреждение, остальное — нейтрально
  const tone: Tone = lvl.tone === "high" ? "event" : lvl.tone === "medium" ? "warning" : "muted";
  return (
    <span className={cx("chip tnum", CHIP_TONE[tone], className)} title="Уверенность системы в эпизоде (0–100 %)">
      <span className="ks-dot" aria-hidden />
      уверенность {lvl.label}
      {value !== undefined && <span className="opacity-70">· {formatPercent(value)}</span>}
    </span>
  );
}

/** Прогресс обработки. tone="brand" (по умолчанию) — цвет акцента: обработка не событие и не ошибка. */
export function ProgressBar({ value, className, tone = "brand" }: { value: number; className?: string; tone?: "brand" | "ink" }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 1000) / 10;
  return (
    <div
      className={cx("ks-progress", className)}
      data-tone={tone}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
    >
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

export function SectionTitle({ icon, title, aside }: { icon?: ReactNode; title: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="ks-section-title">
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
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  // Фокус — только при смене open (onClose у вызывающих часто новый на каждый рендер): в диалог при
  // открытии и обратно на вызвавший элемент при закрытии.
  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => prev?.focus?.();
  }, [open]);
  if (!open || typeof document === "undefined") return null;
  // Портал в body (fixed не зависит от transform предков); класс ks-ws переносит токены рабочей области.
  return createPortal(
    <div className="ks-ws">
      <div
        className="no-print fixed inset-0 z-50 flex items-end justify-center p-0 backdrop-blur-sm sm:items-center sm:p-6"
        style={{ backgroundColor: "var(--ks-overlay)" }}
        onMouseDown={onClose}
      >
        <div
          ref={dialogRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
          className={cx(
            "card max-h-[92vh] w-full animate-fade-in overflow-y-auto rounded-b-none p-5 shadow-ks-lift outline-none sm:rounded-2xl sm:p-6",
            wide ? "sm:max-w-3xl" : "sm:max-w-lg",
          )}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div className="mb-4 flex items-center justify-between gap-4">
            <h2 id={titleId} className="text-lg font-bold">
              {title}
            </h2>
            <button className="ks-icon-btn" onClick={onClose} aria-label="Закрыть">
              <X className="h-5 w-5" />
            </button>
          </div>
          {children}
        </div>
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

/** Справочная метка «обработка локальная». Нейтральная: это свойство установки, а не подтверждённый
 *  успешный исход, поэтому не зелёная и без «живой» пульсации. */
export function LocalBadge({ compact }: { compact?: boolean }) {
  return (
    <span className="chip chip-info" title="Видео обрабатывается на этом сервере; облачные AI-сервисы не используются">
      <Lock className="h-3 w-3" aria-hidden />
      {compact ? "Локально" : "Работает полностью локально"}
    </span>
  );
}
