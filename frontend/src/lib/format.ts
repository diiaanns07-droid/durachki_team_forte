// Форматирование чисел/времени для UI. Чистые функции — покрыты vitest.

/** Таймкод mm:ss.s (h:mm:ss.s для ≥ 1 ч). Округление до десятых с переносом разрядов. */
export function formatTimecode(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec) || sec < 0) sec = 0;
  const tenths = Math.round(sec * 10);
  const totalSec = Math.floor(tenths / 10);
  const frac = tenths % 10;
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const ss = `${String(s).padStart(2, "0")}.${frac}`;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
  return `${String(m).padStart(2, "0")}:${ss}`;
}

/** Короткая длительность: «42 с», «2 мин 05 с». */
export function formatDuration(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec) || sec < 0) return "—";
  if (sec < 60) return `${Math.max(0, Math.round(sec))} с`;
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m} мин ${String(s).padStart(2, "0")} с`;
}

/** Уверенность 0..1 → «87 %». */
export function formatPercent(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const clamped = Math.min(1, Math.max(0, v));
  return `${(clamped * 100).toFixed(digits)} %`;
}

/** Размер файла: «28,9 МБ». */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["Б", "КБ", "МБ", "ГБ"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  const s = i === 0 ? String(Math.round(v)) : v.toFixed(1).replace(".", ",");
  return `${s} ${units[i]}`;
}

/** Склонение: plural(3, ["нарушение","нарушения","нарушений"]). */
export function plural(n: number, forms: [string, string, string]): string {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  if (a > 10 && a < 20) return forms[2];
  if (b > 1 && b < 5) return forms[1];
  if (b === 1) return forms[0];
  return forms[2];
}

/** Оценка оставшегося времени по доле прогресса. null — оценить нельзя. */
export function estimateEta(elapsedSec: number, progress: number): number | null {
  if (!Number.isFinite(elapsedSec) || !Number.isFinite(progress)) return null;
  if (progress <= 0.02 || elapsedSec < 1) return null;
  if (progress >= 1) return 0;
  return Math.max(0, (elapsedSec / progress) * (1 - progress));
}

export function formatDateTime(iso: string | number | Date): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Уровень уверенности → подпись и тон бейджа. */
export function confidenceLevel(c: number | null | undefined): {
  label: string;
  tone: "high" | "medium" | "low" | "unknown";
} {
  if (c === null || c === undefined || !Number.isFinite(c)) return { label: "н/д", tone: "unknown" };
  if (c >= 0.75) return { label: "высокая", tone: "high" };
  if (c >= 0.5) return { label: "средняя", tone: "medium" };
  return { label: "низкая", tone: "low" };
}

/** Цвет риска 0..1: зелёный → янтарный → красный (HSL). */
export function riskColor(risk: number, alpha = 1): string {
  const r = Math.min(1, Math.max(0, Number.isFinite(risk) ? risk : 0));
  // 150° (зелёный) → 40° (янтарный) → 350° (красный)
  const hue = r < 0.5 ? 150 - (150 - 40) * (r / 0.5) : 40 - 50 * ((r - 0.5) / 0.5);
  const h = (hue + 360) % 360;
  const sat = 70 + 15 * r;
  const light = 48 - 6 * r;
  return `hsla(${h.toFixed(0)}, ${sat.toFixed(0)}%, ${light.toFixed(0)}%, ${alpha})`;
}
