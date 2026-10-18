// Рантайм-флаги UI: /ui-config.json рядом с index.html (в dist — правится без пересборки).
// Каждая новая гипотеза поведения — отдельный флаг; спорные режимы по умолчанию ВЫКЛЮЧЕНЫ до замера L2.
// Неверные значения в файле игнорируются (остаётся значение по умолчанию).

export interface UiConfig {
  /** Интервал опроса статуса задачи, мс. */
  pollIntervalMs: number;
  /** Потолок экспоненциальной паузы при сетевых ошибках, мс. */
  pollMaxIntervalMs: number;
  /** Не опрашивать сервер, пока вкладка скрыта. */
  pausePollingWhenHidden: boolean;
  /** Переход к таймкоду ставит паузу на точном кадре (для проверки кадра). */
  seekPauses: boolean;
  /** «Смотреть эпизод»: начать за N с до начала и остановиться через M с после конца. */
  episodePrerollS: number;
  episodePostrollS: number;
  /** Типы, для которых показываются кадры «до / после» (эпизоды-переходы). */
  beforeAfterTypes: string[];
  /** На сколько секунд до начала эпизода брать кадр «до». */
  beforeOffsetS: number;
  /** Рамки подсветки (заданы A12 для кадра t_peak) рисуются только на кадре не дальше N с от t_peak. */
  highlightMaxDtS: number;
  /**
   * Сдвиг времени при перемотке: "none" — таймкод = currentTime видео; "meta_t0" — + meta.video.t0_pts_sec
   * (если первый кадр файла имеет ненулевой PTS). ГИПОТЕЗА, по умолчанию выкл. до замера L2 (e2e-real --check-pts).
   */
  ptsOffsetMode: "none" | "meta_t0";
  /** Показывать пограничные моменты A12 (review_hints) отдельным блоком. */
  showReviewHints: boolean;
}

export const DEFAULT_UI_CONFIG: UiConfig = {
  pollIntervalMs: 1500,
  pollMaxIntervalMs: 10000,
  pausePollingWhenHidden: true,
  seekPauses: true,
  episodePrerollS: 2,
  episodePostrollS: 1,
  beforeAfterTypes: ["V02", "V03", "V07"],
  beforeOffsetS: 2,
  highlightMaxDtS: 0.1,
  ptsOffsetMode: "none",
  showReviewHints: true,
};

let current: UiConfig = { ...DEFAULT_UI_CONFIG };

export function uiConfig(): UiConfig {
  return current;
}

/** Слияние с проверкой типов: чужие/битые значения игнорируются. */
export function mergeUiConfig(base: UiConfig, raw: unknown): UiConfig {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return base;
  const r = raw as Record<string, unknown>;
  const out: UiConfig = { ...base };
  const num = (k: keyof UiConfig, min: number, max: number) => {
    const v = r[k];
    if (typeof v === "number" && Number.isFinite(v) && v >= min && v <= max) (out[k] as number) = v;
  };
  const bool = (k: keyof UiConfig) => {
    if (typeof r[k] === "boolean") (out[k] as boolean) = r[k] as boolean;
  };
  num("pollIntervalMs", 300, 60000);
  num("pollMaxIntervalMs", 1000, 120000);
  num("episodePrerollS", 0, 30);
  num("episodePostrollS", 0, 30);
  num("beforeOffsetS", 0.2, 30);
  num("highlightMaxDtS", 0, 10);
  bool("pausePollingWhenHidden");
  bool("seekPauses");
  bool("showReviewHints");
  if (Array.isArray(r.beforeAfterTypes)) out.beforeAfterTypes = r.beforeAfterTypes.map((x) => String(x).toUpperCase());
  if (r.ptsOffsetMode === "none" || r.ptsOffsetMode === "meta_t0") out.ptsOffsetMode = r.ptsOffsetMode;
  return out;
}

/** Загрузить /ui-config.json (не дольше timeoutMs). Ошибки → значения по умолчанию. */
export async function loadUiConfig(timeoutMs = 1500): Promise<UiConfig> {
  try {
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : undefined;
    const t = setTimeout(() => ctrl?.abort(), timeoutMs);
    const base = (import.meta as { env?: Record<string, string | undefined> }).env?.BASE_URL ?? "/";
    const res = await fetch(`${base}ui-config.json`, { cache: "no-store", signal: ctrl?.signal });
    clearTimeout(t);
    if (res.ok) current = mergeUiConfig(DEFAULT_UI_CONFIG, await res.json());
  } catch {
    /* по умолчанию */
  }
  return current;
}

/** Для тестов. */
export function setUiConfig(c: Partial<UiConfig>) {
  current = { ...current, ...c };
}
