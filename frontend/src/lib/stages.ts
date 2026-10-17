// Стадии пайплайна (A03) → шаги степпера. Имена стадий бэкенда заранее неизвестны,
// поэтому сопоставляем по ключевым словам; неизвестная стадия не ломает UI.

export interface StepDef {
  id: string;
  title: string;
  hint: string;
  keys: string[];
}

export const STEPS: StepDef[] = [
  {
    id: "upload",
    title: "Приём видео",
    hint: "Файл передан на локальный сервер, задача в очереди",
    keys: ["upload", "queue", "queued", "pending", "accept", "start", "очеред", "приём"],
  },
  {
    id: "ingest",
    title: "Декодирование",
    hint: "Чтение .mov, учёт поворота, выборка кадров",
    keys: ["ingest", "decode", "load", "sampl", "read", "probe", "video", "декодир", "загрузка видео"],
  },
  {
    id: "perception",
    title: "Люди и предметы",
    hint: "Детекция, трекинг, роли по зонам и предметам",
    keys: ["detect", "track", "pose", "person", "object", "zone", "поиск люд", "трекинг", "поз", "объект"],
  },
  {
    id: "scene",
    title: "Состояние сцены",
    hint: "Сейф, дверь, наличные, присутствие кассира",
    keys: ["scene", "state", "signal", "сцен"],
  },
  {
    id: "reasoning",
    title: "Анализ событий",
    hint: "Правила, временная модель, локальная VLM",
    keys: ["rule", "temporal", "refine", "vlm", "verify", "discover", "event", "правил", "временн", "уточнен"],
  },
  {
    id: "report",
    title: "Формирование отчёта",
    hint: "Кадры-доказательства, нормы РК, вердикт",
    keys: ["fusion", "fuse", "report", "evidence", "norm", "final", "serializ", "done", "finish", "отчёт", "отчет", "готово"],
  },
];

/** Индекс шага по строке stage и прогрессу 0..1. */
export function stepIndex(stage: string | undefined, progress: number | undefined): number {
  const s = (stage ?? "").toLowerCase();
  if (s) {
    for (let i = STEPS.length - 1; i >= 0; i--) {
      if (STEPS[i].keys.some((k) => s.includes(k))) return i;
    }
  }
  const p = progress ?? 0;
  if (p <= 0) return s ? 1 : 0;
  // Без понятной стадии — равномерно по прогрессу (шаг 0 = приём уже пройден).
  return Math.min(STEPS.length - 1, 1 + Math.floor(p * (STEPS.length - 1)));
}

/** Человекочитаемое имя стадии бэкенда. */
export function stageLabel(stage: string | undefined): string {
  if (!stage) return "";
  const s = stage.toLowerCase();
  const map: [string, string][] = [
    ["queue", "в очереди"],
    ["ingest", "декодирование видео"],
    ["decode", "декодирование видео"],
    ["detect", "детекция объектов"],
    ["track", "трекинг людей"],
    ["pose", "оценка позы"],
    ["scene", "состояние сцены"],
    ["rule", "движок правил"],
    ["temporal", "временная модель"],
    ["vlm", "проверка локальной VLM"],
    ["verify", "верификация событий"],
    ["discover", "поиск новых типов"],
    ["fusion", "слияние сигналов"],
    ["fuse", "слияние сигналов"],
    ["evidence", "кадры-доказательства"],
    ["norm", "подбор норм"],
    ["done", "готово"],
  ];
  for (const [k, v] of map) if (s.includes(k)) return v;
  // A03 присылает готовые русские подписи («Поиск людей и объектов») — показываем как есть, со строчной.
  return /[а-яё]/i.test(stage) ? stage.charAt(0).toLowerCase() + stage.slice(1) : stage;
}
