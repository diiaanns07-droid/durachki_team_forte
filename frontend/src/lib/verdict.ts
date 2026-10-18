import type { JobResult, Verdict } from "../api/types";
import type { Completeness } from "./completeness";

// Формулировки ТЗ — дословно.
export const VERDICT_TEXT: Record<Verdict, string> = {
  violation: "Обнаружены признаки возможных нарушений",
  no_violation: "Нарушение не обнаружено",
};

export interface VerdictView {
  /** success (зелёный) — ТОЛЬКО для завершённого анализа с подтверждённой полнотой и без нарушений. */
  tone: "danger" | "success" | "warning" | "neutral";
  title: string;
  subtitle: string;
  /** Короткие пометки на баннере: «НЕПОЛНЫЙ АНАЛИЗ», «ПОЛНОТА НЕ ПОДТВЕРЖДЕНА», «MOCK». */
  flags: string[];
}

/**
 * completeness: "complete" | "partial" | "unknown" (см. lib/completeness.ts).
 * Для совместимости принимает boolean: true → partial, false → complete.
 */
export function verdictView(
  job: Pick<JobResult, "verdict" | "violations" | "status">,
  completeness: Completeness | boolean = "unknown",
  mock = false,
): VerdictView {
  const state: Completeness = typeof completeness === "boolean" ? (completeness ? "partial" : "complete") : completeness;
  const flags: string[] = [];
  if (mock) flags.push("MOCK: модели не запускались");
  if (state === "partial") flags.push("НЕПОЛНЫЙ АНАЛИЗ");
  if (state === "unknown" && job.status === "done") flags.push("ПОЛНОТА НЕ ПОДТВЕРЖДЕНА");

  if (job.status === "failed") {
    return {
      tone: "warning",
      title: "Анализ не завершён",
      subtitle: "Сервер сообщил об ошибке обработки. Вывод об отсутствии нарушений сделать нельзя — нужен ручной просмотр.",
      flags,
    };
  }
  if (job.status !== "done") {
    return { tone: "neutral", title: "Анализ выполняется", subtitle: "Результат появится после завершения обработки.", flags };
  }
  if (mock) {
    return {
      tone: "neutral",
      title: "Тестовый результат: модели не запускались",
      subtitle: `Сервер работает в mock-режиме, это не анализ видео. Mock-вердикт: «${
        job.verdict === "violation" ? VERDICT_TEXT.violation : job.verdict === "no_violation" ? VERDICT_TEXT.no_violation : "нет"
      }».`,
      flags,
    };
  }
  if (job.verdict === "violation") {
    const n = job.violations.length;
    const tail =
      n === 0
        ? "Требуется просмотр записи проверяющим."
        : `${n} ${n === 1 ? "эпизод" : n < 5 ? "эпизода" : "эпизодов"} для проверки. Результат — подсказка, не юридическое заключение.`;
    return {
      tone: "danger",
      title: VERDICT_TEXT.violation,
      subtitle: state === "partial" ? `${tail} Анализ неполный — часть записи или стадий не проверена.` : tail,
      flags,
    };
  }
  if (job.verdict === "no_violation") {
    if (state === "partial") {
      return {
        tone: "warning",
        title: "Анализ неполный",
        subtitle: "Признаков нарушений не найдено, но часть записи или стадий не проверена. Нужен ручной просмотр.",
        flags,
      };
    }
    if (state === "unknown") {
      return {
        tone: "neutral",
        title: VERDICT_TEXT.no_violation,
        subtitle: "Сервер не подтвердил полноту анализа — уверенная отметка не ставится. См. блок «Полнота анализа».",
        flags,
      };
    }
    return {
      tone: "success",
      title: VERDICT_TEXT.no_violation,
      subtitle: "Видимых признаков нарушений регламента в записи не выявлено.",
      flags,
    };
  }
  return { tone: "warning", title: "Вердикт не получен", subtitle: "Сервер не вернул вердикт — нужен ручной просмотр.", flags };
}

/**
 * Строка «Недавние проверки». Уверенный зелёный «без нарушений» — только при подтверждённой полноте
 * и не для mock; старые записи без completeness считаются «полнота не подтверждена».
 */
export function historyView(h: {
  verdict?: Verdict | null;
  n_violations?: number;
  status?: string;
  completeness?: Completeness;
  mock?: boolean;
}): { tone: "danger" | "success" | "warning" | "neutral"; label: string } {
  if (h.mock) return { tone: "neutral", label: "тестовый результат (mock)" };
  if (h.verdict === "violation") return { tone: "danger", label: `нарушений: ${h.n_violations ?? "?"}` };
  if (h.verdict === "no_violation") {
    if (h.completeness === "complete") return { tone: "success", label: "без нарушений" };
    if (h.completeness === "partial") return { tone: "warning", label: "анализ неполный" };
    return { tone: "neutral", label: "не обнаружено · полнота не подтверждена" };
  }
  if (h.status === "failed") return { tone: "warning", label: "анализ не завершён" };
  if (h.status === "done") return { tone: "neutral", label: "готово" };
  return { tone: "neutral", label: "в обработке" };
}
