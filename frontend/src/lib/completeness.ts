// Полнота анализа: «полный / неполный / неизвестно». unknown ≠ «нарушений нет»:
// зелёный «Нарушение не обнаружено» допустим ТОЛЬКО при подтверждённой полноте.
// Источники (все необязательные, разбор защитный):
//   A03: meta.complete / meta.partial / meta.degraded, meta.stages{имя:{status,seconds,required,origin,error}},
//        meta.required_stages, meta.deadline_exceeded_s, meta.mock
//   A04: meta.coverage (доля декодированной длительности 0..1; complete при ≥ 0.98)
//   A02: status failed (+ partial для UI), предупреждения
import type { JobResult } from "../api/types";

export type Completeness = "complete" | "partial" | "unknown";

export interface StageInfo {
  name: string;
  label: string;
  status: string;
  statusLabel: string;
  ok: boolean;
  seconds?: number;
  required: boolean;
  error?: string;
  vramPeakMb?: number;
  /** Доля длительности, реально просмотренная стадией (A03 meta.coverage[stage]); null — неизвестно. */
  coverage?: number | null;
}

export interface CompletenessReport {
  state: Completeness;
  /** Доля записи, реально декодированная и проанализированная (0..1), если сервер её сообщил. */
  coverage?: number;
  /** Почему неполный / почему неизвестно — по-русски, без домыслов. */
  reasons: string[];
  stages: StageInfo[];
  /** Результат mock/fake-пайплайна: модели не запускались. */
  mock: boolean;
  /** Стадии, взятые из кэша (тот же файл, повторный запуск). */
  cached: string[];
}

/** Порог полноты декодирования — как warn_coverage у A04 и min_ratio_for_no_violation у A01 (0.98). */
export const COVERAGE_COMPLETE = 0.98;
/** Порог покрытия обязательной стадии — как coverage.min_required у A03 (0.9). */
export const STAGE_COVERAGE_MIN = 0.9;

const STAGE_LABELS: Record<string, string> = {
  ingest: "Декодирование видео",
  detect: "Поиск людей и объектов",
  track: "Трекинг",
  pose: "Анализ поз",
  scene: "Состояние сцены",
  rules: "Правила",
  temporal: "Временная модель",
  refine: "Уточнение окон",
  vlm_verify: "Проверка эпизодов VLM",
  vlm_discover: "Поиск новых типов VLM",
  vlm: "Проверка VLM",
  fusion: "Формирование отчёта",
  evidence: "Кадры-доказательства",
  norms: "Нормы",
};

const STATUS_LABELS: Record<string, string> = {
  ok: "выполнена",
  done: "выполнена",
  cached: "из кэша",
  error: "ошибка",
  failed: "ошибка",
  timeout: "таймаут",
  crashed: "сбой",
  missing: "модуль не найден",
  skipped: "пропущена",
  disabled: "выключена",
  pending: "не запускалась",
};

const OK_STATUSES = new Set(["ok", "done", "cached"]);

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

function numOrUndef(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}

export function stageLabelRu(name: string): string {
  return STAGE_LABELS[name] ?? name;
}

/** coverage: число, «87%», {ratio|fraction|value|coverage} — в 0..1. */
export function readCoverage(meta: Obj): number | undefined {
  const cands: unknown[] = [meta.coverage];
  if (isObj(meta.video)) cands.push(meta.video.coverage);
  if (isObj(meta.ingest)) cands.push(meta.ingest.coverage);
  for (let c of cands) {
    // A01: {ratio, complete, stages, required}; A03: {стадия: доля|null} — общая доля = ingest.
    if (isObj(c)) c = c.ratio ?? c.fraction ?? c.value ?? c.coverage ?? c.ingest;
    if (typeof c === "string") c = c.replace("%", "").trim();
    let n = numOrUndef(c);
    if (n === undefined) continue;
    if (n > 1 && n <= 100) n /= 100;
    if (n >= 0 && n <= 1) return n;
  }
  return undefined;
}

/** A03: meta.coverage = {стадия: доля 0..1 | null}. Объект A01 ({ratio,…}) сюда не относится. */
export function readStageCoverage(meta: Obj): Record<string, number | null> {
  let c = meta.coverage;
  // Канонический формат A01/A03 (2055fc4): {ratio, …, by_stage: {стадия: доля|null}}.
  if (isObj(c) && "ratio" in c) c = c.by_stage;
  if (!isObj(c)) return {};
  const out: Record<string, number | null> = {};
  for (const [k, v] of Object.entries(c)) {
    if (v === null) out[k] = null;
    else {
      const n = numOrUndef(v);
      if (n !== undefined && n >= 0 && n <= 1) out[k] = n;
    }
  }
  return out;
}

export function readStages(meta: Obj): StageInfo[] {
  const st = meta.stages;
  if (!isObj(st)) return [];
  const cov = readStageCoverage(meta);
  const required = new Set(Array.isArray(meta.required_stages) ? meta.required_stages.map(String) : []);
  return Object.entries(st)
    .filter(([, v]) => isObj(v))
    .map(([name, v]) => {
      const r = v as Obj;
      const status = String(r.status ?? "pending").toLowerCase();
      return {
        name,
        label: stageLabelRu(name),
        status,
        statusLabel: STATUS_LABELS[status] ?? status,
        ok: OK_STATUSES.has(status),
        seconds: numOrUndef(r.seconds),
        required: r.required === true || required.has(name),
        error: typeof r.error === "string" && r.error ? r.error : undefined,
        vramPeakMb: numOrUndef(r.vram_peak_mb),
        coverage: name in cov ? cov[name] : undefined,
      };
    });
}

export function isMockResult(job: JobResult): boolean {
  const m = job.meta as Obj;
  if (m.mock === true || m.fake === true || m.pipeline === "fake" || m.pipeline === "mock") return true;
  if (isObj(m.stages) && Object.values(m.stages).some((s) => isObj(s) && String(s.origin ?? "").toLowerCase() === "mock")) return true;
  const warnings = Array.isArray(m.warnings) ? m.warnings.map((w) => String(w).toLowerCase()) : [];
  return warnings.some((w) => w.includes("fake pipeline") || w.includes("mock pipeline") || w.includes("фейковый пайплайн"));
}

const PARTIAL_WARNING = /partial|timeout|timed out|skipped|неполн|таймаут|пропущ|не заверш/;

export function analyzeCompleteness(job: JobResult): CompletenessReport {
  const m = job.meta as Obj;
  const reasons: string[] = [];
  const stages = readStages(m);
  const coverage = readCoverage(m);

  if (job.status === "failed") reasons.push("анализ завершился ошибкой");
  if (m.partial === true || job.partialResult) reasons.push("сервер вернул частичный результат");
  if (m.complete === false || m.incomplete === true) reasons.push("сервер отметил анализ как неполный");
  if (m.degraded === true) reasons.push("часть стадий работала в упрощённом режиме");
  if (m.timed_out === true || (numOrUndef(m.deadline_exceeded_s) ?? 0) > 0) reasons.push("превышен бюджет времени анализа");
  if (coverage !== undefined && coverage < COVERAGE_COMPLETE)
    reasons.push(`проанализировано ${Math.floor(coverage * 100)} % длительности записи`);
  if (isObj(m.coverage) && m.coverage.complete === false) reasons.push("покрытие записи отмечено сервером как неполное");
  for (const s of stages) {
    if (s.required && !s.ok) reasons.push(`обязательная стадия «${s.label}»: ${s.statusLabel}`);
    if (s.required && s.ok && typeof s.coverage === "number" && s.coverage < STAGE_COVERAGE_MIN)
      reasons.push(`обязательная стадия «${s.label}» просмотрела ${Math.floor(s.coverage * 100)} % записи`);
  }
  // A01 UI-report: причина, по которой «нет нарушения» не подтверждено.
  let unknownReason: string | undefined;
  if (job.coverageProblem) {
    if (/проанализировано|незаверш|complete=false|обязательная стадия|неполн/i.test(job.coverageProblem)) reasons.push(job.coverageProblem);
    else unknownReason = job.coverageProblem;
  }
  for (const k of ["skipped_stages", "stages_skipped", "failed_stages"]) {
    const v = m[k];
    if (Array.isArray(v) && v.length) reasons.push(`пропущены стадии: ${v.map((x) => stageLabelRu(String(x))).join(", ")}`);
  }
  const warnings = [...(Array.isArray(m.warnings) ? m.warnings.map(String) : []), ...(job.warnings ?? [])];
  for (const w of warnings) if (PARTIAL_WARNING.test(w.toLowerCase())) reasons.push(`предупреждение сервера: ${w}`);

  const uniq = [...new Set(reasons)];
  const cached = stages.filter((s) => s.status === "cached").map((s) => s.label);
  const mock = isMockResult(job);
  if (uniq.length) return { state: "partial", coverage, reasons: uniq, stages, mock, cached };

  const requiredKnownOk = stages.length > 0 && stages.filter((s) => s.required).every((s) => s.ok);
  // A01 analysis_complete=true — полнота подтверждена сериализатором; A01 false/коверидж-проблема — не подтверждена.
  if (job.analysisComplete === false || unknownReason) {
    return { state: "unknown", coverage, reasons: [unknownReason ?? "сервер не подтвердил полноту анализа"], stages, mock, cached };
  }
  const confirmed =
    job.analysisComplete === true || m.complete === true || (coverage !== undefined && coverage >= COVERAGE_COMPLETE && requiredKnownOk);
  if (confirmed) return { state: "complete", coverage, reasons: [], stages, mock, cached };
  return {
    state: "unknown",
    coverage,
    reasons: ["сервер не подтвердил полноту анализа (нет meta.complete / meta.coverage)"],
    stages,
    mock,
    cached,
  };
}
