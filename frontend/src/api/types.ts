// Типы ответа API Kassa Sentinel.
// Обязательные поля — строго по ТЗ; остальные — доп. поля контракта v1
// (contracts/sentinel_contracts.py → contracts.serialize.result_to_api).

export type JobStatus = "queued" | "processing" | "done" | "failed";
export type Verdict = "violation" | "no_violation";

/** Кадр-доказательство. image_base64 — JPEG base64 (без префикса data:) или data-URI. */
export interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
  /**
   * Только UI: сервер сообщил кадр, но без изображения (A02 у failed заменяет base64 на "<jpeg omitted>").
   * Такой кадр НЕ рисуется и ничем не подменяется — показывается плашка «кадр не передан».
   */
  missing?: boolean;
}

/** Ссылка на норму РК — только из norms/catalog.yaml (A13). */
export interface RuleRef {
  document: string;
  clause?: string | null;
  url: string;
  quote?: string;
  verified?: boolean;
  /** A12 (meta.ui.evidence_overlays[].norms[].status_ru): статус сверки по-русски. */
  status_ru?: string;
}

export interface Violation {
  // --- обязательные (ТЗ) ---
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
  // --- доп. поля контракта ---
  type_id?: string;
  confidence?: number;
  t_start?: number;
  t_end?: number;
  rules?: RuleRef[];
  // --- необязательные расширения (если бэкенд их отдаёт) ---
  facts?: string[];
  sources?: string[];
  severity?: string;
  /** Исходный порядковый номер в ответе сервера (A12 ссылается на него в meta.ui.evidence_overlays). */
  index?: number;
}

export type TimelinePoint = [number, number]; // (t сек, risk 0..1)

export interface JobMeta {
  timings?: Record<string, number>;
  models?: unknown;
  warnings?: string[];
  version?: string;
  duration?: number;
  file_name?: string;
  [key: string]: unknown;
}

export interface JobResult {
  job_id: string;
  status: JobStatus;
  verdict?: Verdict | null;
  violations: Violation[];
  timeline: TimelinePoint[];
  meta: JobMeta;
  /** 0..1 */
  progress?: number;
  stage?: string;
  error?: string;
  /** A02: позиция в очереди (processing/queued). */
  queuePosition?: number;
  /** A02: секунд с постановки задачи (реальный источник для «прошло»). */
  elapsedSec?: number;
  /** A02 (только /ui/v1): у failed — частичный результат, если пайплайн успел что-то найти. */
  partialResult?: JobResult;
  /** A01 UI-report: violation | no_violation | unknown | pending. */
  verdictState?: string;
  /** A01 UI-report: полнота подтверждена (coverage_problem = null). */
  analysisComplete?: boolean;
  /** A01 UI-report: почему «нет нарушения» не подтверждено. */
  coverageProblem?: string;
  /** A01 UI-report: найденные, но не отданные жюри нарушения (ниже порога / не подтверждено). */
  candidates?: Violation[];
  /** A01 UI-report: предупреждения верхнего уровня. */
  warnings?: string[];
}

export interface AskResponse {
  answer: string;
  sources?: string[];
}

export interface SessionInfo {
  token: string | null;
  /** Сервер ответил (любой HTTP-статус). */
  reachable: boolean;
  /** Есть действующая сессия (HttpOnly cookie) или API-ключ в памяти вкладки. */
  authenticated: boolean;
  features: { copilot?: boolean; [k: string]: unknown };
  version?: string;
}
