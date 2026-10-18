// Защитный разбор ответа GET /…/jobs/{id}. Бэкенд собирают 18 агентов —
// UI не должен падать от лишних/пропущенных полей, строковых чисел и т.п.
import { typeInfo } from "../lib/catalog";
import { analyzeCompleteness } from "../lib/completeness";
import type {
  EvidenceFrame,
  JobMeta,
  JobResult,
  JobStatus,
  RuleRef,
  TimelinePoint,
  Verdict,
  Violation,
} from "./types";

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

export function num(v: unknown, fallback = 0): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v.replace(",", "."));
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function optNum(v: unknown): number | undefined {
  const n = num(v, Number.NaN);
  return Number.isFinite(n) ? n : undefined;
}

function str(v: unknown, fallback = ""): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return fallback;
}

function strList(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((x) => str(x)).filter((s) => s.trim() !== "");
  if (typeof v === "string" && v.trim() !== "") return [v];
  return [];
}

export function normalizeStatus(v: unknown): JobStatus {
  const s = str(v).toLowerCase().trim();
  if (["done", "completed", "complete", "finished", "success", "succeeded", "ok"].includes(s)) return "done";
  if (["failed", "error", "failure", "cancelled", "canceled", "timeout"].includes(s)) return "failed";
  if (["queued", "pending", "waiting", "accepted"].includes(s)) return "queued";
  return "processing";
}

export function normalizeVerdict(v: unknown, violations: Violation[] = []): Verdict | null {
  if (typeof v === "boolean") return v ? "violation" : "no_violation";
  const s = str(v).toLowerCase().trim().replace(/[\s-]+/g, "_");
  if (["violation", "violations", "violated", "true", "1", "нарушение"].includes(s)) return "violation";
  if (["no_violation", "noviolation", "clean", "ok", "none", "false", "0", "нет_нарушения"].includes(s))
    return "no_violation";
  if (s === "") return null;
  // Неизвестная строка: решаем по наличию нарушений.
  return violations.length > 0 ? "violation" : null;
}

/** Прогресс: 0..1 или 0..100 → 0..1. */
export function normalizeProgress(v: unknown): number | undefined {
  const n = optNum(v);
  if (n === undefined) return undefined;
  const p = n > 1 ? n / 100 : n;
  return Math.min(1, Math.max(0, p));
}

/**
 * Похоже ли значение на изображение: data:/blob:/http(s)-URL, путь или base64 (≥ 16 символов алфавита base64).
 * Заглушки вроде "<jpeg omitted>" (A02 у failed) — НЕ изображение.
 */
export function isImagePayload(s: string): boolean {
  const t = s.trim();
  if (!t) return false;
  if (/^data:image\//i.test(t) || t.startsWith("blob:") || /^https?:\/\//i.test(t)) return true;
  if (t.startsWith("/") && /\.(jpe?g|png|webp)(\?|$)/i.test(t)) return true;
  return t.length >= 16 && /^[A-Za-z0-9+/=\s]+$/.test(t);
}

function normalizeFrame(v: unknown): EvidenceFrame | null {
  if (typeof v === "string") return isImagePayload(v) ? { timestamp_sec: 0, image_base64: v } : null;
  if (!isObj(v)) return null;
  const img = str(v.image_base64 ?? v.image ?? v.jpeg_base64 ?? v.data ?? v.url);
  const t = num(v.timestamp_sec ?? v.t ?? v.timestamp, 0);
  if (!img) return null;
  // Кадр заявлен, но изображения нет: сохраняем момент, картинку не подставляем.
  if (!isImagePayload(img)) return { timestamp_sec: t, image_base64: "", missing: true };
  return { timestamp_sec: t, image_base64: img };
}

function normalizeRule(v: unknown): RuleRef | null {
  if (!isObj(v)) return null;
  const document = str(v.document ?? v.doc ?? v.title);
  const url = str(v.url ?? v.link ?? v.href);
  if (!document && !url) return null;
  const clause = v.clause === null || v.clause === undefined ? null : str(v.clause);
  return {
    document: document || url,
    clause,
    url,
    quote: str(v.quote ?? v.text),
    verified: v.verified === true || v.verified === "true",
  };
}

export function normalizeViolation(v: unknown): Violation | null {
  if (!isObj(v)) return null;
  const timecode = num(v.timecode_sec ?? v.t_peak ?? v.timecode ?? v.t, 0);
  const frames = Array.isArray(v.evidence_frames)
    ? v.evidence_frames.map(normalizeFrame).filter((f): f is EvidenceFrame => f !== null)
    : [];
  const rulesRaw: unknown[] = Array.isArray(v.rules) && v.rules.length ? v.rules : isObj(v.rule) ? [v.rule] : [];
  const rules = rulesRaw.map(normalizeRule).filter((r): r is RuleRef => r !== null);
  const sources = strList(v.sources ?? v.source);
  const out: Violation = {
    type: str(v.type, "") || str(v.type_name, "") || str(v.type_id, "Возможное нарушение"),
    timecode_sec: Math.max(0, timecode),
    description: str(v.description),
    evidence_frames: frames,
    type_id: str(v.type_id) || undefined,
    confidence: optNum(v.confidence),
    t_start: optNum(v.t_start),
    t_end: optNum(v.t_end),
    rules,
    facts: strList(v.facts),
    sources,
    severity: str(v.severity) || undefined,
  };
  if (out.confidence !== undefined) out.confidence = Math.min(1, Math.max(0, out.confidence));
  return out;
}

export function normalizeTimeline(v: unknown): TimelinePoint[] {
  if (!Array.isArray(v)) return [];
  const pts: TimelinePoint[] = [];
  for (const p of v) {
    let t: number | undefined;
    let r: number | undefined;
    if (Array.isArray(p) && p.length >= 2) {
      t = optNum(p[0]);
      r = optNum(p[1]);
    } else if (isObj(p)) {
      t = optNum(p.t ?? p.time ?? p.timestamp_sec);
      r = optNum(p.risk ?? p.score ?? p.value);
    }
    if (t === undefined || r === undefined || t < 0) continue;
    pts.push([t, Math.min(1, Math.max(0, r))]);
  }
  pts.sort((a, b) => a[0] - b[0]);
  return pts;
}

/** Главная функция: сырой JSON → JobResult. Бросает Error, если это не похоже на ответ API. */
export function normalizeJob(raw: unknown, fallbackJobId = ""): JobResult {
  if (!isObj(raw)) throw new Error("Некорректный ответ сервера: ожидался JSON-объект");
  const violations = Array.isArray(raw.violations)
    ? raw.violations
        .map((x, i) => {
          const v = normalizeViolation(x);
          if (v) v.index = i;
          return v;
        })
        .filter((v): v is Violation => v !== null)
    : [];
  violations.sort((a, b) => a.timecode_sec - b.timecode_sec);
  const status = normalizeStatus(raw.status);
  let verdict = normalizeVerdict(raw.verdict, violations);
  if (status === "done" && verdict === null) verdict = violations.length > 0 ? "violation" : "no_violation";
  const meta: JobMeta = isObj(raw.meta) ? { ...(raw.meta as JobMeta) } : {};
  // A01 UI-report дублирует coverage на верхний уровень — подставляем, если в meta его нет.
  if (meta.coverage === undefined && raw.coverage !== undefined) meta.coverage = raw.coverage;
  const candidates = Array.isArray(raw.candidates)
    ? raw.candidates
        .map((x, i) => {
          const v = normalizeViolation(x);
          if (v) v.index = i;
          return v;
        })
        .filter((v): v is Violation => v !== null)
        .sort((a, b) => a.timecode_sec - b.timecode_sec)
    : [];
  const error = str(raw.error ?? raw.detail ?? meta.error ?? "") || undefined;
  let partialResult: JobResult | undefined;
  if (status === "failed" && isObj(raw.partial)) {
    try {
      // A02 (2055fc4): partial = {code, complete, partial: <partial.json A03>, rejected_result: <отвергнутый отчёт>};
      // раньше — сам отчёт. Берём отвергнутый отчёт (полный meta), иначе partial.json, иначе объект как есть.
      const wrap = raw.partial;
      const inner = isObj(wrap.rejected_result) ? wrap.rejected_result : isObj(wrap.partial) ? wrap.partial : wrap;
      partialResult = normalizeJob({ ...inner, status: "done" }, fallbackJobId);
    } catch {
      partialResult = undefined;
    }
  }
  const qp = optNum(raw.queue_position);
  const el = optNum(raw.elapsed_s);
  return {
    job_id: str(raw.job_id ?? raw.id, fallbackJobId),
    verdictState: str(raw.verdict_state) || undefined,
    analysisComplete: typeof raw.analysis_complete === "boolean" ? raw.analysis_complete : undefined,
    coverageProblem: str(raw.coverage_problem) || undefined,
    candidates,
    warnings: strList(raw.warnings),
    queuePosition: qp,
    elapsedSec: el,
    partialResult,
    status,
    verdict,
    violations,
    timeline: normalizeTimeline(raw.timeline),
    meta,
    progress: status === "done" ? 1 : normalizeProgress(raw.progress),
    stage: str(raw.stage ?? meta.stage) || undefined,
    error,
  };
}

/** image_base64 → src для <img>: распознаём JPEG/PNG/SVG/WebP, data-URI и URL. */
export function imageSrc(b64: string): string {
  const s = b64.trim();
  if (s.startsWith("data:") || s.startsWith("blob:") || /^https?:\/\//.test(s)) return s;
  // JPEG base64 начинается с "/9j/" — это НЕ путь. Путь — если есть символы вне алфавита base64.
  if (s.startsWith("/") && !/^[A-Za-z0-9+/=\s]+$/.test(s)) return s;
  if (s.startsWith("iVBOR")) return `data:image/png;base64,${s}`;
  if (s.startsWith("PHN2") || s.startsWith("PD94")) return `data:image/svg+xml;base64,${s}`;
  if (s.startsWith("UklGR")) return `data:image/webp;base64,${s}`;
  return `data:image/jpeg;base64,${s}`;
}

/** Длительность видео: meta.duration → последняя точка timeline → последний t_end. */
export function inferDuration(job: JobResult): number {
  const m = job.meta;
  const fromMeta =
    optNum(m.duration) ??
    optNum(m.duration_sec) ??
    (isObj(m.video) ? optNum((m.video as Obj).duration) : undefined);
  if (fromMeta && fromMeta > 0) return fromMeta;
  let d = 0;
  for (const [t] of job.timeline) d = Math.max(d, t);
  for (const v of job.violations) d = Math.max(d, v.t_end ?? 0, v.timecode_sec + 1);
  return d;
}

/** A12: meta.ui.evidence_overlays[] — факты, источники (dict источник→вес), состояние VLM по нарушению. */
export function violationOverlay(v: Violation, job: JobResult): Obj | undefined {
  const ui = (job.meta as Obj).ui;
  const list = isObj(ui) ? ui.evidence_overlays : undefined;
  if (!Array.isArray(list)) return undefined;
  const objs = list.filter(isObj);
  if (v.index !== undefined) {
    const byIdx = objs.find((o) => num(o.violation_index, -1) === v.index);
    if (byIdx && (!v.type_id || !o_type(byIdx) || o_type(byIdx) === v.type_id)) return byIdx;
  }
  return objs.find((o) => (!v.type_id || o_type(o) === v.type_id) && Math.abs(num(o.timecode_sec, -1e9) - v.timecode_sec) < 0.6);
}

function o_type(o: Obj): string {
  return str(o.type_id);
}

/** Список источников: массив строк или словарь {источник: вес} (по убыванию веса). */
function sourceNames(x: unknown): string[] {
  if (isObj(x)) {
    return Object.entries(x)
      .sort((a, b) => num(b[1]) - num(a[1]))
      .map(([k]) => k);
  }
  return strList(x);
}

/** Состояние проверки VLM (A12: confirm | reject | reject_weak | unsure). */
export function violationVlm(v: Violation, job: JobResult): { label: string; tone: "ok" | "warn" | "muted" } | null {
  const o = violationOverlay(v, job);
  const st = o ? str(o.vlm).toLowerCase() : "";
  if (!st) return null;
  if (st.startsWith("confirm")) return { label: "VLM подтвердила", tone: "ok" };
  if (st.startsWith("reject")) return { label: "VLM не подтвердила", tone: "warn" };
  if (st === "unsure") return { label: "VLM не уверена", tone: "muted" };
  return { label: `VLM: ${st}`, tone: "muted" };
}

/** Источники события: из самого нарушения, из meta.ui.evidence_overlays (A12) или из meta.events. */
export function violationSources(v: Violation, job: JobResult): string[] {
  const own = v.sources ?? [];
  if (own.length) return own;
  const ov = violationOverlay(v, job);
  if (ov) {
    const names = sourceNames(ov.sources);
    if (names.length) return names;
  }
  const events = (job.meta as Obj).events;
  if (Array.isArray(events)) {
    const found = new Set<string>();
    for (const e of events) {
      if (!isObj(e)) continue;
      if (v.type_id && str(e.type_id) !== v.type_id) continue;
      const tp = num(e.t_peak ?? e.timecode_sec, Number.NaN);
      if (Number.isFinite(tp) && Math.abs(tp - v.timecode_sec) > 15) continue;
      sourceNames(e.source ?? e.sources).forEach((s) => found.add(s));
    }
    return [...found];
  }
  return [];
}

/** Факты: явные facts → meta.ui.evidence_overlays (A12) → meta.events → предложения описания. */
export function violationFacts(v: Violation, job: JobResult): string[] {
  if (v.facts && v.facts.length) return v.facts;
  const ov = violationOverlay(v, job);
  if (ov) {
    const f = strList(ov.facts);
    if (f.length) return f;
  }
  const events = (job.meta as Obj).events;
  if (Array.isArray(events)) {
    const facts: string[] = [];
    for (const e of events) {
      if (!isObj(e) || (v.type_id && str(e.type_id) !== v.type_id)) continue;
      const tp = num(e.t_peak ?? e.timecode_sec, Number.NaN);
      if (Number.isFinite(tp) && Math.abs(tp - v.timecode_sec) > 15) continue;
      facts.push(...strList(e.facts));
    }
    if (facts.length) return [...new Set(facts)];
  }
  return v.description
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 3)
    .slice(0, 5);
}

/**
 * Анализ неполный? Делегирует в lib/completeness.ts (tri-state: complete / partial / unknown).
 * Оставлено для совместимости: true только для подтверждённо НЕПОЛНОГО анализа.
 */
export function isPartial(job: JobResult): boolean {
  return analyzeCompleteness(job).state === "partial";
}

/** Внешний JSON жюри: строго обязательные поля ТЗ (без timeline/meta/норм). */
export function toJuryJson(job: JobResult): Record<string, unknown> {
  return {
    job_id: job.job_id,
    status: job.status,
    verdict: job.verdict ?? null,
    violations: job.violations.map((v) => ({
      type: v.type,
      timecode_sec: v.timecode_sec,
      description: v.description,
      evidence_frames: v.evidence_frames
        .filter((f) => !f.missing)
        .map((f) => ({ timestamp_sec: f.timestamp_sec, image_base64: f.image_base64 })),
    })),
  };
}

export interface ReviewHint {
  type: string;
  type_id?: string;
  timecode_sec: number;
  t_start?: number;
  t_end?: number;
  reasons: string[];
  facts: string[];
}

/**
 * A12: meta.ui.review_hints — кандидаты ниже порога/подавленные. В вердикт НЕ входят;
 * UI показывает их отдельно как «стоит просмотреть», чтобы проверяющий не пропустил пограничный момент.
 */
export function reviewHints(job: JobResult): ReviewHint[] {
  const ui = (job.meta as Obj).ui;
  const list = isObj(ui) ? ui.review_hints : undefined;
  if (!Array.isArray(list)) return [];
  const out: ReviewHint[] = [];
  for (const h of list) {
    if (!isObj(h)) continue;
    const t = optNum(h.timecode_sec ?? h.t_peak);
    if (t === undefined || t < 0) continue;
    out.push({
      type: str(h.type) || str(h.type_id) || "Возможный эпизод",
      type_id: str(h.type_id) || undefined,
      timecode_sec: t,
      t_start: optNum(h.t_start),
      t_end: optNum(h.t_end),
      reasons: strList(h.reasons_ru).length ? strList(h.reasons_ru) : strList(h.reasons),
      facts: strList(h.facts),
    });
  }
  return out.sort((a, b) => a.timecode_sec - b.timecode_sec);
}

/** Рамка подсветки (A12 shapes kind=box, bbox нормирован к кадру: x1,y1,x2,y2). */
export interface HighlightBox {
  label: string;
  color: string;
  bbox: [number, number, number, number];
}

export function overlayBoxes(v: Violation, job: JobResult): HighlightBox[] {
  const ov = violationOverlay(v, job);
  if (!ov || !Array.isArray(ov.shapes)) return [];
  const out: HighlightBox[] = [];
  for (const sh of ov.shapes) {
    if (!isObj(sh) || str(sh.kind) !== "box" || !Array.isArray(sh.bbox) || sh.bbox.length !== 4) continue;
    const b = sh.bbox.map((x) => num(x, Number.NaN));
    if (b.some((x) => !Number.isFinite(x))) continue;
    const [x1, y1, x2, y2] = b.map((x) => Math.min(1, Math.max(0, x)));
    if (x2 <= x1 || y2 <= y1) continue;
    const color = /^#[0-9a-f]{3,8}$/i.test(str(sh.color)) ? str(sh.color) : "#ff3b5c";
    out.push({ label: str(sh.label) || str(sh.key), color, bbox: [x1, y1, x2, y2] });
  }
  return out;
}

/**
 * Момент, к которому относятся рамки A12 (bboxes Event — «что подсветить на кадре t_peak»):
 * meta.ui.evidence_overlays[].timecode_sec = t_peak; если оверлея нет — таймкод нарушения.
 */
export function overlayPeak(v: Violation, job: JobResult): number {
  const ov = violationOverlay(v, job);
  const t = ov ? optNum(ov.timecode_sec) : undefined;
  return t ?? v.timecode_sec;
}

/** Нормы с русским статусом сверки из оверлея A12 (если есть). */
export function rulesWithStatus(v: Violation, job: JobResult): RuleRef[] {
  const rules = v.rules ?? [];
  const ov = violationOverlay(v, job);
  const norms = ov && Array.isArray(ov.norms) ? ov.norms.filter(isObj) : [];
  if (!norms.length) return rules;
  return rules.map((r) => {
    const n = norms.find((x) => str(x.document) === r.document && (str(x.clause) || null) === (r.clause || null));
    return n && str(n.status_ru) ? { ...r, status_ru: str(n.status_ru) } : r;
  });
}

/** Что по видео установить нельзя для этого типа: от сервера (not_observable) или из каталога A13. */
export function notCheckedFor(v: Violation, job: JobResult): string[] {
  const ov = violationOverlay(v, job);
  const fromServer = ov ? strList(ov.not_observable) : [];
  if (fromServer.length) return fromServer;
  return typeInfo(v.type_id)?.notChecked ?? [];
}
