// Наблюдения правил по последовательностям (ml.seqrules, профиль pose_rules) — расширение контракта:
// GET /ui/v1/jobs/{id}/observations. Только то, что использовали правила: эпизоды с кадрами (bbox человека,
// скелет COCO-17, предмет, сустав), зоны и статусы правил. Нет файла → null (профиль без seqrules).
import type { Violation } from "./types";

export type Box = [number, number, number, number];
export type ObsState = "observed" | "absent" | "unknown" | "occupied";

export interface ObsFrame {
  t: number;
  state: ObsState;
  person?: Box;
  object?: Box;
  object_label?: string;
  object_score?: number;
  joint?: string;
  joint_xy?: [number, number];
  kp?: [number, number, number][];
  in_zone?: boolean | null;
}

export interface ObsEpisode {
  id: string;
  rule: string;
  type_id: string | null;
  status: string;
  reason?: string;
  track_id: number | null;
  role?: string;
  t_start: number;
  t_end: number;
  t_anchor: number;
  duration_s?: number;
  counts: Record<string, number>;
  frames: ObsFrame[];
  safe_checks?: { t: number; state: string; bbox?: Box | null }[];
  exit_box?: Box;
}

export interface ObsZone {
  name: string;
  polygon: [number, number][];
  conf: number;
}

export interface Observations {
  zones: ObsZone[];
  workZone: string | null;
  rules: Record<string, { status: string; reason?: string; episodes?: number }>;
  warnings: string[];
  episodes: ObsEpisode[];
}

const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const box = (x: unknown): Box | undefined =>
  Array.isArray(x) && x.length === 4 && x.every((v) => num(v) !== null) ? (x as Box) : undefined;

function parseFrame(x: unknown): ObsFrame | null {
  if (!x || typeof x !== "object") return null;
  const o = x as Record<string, unknown>;
  const t = num(o.t);
  const st = o.state;
  if (t === null || (st !== "observed" && st !== "absent" && st !== "unknown" && st !== "occupied")) return null;
  const f: ObsFrame = { t, state: st };
  f.person = box(o.person);
  f.object = box(o.object);
  if (typeof o.object_label === "string") f.object_label = o.object_label;
  if (num(o.object_score) !== null) f.object_score = o.object_score as number;
  if (typeof o.joint === "string") f.joint = o.joint;
  if (Array.isArray(o.joint_xy) && o.joint_xy.length === 2) f.joint_xy = o.joint_xy as [number, number];
  if (Array.isArray(o.kp) && o.kp.length >= 17) f.kp = (o.kp as unknown[]).slice(0, 17) as [number, number, number][];
  if (typeof o.in_zone === "boolean" || o.in_zone === null) f.in_zone = o.in_zone as boolean | null;
  return f;
}

/** {job_id, items: [meta, ...episodes]} → Observations; что угодно другое → null (не «пусто, всё чисто»). */
export function parseObservations(raw: unknown): Observations | null {
  const items = raw && typeof raw === "object" ? (raw as { items?: unknown }).items : undefined;
  if (!Array.isArray(items) || !items.length) return null;
  const meta = items[0] as Record<string, unknown>;
  if (!meta || meta.kind !== "meta") return null;
  const zones: ObsZone[] = Array.isArray(meta.zones)
    ? (meta.zones as Record<string, unknown>[])
        .filter((z) => typeof z?.name === "string" && Array.isArray(z.polygon))
        .map((z) => ({ name: z.name as string, polygon: z.polygon as [number, number][], conf: num(z.conf) ?? 0 }))
    : [];
  const rules: Observations["rules"] = {};
  if (meta.rules && typeof meta.rules === "object") {
    for (const [k, v] of Object.entries(meta.rules as Record<string, Record<string, unknown>>)) {
      rules[k] = {
        status: String(v?.status ?? "unknown"),
        reason: typeof v?.reason === "string" ? v.reason : typeof v?.error === "string" ? v.error : undefined,
        episodes: num(v?.episodes) ?? undefined,
      };
    }
  }
  const episodes: ObsEpisode[] = [];
  for (const it of items.slice(1)) {
    const e = it as Record<string, unknown>;
    if (!e || e.kind !== "episode" || typeof e.id !== "string") continue;
    const ts = num(e.t_start);
    const te = num(e.t_end);
    if (ts === null || te === null) continue;
    const frames = (Array.isArray(e.frames) ? e.frames : []).map(parseFrame).filter((f): f is ObsFrame => !!f);
    frames.sort((a, b) => a.t - b.t);
    episodes.push({
      id: e.id,
      rule: String(e.rule ?? ""),
      type_id: typeof e.type_id === "string" ? e.type_id : null,
      status: String(e.status ?? ""),
      reason: typeof e.reason === "string" ? e.reason : undefined,
      track_id: num(e.track_id),
      role: typeof e.role === "string" ? e.role : undefined,
      t_start: ts,
      t_end: te,
      t_anchor: num(e.t_anchor) ?? ts,
      duration_s: num(e.duration_s) ?? undefined,
      counts: (e.counts && typeof e.counts === "object" ? e.counts : {}) as Record<string, number>,
      frames,
      safe_checks: Array.isArray(e.safe_checks)
        ? (e.safe_checks as Record<string, unknown>[]).map((c) => ({ t: num(c.t) ?? 0, state: String(c.state ?? ""), bbox: box(c.bbox) ?? null }))
        : undefined,
      exit_box: box(e.exit_box),
    });
  }
  return {
    zones,
    workZone: typeof meta.work_zone === "string" ? meta.work_zone : null,
    rules,
    warnings: Array.isArray(meta.warnings) ? (meta.warnings as unknown[]).map(String) : [],
    episodes,
  };
}

/** Эпизоды-основания нарушения: тот же тип и пересечение интервалов (fusion не переносит id эпизода). */
export function episodesForViolation(obs: Observations | null, v: Violation | undefined, padS = 0.5): ObsEpisode[] {
  if (!obs || !v) return [];
  const a = (v.t_start ?? v.timecode_sec - 1) - padS;
  const b = (v.t_end ?? v.timecode_sec + 1) + padS;
  return obs.episodes.filter(
    (e) => e.status === "event" && (!v.type_id || e.type_id === v.type_id) && e.t_end >= a && e.t_start <= b,
  );
}

/** Ближайший сохранённый кадр эпизода к t (в пределах tol); вне эпизода — null. */
export function frameAt(ep: ObsEpisode, t: number, tol = 0.3): ObsFrame | null {
  const fs = ep.frames;
  if (!fs.length || t < fs[0].t - tol || t > fs[fs.length - 1].t + tol) return null;
  let lo = 0;
  let hi = fs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (fs[mid].t < t) lo = mid;
    else hi = mid;
  }
  const best = Math.abs(fs[lo].t - t) <= Math.abs(fs[hi].t - t) ? fs[lo] : fs[hi];
  return Math.abs(best.t - t) <= tol ? best : null;
}

export const COCO_EDGES: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

const PALETTE = ["#38bdf8", "#a78bfa", "#34d399", "#f472b6", "#facc15", "#fb923c", "#60a5fa", "#4ade80"];
export const trackColor = (tid: number | null | undefined) => (tid === null || tid === undefined ? "#94a3b8" : PALETTE[Math.abs(tid) % PALETTE.length]);

export const RULE_RU: Record<string, string> = {
  phone: "Телефон у руки или головы",
  safe: "Открытый сейф, рабочая зона пуста",
  storage: "Место хранения оставлено открытым (найдено по рукам)",
  entry: "Вход в рабочую зону",
  hand_cash: "Рука рядом с деньгами",
};

export const STATE_RU: Record<ObsState, string> = {
  observed: "наблюдалось",
  absent: "не видно рядом",
  unknown: "неизвестно",
  occupied: "в зоне есть люди",
};
