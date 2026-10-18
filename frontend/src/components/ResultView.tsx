import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  Ban,
  Download,
  Eye,
  FileJson,
  FileText,
  ListChecks,
  Printer,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { ApiError, localVideo } from "../api/client";
import { inferDuration, reviewHints, toJuryJson } from "../api/normalize";
import { episodesForViolation, type Observations } from "../api/observations";
import type { JobResult } from "../api/types";
import { useApp } from "../context";
import { formatDateTime, formatDuration, formatTimecode } from "../lib/format";
import { analyzeCompleteness } from "../lib/completeness";
import { verdictView } from "../lib/verdict";
import CompletenessPanel from "./CompletenessPanel";
import EpisodeNavigator from "./EpisodeNavigator";
import RetryUpload from "./RetryUpload";
import { uiConfig } from "../lib/uiConfig";
import CopilotPanel from "./CopilotPanel";
import ObservationOverlay, { DEFAULT_LAYERS, type OverlayLayers } from "./ObservationOverlay";
import RuleObservations, { OverlayToggles } from "./RuleObservations";
import VideoPlayer, { type PlayerHandle } from "./VideoPlayer";
import ViolationCard from "./ViolationCard";
import { cx, LocalBadge } from "./ui";

function download(name: string, data: unknown) {
  saveBlob(name, new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
}

function saveBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export const NOT_CHECKED = [
  "корректность проводок и записей в учётных системах",
  "суммы, номиналы и подлинность банкнот",
  "личность людей — лица не распознаются, роли определяются по зоне, поведению и предметам",
  "наличие допуска или полномочий у человека",
  "умысел и причины действий",
  "события вне поля зрения камеры и звук",
];

export default function ResultView({
  job,
  raw,
  fileName,
  replay,
}: {
  job: JobResult;
  raw: unknown;
  fileName?: string;
  /** Повторный просмотр сохранённого отчёта: без запросов к серверу, с явной пометкой. */
  replay?: boolean;
}) {
  const { backend, demo, toast } = useApp();
  const player = useRef<PlayerHandle>(null);
  const playerBox = useRef<HTMLDivElement>(null);
  const [t, setT] = useState(0);
  const completeness = useMemo(() => analyzeCompleteness(job), [job]);
  const baseView = verdictView(job, completeness.state, completeness.mock);
  const view = replay ? { ...baseView, flags: ["ПОВТОРНЫЙ ПРОСМОТР: сохранённый отчёт", ...baseView.flags] } : baseView;
  const duration = inferDuration(job);
  const candidates = job.candidates ?? [];
  // Эпизоды для просмотра: нарушения; у упавшего анализа без нарушений — кандидаты A01 (не вошли в вердикт).
  const episodes = job.violations.length ? job.violations : job.status === "failed" ? candidates : [];
  const episodesAreCandidates = job.violations.length === 0 && episodes.length > 0;
  // У завершённого анализа кандидаты (ниже порога) — в «Стоит просмотреть», вместе с review_hints A12.
  const hints = [
    ...reviewHints(job),
    ...(job.status === "done" && job.violations.length === 0
      ? candidates.map((c) => ({ type: c.type, type_id: c.type_id, timecode_sec: c.timecode_sec, reasons: ["не вошло в вердикт"], facts: [] as string[] }))
      : []),
  ].sort((a, b) => a.timecode_sec - b.timecode_sec);
  const source = useMemo(() => {
    const server = replay ? null : backend.videoSource(job.job_id);
    const local = localVideo(job.job_id);
    // Только что загруженный файл играет сразу локально; если браузер его не декодирует (напр. HEVC) —
    // пробуем серверный маршрут (может отдать перекодированный preview.mp4), затем запасные пути.
    if (local) return { url: local, fallbacks: server ? [async () => server.url, ...(server.fallbacks ?? [])] : [] };
    return server;
  }, [backend, job.job_id, replay]);
  const onTime = useCallback((x: number) => setT(x), []);
  const [current, setCurrent] = useState(0);
  const [reviewed, setReviewed] = useState<Set<number>>(() => loadReviewed(job.job_id));
  useEffect(() => saveReviewed(job.job_id, reviewed), [job.job_id, reviewed]);
  // Сдвиг PTS — только по флагу (гипотеза до замера L2): meta.video.t0_pts_sec.
  const timeOffset = uiConfig().ptsOffsetMode === "meta_t0" ? readT0(job.meta) : 0;
  const capture = useCallback((x: number) => player.current?.captureAt(x) ?? Promise.resolve(null), []);

  // Наблюдения правил (профиль pose_rules) — оверлей и объяснение; нет на сервере → null, UI как раньше.
  const [obs, setObs] = useState<Observations | null>(null);
  useEffect(() => {
    let alive = true;
    setObs(null);
    if (!replay && backend.observations && job.status === "done") {
      backend.observations(job.job_id).then((o) => alive && setObs(o)).catch(() => alive && setObs(null));
    }
    return () => {
      alive = false;
    };
  }, [backend, job.job_id, job.status, replay]);
  const [layers, setLayers] = useState<OverlayLayers>(loadLayers);
  useEffect(() => saveLayers(layers), [layers]);
  const selIdx = Math.min(current, Math.max(0, episodes.length - 1));
  const selEps = useMemo(() => episodesForViolation(obs, episodes[selIdx]), [obs, episodes, selIdx]);
  const selIds = useMemo(() => new Set(selEps.map((e) => e.id)), [selEps]);
  const shownEps = useMemo(
    () => (layers.all ? (obs?.episodes ?? []).filter((e) => e.status === "event") : selEps),
    [layers.all, obs, selEps],
  );
  const overlay = useMemo(
    () =>
      obs
        ? (tt: number, box: { w: number; h: number }) => (
            <ObservationOverlay obs={obs} episodes={shownEps} selectedIds={selIds} t={tt} width={box.w} height={box.h} layers={layers} />
          )
        : undefined,
    [obs, shownEps, selIds, layers],
  );

  const seek = (time: number) => {
    player.current?.seek(time);
    playerBox.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const goEpisode = (i: number) => {
    if (i < 0 || i >= episodes.length) return;
    setCurrent(i);
    seek(episodes[i].timecode_sec);
  };
  const playEpisode = (i: number) => {
    const v = episodes[i];
    if (!v) return;
    setCurrent(i);
    const from = Math.max(0, (v.t_start ?? v.timecode_sec) - uiConfig().episodePrerollS);
    const to = (v.t_end ?? v.timecode_sec + 2) + uiConfig().episodePostrollS;
    player.current?.playRange(from, to);
    playerBox.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const toggleReviewed = (i: number) =>
    setReviewed((r) => {
      const n = new Set(r);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return n;
    });

  const timings = (job.meta.timings ?? {}) as Record<string, number>;
  const total = typeof timings.total === "number" ? timings.total : undefined;
  const warnings = Array.isArray(job.meta.warnings) ? (job.meta.warnings as unknown[]).map(String) : [];
  const rawModels = job.meta.models;
  const models: unknown[] = Array.isArray(rawModels)
    ? rawModels
    : rawModels && typeof rawModels === "object"
      ? Object.entries(rawModels as Record<string, unknown>).map(([k, v]) => {
          const o = (v ?? {}) as { origin?: unknown; version?: unknown };
          return `${k}: ${[o.origin, o.version].filter(Boolean).join(" ")}`;
        })
      : [];
  const maxRisk = job.timeline.reduce((m, [, r]) => Math.max(m, r), 0);

  const Icon = view.tone === "danger" ? AlertOctagon : view.tone === "success" ? ShieldCheck : AlertTriangle;
  const bannerTone = {
    danger: "from-rose-600 to-brand-700 text-white",
    success: "from-emerald-600 to-teal-700 text-white",
    warning: "from-amber-400 to-amber-500 text-amber-950",
    neutral: "from-slate-600 to-slate-700 text-white",
  }[view.tone];

  const name = fileName ?? (typeof job.meta.file_name === "string" ? job.meta.file_name : undefined);

  return (
    <div className="animate-fade-in space-y-6">
      {/* печатная шапка отчёта */}
      <div className="print-only mb-4 border-b border-slate-300 pb-3">
        <div className="text-xl font-bold">Kassa Sentinel — отчёт о проверке видеозаписи</div>
        <div className="text-sm">
          Файл: {name ?? "—"} · Задача: {job.job_id} · Сформирован: {formatDateTime(new Date())}
          {demo && " · СИНТЕТИЧЕСКОЕ ДЕМО"}
        </div>
      </div>

      <section className={cx("relative overflow-hidden rounded-3xl bg-gradient-to-br p-6 shadow-lift sm:p-8", bannerTone)}>
        <div className="bg-grid pointer-events-none absolute inset-0 opacity-30" />
        <div className="relative flex flex-wrap items-start gap-5">
          <span className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-white/15 ring-1 ring-white/25">
            <Icon className="h-8 w-8" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-semibold uppercase tracking-[.14em] opacity-75">
              Вердикт системы{demo ? " · синтетическое демо" : ""}
              {replay ? " · сохранённый отчёт" : ""}
            </div>
            <h1 className="mt-1 text-2xl font-extrabold leading-tight sm:text-[32px]">{view.title}</h1>
            <p className="mt-2 max-w-3xl text-sm opacity-90 sm:text-[15px]">{view.subtitle}</p>
            {view.flags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {view.flags.map((f) => (
                  <span key={f} className="rounded-md bg-black/25 px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider ring-1 ring-white/30">
                    {f}
                  </span>
                ))}
              </div>
            )}
            <div className="mt-4 flex flex-wrap gap-2 text-xs font-semibold">
              {name && <span className="rounded-full bg-white/15 px-3 py-1">{name}</span>}
              {duration > 0 && <span className="rounded-full bg-white/15 px-3 py-1 tnum">длительность {formatTimecode(duration)}</span>}
              <span className="rounded-full bg-white/15 px-3 py-1 tnum">эпизодов: {job.violations.length}</span>
              {total !== undefined && <span className="rounded-full bg-white/15 px-3 py-1 tnum">анализ {formatDuration(total)}</span>}
              {job.timeline.length > 0 && (
                <span className="rounded-full bg-white/15 px-3 py-1 tnum">макс. индекс риска {maxRisk.toFixed(2).replace(".", ",")}</span>
              )}
            </div>
          </div>
        </div>
        <div className="no-print relative mt-6 flex flex-wrap gap-2">
          <button className="btn bg-white/15 text-current ring-1 ring-white/25 hover:bg-white/25" onClick={() => download(`kassa-sentinel_${job.job_id}_jury.json`, toJuryJson(job))} title="Только обязательные поля ТЗ">
            <FileJson className="h-4 w-4" /> JSON (схема ТЗ)
          </button>
          <button className="btn bg-white/15 text-current ring-1 ring-white/25 hover:bg-white/25" onClick={() => download(`kassa-sentinel_${job.job_id}_full.json`, raw)} title="Полный ответ: нормы, timeline, meta">
            <Download className="h-4 w-4" /> Полный отчёт JSON
          </button>
          {!replay && backend.report && job.status === "done" && (
            <button
              className="btn bg-white/15 text-current ring-1 ring-white/25 hover:bg-white/25"
              title="Текстовое заключение для проверяющего (Markdown, формирует сервер)"
              onClick={async () => {
                try {
                  const md = await backend.report!(job.job_id);
                  saveBlob(`kassa-sentinel_${job.job_id}_report.md`, new Blob([md], { type: "text/markdown;charset=utf-8" }));
                } catch (e) {
                  toast(e instanceof ApiError ? e.message : "Заключение недоступно", "error");
                }
              }}
            >
              <FileText className="h-4 w-4" /> Заключение .md
            </button>
          )}
          <button className="btn bg-white/15 text-current ring-1 ring-white/25 hover:bg-white/25" onClick={() => window.print()}>
            <Printer className="h-4 w-4" /> Печать / PDF
          </button>
          <a className="btn bg-white text-ink-900 hover:bg-slate-100" href={demo ? "?demo=1#/" : "#/"}>
            <Upload className="h-4 w-4" /> Новая проверка
          </a>
        </div>
      </section>

      {job.status === "failed" && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-200" role="alert">
          <b>Анализ не завершён{job.error ? `: ${job.error}` : ""}.</b>{" "}
          {episodes.length
            ? "Ниже — только то, что система успела найти; в вердикт это не вошло, остальная часть записи могла остаться непроверенной."
            : "Вывод об отсутствии нарушений сделать нельзя — нужен ручной просмотр."}
          <div className="mt-2">
            {!replay && <RetryUpload jobId={job.job_id} className="btn-ghost px-3 py-1.5 text-xs" />}
          </div>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-200">
          <b>Предупреждения сервера:</b> {warnings.join(" · ")}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0 space-y-6">
          <section ref={playerBox} className="card no-print scroll-mt-32 p-4 sm:p-5">
            <VideoPlayer
              ref={player}
              source={source}
              synthetic={demo}
              duration={duration}
              timeline={job.timeline}
              violations={episodes}
              hints={uiConfig().showReviewHints ? hints : []}
              onTime={onTime}
              timeOffset={timeOffset}
              overlay={overlay}
              onSelect={(i) => setCurrent(i)}
            />
            {obs && <OverlayToggles layers={layers} onChange={setLayers} />}
            <EpisodeNavigator
              violations={episodes}
              current={Math.min(current, Math.max(0, episodes.length - 1))}
              reviewed={reviewed}
              onGo={goEpisode}
              onPlay={playEpisode}
              onToggleReviewed={toggleReviewed}
            />
          </section>

          {episodes.length > 0 ? (
            <section className="space-y-4">
              <h2 className="flex items-center gap-2 text-lg font-bold">
                <ListChecks className="h-5 w-5 text-ks-event" />
                {episodesAreCandidates
                  ? "Кандидаты — не подтверждены, в вердикт не вошли"
                  : job.status === "failed"
                    ? "Найдено до сбоя анализа"
                    : "Эпизоды с признаками возможных нарушений"}
              </h2>
              {episodes.map((v, i) => (
                <ViolationCard
                  candidate={episodesAreCandidates}
                  key={`${v.type_id}-${v.timecode_sec}-${i}`}
                  v={v}
                  index={i}
                  job={job}
                  active={t >= (v.t_start ?? v.timecode_sec - 1) && t <= (v.t_end ?? v.timecode_sec + 1)}
                  onSeek={(x) => {
                    setCurrent(i);
                    seek(x);
                  }}
                  capture={capture}
                />
              ))}
            </section>
          ) : (
            <section className="card p-6 text-center">
              <ShieldCheck
                className={cx("mx-auto h-10 w-10", view.tone === "success" ? "text-emerald-500" : view.tone === "warning" ? "text-amber-500" : "text-slate-400")}
              />
              <h2 className="mt-3 text-lg font-bold">Эпизодов с признаками нарушений не найдено</h2>
              <p className="mx-auto mt-1 max-w-lg text-sm text-slate-500">
                {view.tone === "success"
                  ? "Система не выявила видимых признаков нарушений. Ограничения — в блоке «Что система не проверяет»."
                  : completeness.state === "partial"
                    ? "Анализ неполный — вывод об отсутствии нарушений сделать нельзя, просмотрите запись вручную."
                    : "Полнота анализа не подтверждена — отсутствие найденных эпизодов не означает отсутствия нарушений."}
              </p>
            </section>
          )}
        </div>

        <aside className="space-y-6 lg:sticky lg:top-24 lg:self-start">
          {episodes.length > 0 && (
            <section className="card no-print p-4">
              <div className="label mb-2">Навигация по эпизодам</div>
              <ul className="space-y-1">
                {episodes.map((v, i) => (
                  <li key={i}>
                    <button
                      className={cx(
                        "flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-slate-50 dark:hover:bg-white/5",
                        i === selIdx && "bg-slate-50 ring-1 ring-ks-event/40 dark:bg-white/5",
                      )}
                      aria-current={i === selIdx ? "true" : undefined}
                      onClick={() => {
                        setCurrent(i);
                        seek(v.timecode_sec);
                      }}
                    >
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ks-event text-[11px] font-bold text-ks-event-contrast">{i + 1}</span>
                      <span className="min-w-0 flex-1 truncate text-sm font-medium">{v.type}</span>
                      <span className="font-mono text-xs text-slate-500 tnum">{formatTimecode(v.timecode_sec)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {uiConfig().showReviewHints && hints.length > 0 && (
            <section className="card p-4">
              <div className="label mb-1 flex items-center gap-1.5">
                <Eye className="h-3.5 w-3.5" /> Стоит просмотреть
              </div>
              <p className="mb-2 text-xs text-slate-500">
                Пограничные моменты ниже порога уверенности. В вердикт не входят — система не считает их нарушениями.
              </p>
              <ul className="space-y-1">
                {hints.slice(0, 8).map((h, i) => (
                  <li key={i}>
                    <button
                      className="no-print flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left hover:bg-slate-50 dark:hover:bg-white/5"
                      onClick={() => seek(h.timecode_sec)}
                      title={h.facts.join("; ") || undefined}
                    >
                      <span className="mt-1 block h-2.5 w-2.5 shrink-0 rotate-45 border-2 border-slate-500 dark:border-slate-300" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{h.type}</span>
                        {h.reasons.length > 0 && <span className="block truncate text-xs text-slate-500">{h.reasons.join(", ")}</span>}
                      </span>
                      <span className="font-mono text-xs text-slate-500 tnum">{formatTimecode(h.timecode_sec)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {obs && <RuleObservations obs={obs} selected={selEps} violation={episodes[selIdx]} onSeek={seek} />}

          {!replay && job.status === "done" && <CopilotPanel jobId={job.job_id} />}

          <CompletenessPanel report={completeness} />

          <section className="card p-4">
            <div className="label mb-2 flex items-center gap-1.5">
              <Ban className="h-3.5 w-3.5" /> Что система не проверяет
            </div>
            <ul className="space-y-1.5 text-sm text-slate-600 dark:text-slate-300">
              {NOT_CHECKED.map((x) => (
                <li key={x} className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
                  {x}
                </li>
              ))}
            </ul>
            <p className="mt-3 text-xs leading-relaxed text-slate-500">
              Система описывает только видимое на записи. Результат — подсказка для проверяющего, не юридическое заключение.
            </p>
          </section>

          <details className="card no-print group p-4">
            <summary className="label cursor-pointer list-none">Технические детали ▾</summary>
            <div className="mt-3 space-y-3 text-xs text-slate-600 dark:text-slate-300">
              <div className="flex items-center justify-between gap-2">
                <span>Версия</span>
                <span className="font-mono">{String(job.meta.version ?? "—")}</span>
              </div>
              {Object.keys(timings).length > 0 && (
                <div>
                  <div className="mb-1 font-semibold">Время стадий, с</div>
                  {Object.entries(timings)
                    .filter(([k]) => k !== "total")
                    .map(([k, v]) => (
                      <div key={k} className="flex justify-between font-mono tnum">
                        <span>{k}</span>
                        <span>{typeof v === "number" ? v.toFixed(1) : String(v)}</span>
                      </div>
                    ))}
                </div>
              )}
              {models.length > 0 && (
                <div>
                  <div className="mb-1 font-semibold">Модели</div>
                  {models.map((m, i) => (
                    <div key={i} className="font-mono">
                      {typeof m === "string" ? m : `${(m as { name?: string }).name ?? "?"} · ${(m as { license?: string }).license ?? ""}`}
                    </div>
                  ))}
                </div>
              )}
              <LocalBadge />
            </div>
          </details>
        </aside>
      </div>
    </div>
  );
}

function readT0(meta: Record<string, unknown>): number {
  const pick = (o: unknown) => (o && typeof o === "object" ? (o as Record<string, unknown>).t0_pts_sec : undefined);
  for (const v of [pick(meta.video), meta.t0_pts_sec, pick(meta.ingest)]) {
    if (typeof v === "number" && Number.isFinite(v) && v >= 0 && v < 60) return v;
  }
  return 0;
}

const LAYERS_KEY = "ks.overlay.layers";
function loadLayers(): OverlayLayers {
  try {
    const v = JSON.parse(localStorage.getItem(LAYERS_KEY) ?? "null") as Partial<OverlayLayers> | null;
    return { ...DEFAULT_LAYERS, ...(v && typeof v === "object" ? v : {}) };
  } catch {
    return { ...DEFAULT_LAYERS };
  }
}
function saveLayers(l: OverlayLayers) {
  try {
    localStorage.setItem(LAYERS_KEY, JSON.stringify(l));
  } catch {
    /* noop */
  }
}

const REV_KEY = (id: string) => `ks.reviewed.${id}`;
function loadReviewed(jobId: string): Set<number> {
  try {
    const v = JSON.parse(localStorage.getItem(REV_KEY(jobId)) ?? "[]") as unknown;
    return new Set(Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : []);
  } catch {
    return new Set();
  }
}
function saveReviewed(jobId: string, s: Set<number>) {
  try {
    if (s.size) localStorage.setItem(REV_KEY(jobId), JSON.stringify([...s]));
    else localStorage.removeItem(REV_KEY(jobId));
  } catch {
    /* noop */
  }
}
