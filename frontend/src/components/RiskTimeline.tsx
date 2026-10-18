import { useCallback, useMemo, useRef, useState } from "react";
import type { TimelinePoint, Violation } from "../api/types";
import { formatTimecode, riskColor } from "../lib/format";
import { cx } from "./ui";

interface Props {
  duration: number;
  timeline: TimelinePoint[];
  violations: Violation[];
  /** Пограничные моменты (ниже порога, A12 review_hints) — серые засечки, в вердикт не входят. */
  hints?: { timecode_sec: number; type: string }[];
  currentTime: number;
  onSeek: (t: number) => void;
  /** Клик по маркеру эпизода — тот же выбор, что в навигаторе, карточках и оверлее. */
  onSelect?: (i: number) => void;
}

const VB_W = 1000;

/** Кастомная шкала плеера: тепловая полоса риска + интервалы эпизодов + маркеры (клик → перемотка). */
export default function RiskTimeline({ duration, timeline, violations, hints = [], currentTime, onSeek, onSelect }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ x: number; t: number; r: number | null } | null>(null);
  const dragging = useRef(false);
  const d = duration > 0 ? duration : 1;

  const riskAt = useCallback(
    (t: number): number | null => {
      if (!timeline.length) return null;
      let best = timeline[0];
      for (const p of timeline) {
        if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
        if (p[0] > t) break;
      }
      return best[1];
    },
    [timeline],
  );

  const bars = useMemo(() => {
    const out: { x: number; w: number; r: number }[] = [];
    for (let i = 0; i < timeline.length; i++) {
      const [t, r] = timeline[i];
      const step = i > 0 ? t - timeline[i - 1][0] : 0.5;
      const tNext = i + 1 < timeline.length ? timeline[i + 1][0] : Math.min(d, t + (step > 0 ? step : 0.5));
      const x = (t / d) * VB_W;
      const w = Math.max(0.8, ((tNext - t) / d) * VB_W + 0.4);
      out.push({ x, w, r });
    }
    return out;
  }, [timeline, d]);

  const linePath = useMemo(() => {
    if (timeline.length < 2) return "";
    return timeline
      .map(([t, r], i) => `${i === 0 ? "M" : "L"}${((t / d) * VB_W).toFixed(1)} ${(40 - r * 34).toFixed(1)}`)
      .join(" ");
  }, [timeline, d]);

  const tFromEvent = (clientX: number): { t: number; x: number } | number => {
    const el = ref.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const k = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    return { t: k * d, x: k * rect.width };
  };

  const playX = Math.min(1, Math.max(0, currentTime / d)) * 100;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="select-none">
      <div
        ref={ref}
        className="relative h-14 cursor-pointer touch-none overflow-visible rounded-lg bg-slate-100 ring-1 ring-slate-200 dark:bg-white/[.04] dark:ring-white/10"
        role="slider"
        aria-label="Шкала времени с индексом риска"
        aria-valuemin={0}
        aria-valuemax={Math.round(d)}
        aria-valuenow={Math.round(currentTime)}
        aria-valuetext={formatTimecode(currentTime)}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") onSeek(Math.min(d, currentTime + (e.shiftKey ? 5 : 1)));
          if (e.key === "ArrowLeft") onSeek(Math.max(0, currentTime - (e.shiftKey ? 5 : 1)));
        }}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
          const p = tFromEvent(e.clientX);
          if (typeof p !== "number") onSeek(p.t);
        }}
        onPointerMove={(e) => {
          const p = tFromEvent(e.clientX);
          if (typeof p === "number") return;
          setHover({ x: p.x, t: p.t, r: riskAt(p.t) });
          if (dragging.current) onSeek(p.t);
        }}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => {
          dragging.current = false;
          setHover(null);
        }}
      >
        <svg className="absolute inset-0 h-full w-full overflow-hidden rounded-lg" viewBox={`0 0 ${VB_W} 48`} preserveAspectRatio="none" aria-hidden>
          {bars.map((b, i) => (
            <rect key={i} x={b.x} y={0} width={b.w} height={48} fill={riskColor(b.r, 0.18 + 0.72 * b.r)} />
          ))}
          {violations.map((v, i) => {
            const a = ((v.t_start ?? Math.max(0, v.timecode_sec - 1)) / d) * VB_W;
            const b = ((v.t_end ?? v.timecode_sec + 1) / d) * VB_W;
            return (
              <rect key={`r${i}`} x={a} y={0} width={Math.max(2, b - a)} height={48} fill="none" className="stroke-ks-event" strokeWidth={2} strokeDasharray="5 3" vectorEffect="non-scaling-stroke" />
            );
          })}
          {linePath && <path d={linePath} fill="none" stroke="rgba(15,23,42,.55)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" className="dark:stroke-white/70" />}
        </svg>
        {/* пограничные моменты (ниже порога) */}
        {hints.map((h, i) => (
          <button
            key={`h${i}`}
            className="absolute -bottom-1.5 z-10 -translate-x-1/2"
            style={{ left: `${Math.min(1, h.timecode_sec / d) * 100}%` }}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(h.timecode_sec);
            }}
            title={`Стоит просмотреть (ниже порога): ${h.type} — ${formatTimecode(h.timecode_sec)}`}
            aria-label={`Пограничный момент ${formatTimecode(h.timecode_sec)}`}
          >
            <span className="block h-2.5 w-2.5 rotate-45 border-2 border-slate-500 bg-white dark:border-slate-300 dark:bg-ink-900" />
          </button>
        ))}
        {/* маркеры эпизодов */}
        {violations.map((v, i) => (
          <button
            key={`m${i}`}
            className="group absolute -top-3 z-10 -translate-x-1/2"
            style={{ left: `${Math.min(1, v.timecode_sec / d) * 100}%` }}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onSelect?.(i);
              onSeek(v.timecode_sec);
            }}
            title={`Эпизод ${i + 1}: ${v.type} — ${formatTimecode(v.timecode_sec)}`}
            aria-label={`Перейти к эпизоду ${i + 1}, ${formatTimecode(v.timecode_sec)}`}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-ks-event text-[11px] font-bold text-ks-event-contrast shadow-lift ring-2 ring-white transition group-hover:scale-110 dark:ring-ink-900">
              {i + 1}
            </span>
            <span className="mx-auto block h-[46px] w-0.5 bg-ks-event/80" />
          </button>
        ))}
        {/* playhead */}
        <div className="pointer-events-none absolute inset-y-0 z-20 w-0.5 bg-ink-900 dark:bg-white" style={{ left: `${playX}%` }}>
          <div className="absolute -left-[5px] -top-1 h-3 w-3 rounded-full bg-ink-900 ring-2 ring-white dark:bg-white dark:ring-ink-900" />
        </div>
        {hover && (
          <div
            className="pointer-events-none absolute -top-9 z-30 -translate-x-1/2 whitespace-nowrap rounded-md bg-ink-900 px-2 py-1 text-[11px] font-medium text-white shadow-lift tnum dark:bg-white dark:text-ink-900"
            style={{ left: hover.x }}
          >
            {formatTimecode(hover.t)}
            {hover.r !== null && <span className="opacity-70"> · индекс риска {hover.r.toFixed(2).replace(".", ",")}</span>}
          </div>
        )}
      </div>
      <div className="relative mt-1.5 h-4 text-[10px] text-slate-500 tnum">
        {ticks.map((k) => (
          <span
            key={k}
            className={cx("absolute", k === 0 ? "left-0" : k === 1 ? "right-0" : "-translate-x-1/2")}
            style={k > 0 && k < 1 ? { left: `${k * 100}%` } : undefined}
          >
            {formatTimecode(k * d).replace(/\.\d$/, "")}
          </span>
        ))}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-8 rounded-sm" style={{ background: `linear-gradient(90deg, ${riskColor(0, 0.5)}, ${riskColor(0.5, 0.8)}, ${riskColor(1, 0.95)})` }} />
          индекс риска по кадрам (не вероятность)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm border-2 border-dashed border-[rgb(var(--ks-event))]" /> интервал эпизода
        </span>
        {hints.length > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="block h-2 w-2 rotate-45 border-2 border-slate-500" /> стоит просмотреть (ниже порога)
          </span>
        )}
        {!timeline.length && <span>сервер не передал кривую риска</span>}
      </div>
    </div>
  );
}
