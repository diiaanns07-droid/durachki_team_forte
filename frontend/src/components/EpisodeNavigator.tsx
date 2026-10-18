import { CheckSquare, ChevronLeft, ChevronRight, PlayCircle, Square } from "lucide-react";
import type { Violation } from "../api/types";
import { formatTimecode } from "../lib/format";
import { cx } from "./ui";

/** Последовательный просмотр эпизодов: ‹ › , «Смотреть эпизод», отметка «просмотрено». */
export default function EpisodeNavigator({
  violations,
  current,
  reviewed,
  onGo,
  onPlay,
  onToggleReviewed,
}: {
  violations: Violation[];
  current: number;
  reviewed: Set<number>;
  onGo: (i: number) => void;
  onPlay: (i: number) => void;
  onToggleReviewed: (i: number) => void;
}) {
  const n = violations.length;
  if (n === 0) return null;
  const v = violations[current];
  const isRev = reviewed.has(current);
  const btn = "rounded-lg border border-slate-200 p-2 text-slate-600 hover:bg-slate-50 disabled:opacity-40 dark:border-white/10 dark:text-slate-300 dark:hover:bg-white/5";
  return (
    <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50/70 p-3 dark:border-white/10 dark:bg-white/[.03]" aria-label="Просмотр эпизодов">
      <div className="flex flex-wrap items-center gap-2">
        <button className={btn} onClick={() => onGo(current - 1)} disabled={current <= 0} aria-label="Предыдущий эпизод" title="Предыдущий эпизод (P)">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 tnum">
            Эпизод {current + 1} из {n} · просмотрено {reviewed.size} из {n}
          </div>
          <button className="block max-w-full truncate text-left text-sm font-semibold hover:underline" onClick={() => onGo(current)}>
            {v.type} · <span className="font-mono tnum">{formatTimecode(v.timecode_sec)}</span>
          </button>
        </div>
        <button className={btn} onClick={() => onGo(current + 1)} disabled={current >= n - 1} aria-label="Следующий эпизод" title="Следующий эпизод (N)">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <button className="btn-brand px-3 py-1.5 text-xs" onClick={() => onPlay(current)} title="Проиграть эпизод с небольшим запасом до и после">
          <PlayCircle className="h-4 w-4" /> Смотреть эпизод
        </button>
        <button
          className={cx("btn-ghost px-3 py-1.5 text-xs", isRev && "border-emerald-300 text-emerald-700 dark:border-emerald-400/30 dark:text-emerald-300")}
          onClick={() => onToggleReviewed(current)}
          aria-pressed={isRev}
        >
          {isRev ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />} {isRev ? "Просмотрено" : "Отметить просмотренным"}
        </button>
      </div>
    </div>
  );
}
