import { CheckSquare, ChevronLeft, ChevronRight, PlayCircle, Square } from "lucide-react";
import type { Violation } from "../api/types";
import { formatTimecode } from "../lib/format";
import { cx } from "./ui";
import s from "./a15.module.css";

/** До стольких эпизодов шкала просмотра рисуется по сегменту на эпизод; дальше — сплошной полосой. */
const SEGMENTS_MAX = 48;

/** Последовательный просмотр эпизодов: ‹ › , «Смотреть эпизод», отметка «просмотрено», шкала просмотра. */
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
  return (
    <div className={cx(s.scope, s.nav)} role="group" aria-label="Просмотр эпизодов">
      <div className={s.navHead}>
        <div className={cx(s.eyebrow, "tnum")}>
          Эпизод {current + 1} из {n} · просмотрено {reviewed.size} из {n}
        </div>
      </div>
      {/* Шкала — только наглядность (кнопки ‹ › и список эпизодов дублируют её для клавиатуры и чтения с экрана). */}
      {n <= SEGMENTS_MAX ? (
        <div className={s.navTrack} style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }} aria-hidden>
          {violations.map((_, i) => (
            <span key={i} className={s.navSeg} data-current={i === current} data-reviewed={reviewed.has(i)} />
          ))}
        </div>
      ) : (
        <div className={s.navBar} aria-hidden>
          <span style={{ width: `${(reviewed.size / n) * 100}%` }} />
        </div>
      )}
      <div className={s.navMain}>
        <button
          type="button"
          className={s.navArrow}
          onClick={() => onGo(current - 1)}
          disabled={current <= 0}
          aria-label="Предыдущий эпизод"
          title="Предыдущий эпизод (P)"
        >
          <ChevronLeft className="h-4 w-4" aria-hidden />
        </button>
        <button type="button" className={s.navEpisode} onClick={() => onGo(current)} title="Перейти к эпизоду в видео">
          <span className={s.navNum}>{current + 1}</span>
          <span className={s.navText}>
            <span className={s.navType}>{v.type}</span>
            <span className={s.navTime}>{formatTimecode(v.timecode_sec)}</span>
          </span>
        </button>
        <button
          type="button"
          className={s.navArrow}
          onClick={() => onGo(current + 1)}
          disabled={current >= n - 1}
          aria-label="Следующий эпизод"
          title="Следующий эпизод (N)"
        >
          <ChevronRight className="h-4 w-4" aria-hidden />
        </button>
      </div>
      <div className={s.navActions}>
        <button type="button" className="btn-primary px-3 py-1.5 text-xs" onClick={() => onPlay(current)} title="Проиграть эпизод с небольшим запасом до и после">
          <PlayCircle className="h-4 w-4" aria-hidden /> Смотреть эпизод
        </button>
        <button type="button" className={cx("btn-ghost px-3 py-1.5 text-xs", s.reviewBtn)} onClick={() => onToggleReviewed(current)} aria-pressed={isRev}>
          {isRev ? <CheckSquare className="h-4 w-4" aria-hidden /> : <Square className="h-4 w-4" aria-hidden />} {isRev ? "Просмотрено" : "Отметить просмотренным"}
        </button>
      </div>
    </div>
  );
}
