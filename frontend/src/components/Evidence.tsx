import { useEffect, useState } from "react";
import { ArrowRight, Crosshair, ImageOff } from "lucide-react";
import type { EvidenceFrame, Violation } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { formatTimecode } from "../lib/format";
import { uiConfig } from "../lib/uiConfig";
import { cx, Spinner } from "./ui";
import s from "./a15.module.css";

/** Кадр с необязательной подсветкой рамками (A12 shapes; координаты нормированы к кадру). */
export function EvidenceImage({
  src,
  boxes,
  alt,
  className,
  imgClassName,
}: {
  src: string;
  boxes: HighlightBox[];
  alt: string;
  className?: string;
  imgClassName?: string;
}) {
  const labelled = boxes.slice(0, 3).filter((b) => b.label);
  return (
    <div className={cx("relative", className ?? "block")}>
      <img src={src} alt={alt} draggable={false} className={cx("block select-none", imgClassName ?? "h-auto w-full")} />
      {boxes.length > 0 && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
          {boxes.map((b, i) => {
            const r = { x: b.bbox[0], y: b.bbox[1], width: b.bbox[2] - b.bbox[0], height: b.bbox[3] - b.bbox[1] };
            return (
              <g key={i}>
                {/* тёмная подложка: рамка видна и на светлом участке кадра */}
                <rect {...r} fill="none" stroke="#000" strokeOpacity={0.5} strokeWidth={5} vectorEffect="non-scaling-stroke" />
                <rect {...r} fill="none" stroke={b.color} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              </g>
            );
          })}
        </svg>
      )}
      {labelled.length > 0 && (
        <div className={s.boxLabels}>
          {labelled.map((b, i) => (
            <span key={i} className={s.boxLabel} style={{ color: b.color }}>
              <span className={s.boxLabelText}>{b.label}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Рамки A12 заданы для кадра t_peak — рисуем их только на нём (|ts − t_peak| ≤ maxDt), не на соседних кадрах. */
export function boxesForFrame(frame: EvidenceFrame, peakT: number, boxes: HighlightBox[], maxDt = uiConfig().highlightMaxDtS): HighlightBox[] {
  return !frame.missing && Math.abs(frame.timestamp_sec - peakT) <= maxDt ? boxes : [];
}

/** «2,0 с» — промежуток между показанными кадрами (из их меток времени). */
function formatGap(sec: number): string {
  return `${sec.toFixed(1).replace(".", ",")} с`;
}

/**
 * «До / после» для эпизодов-переходов («ушла и оставила», сейф оставлен открытым, сумка оставлена).
 * «До» — кадр сервера до начала эпизода, иначе кадр из видео в браузере (t_start − beforeS); если видео
 * не воспроизводится — кнопка перехода к этому моменту. «После» — кадр сервера в момент/после начала.
 */
export function BeforeAfter({
  v,
  peakT,
  boxes,
  highlight,
  capture,
  onSeek,
  onOpen,
  beforeS = 2,
}: {
  v: Violation;
  /** t_peak, к которому относятся рамки. */
  peakT: number;
  boxes: HighlightBox[];
  highlight: boolean;
  capture?: (t: number) => Promise<string | null>;
  onSeek: (t: number) => void;
  onOpen: (index: number) => void;
  beforeS?: number;
}) {
  const start = v.t_start ?? v.timecode_sec;
  const beforeT = Math.max(0, start - beforeS);
  // Только кадры с изображением; индексы — в этом же списке (его же получает лайтбокс).
  const frames = v.evidence_frames.filter((f) => !f.missing);
  const serverBeforeIdx = frames.findIndex((f) => f.timestamp_sec <= start - 0.3);
  // «После» — только кадр в момент/после начала эпизода; более ранний кадр НЕ подставляется.
  const after = frames
    .map((f, i) => ({ f, i }))
    .filter(({ f }) => f.timestamp_sec >= start - 0.05)
    .sort((a, b) => Math.abs(a.f.timestamp_sec - v.timecode_sec) - Math.abs(b.f.timestamp_sec - v.timecode_sec))[0];
  const [captured, setCaptured] = useState<string | null | "loading" | "error">(null);

  useEffect(() => {
    let alive = true;
    if (serverBeforeIdx >= 0 || !capture) return;
    setCaptured("loading");
    void capture(beforeT).then((r) => alive && setCaptured(r ?? "error"));
    return () => {
      alive = false;
    };
  }, [serverBeforeIdx, capture, beforeT]);

  const capturedSrc = captured && captured !== "loading" && captured !== "error" ? captured : null;
  const beforeTs = serverBeforeIdx >= 0 ? frames[serverBeforeIdx].timestamp_sec : beforeT;
  const gap = (serverBeforeIdx >= 0 || capturedSrc) && after ? after.f.timestamp_sec - beforeTs : null;

  return (
    <div className={cx(s.scope, s.ba)}>
      <div className={cx(s.eyebrow, s.baHead)}>До и после начала эпизода</div>
      <div className={s.baGrid}>
        <figure className={s.baCell}>
          {serverBeforeIdx >= 0 ? (
            <button type="button" className={s.baTile} onClick={() => onOpen(serverBeforeIdx)} aria-label={`Открыть кадр «до» (${formatTimecode(beforeTs)})`}>
              <EvidenceImage src={imageSrc(frames[serverBeforeIdx].image_base64)} boxes={[]} alt="Кадр до начала эпизода" className="w-full" />
              <span className={s.baTime}>{formatTimecode(beforeTs)}</span>
            </button>
          ) : capturedSrc ? (
            <button type="button" className={s.baTile} onClick={() => onSeek(beforeT)} aria-label={`Перейти к моменту «до» в видео (${formatTimecode(beforeT)})`}>
              <EvidenceImage src={capturedSrc} boxes={[]} alt="Кадр из видео до начала эпизода" className="w-full" />
              <span className={s.baTime}>{formatTimecode(beforeT)}</span>
            </button>
          ) : (
            <div className={cx(s.baTile, s.baEmpty)}>
              {captured === "loading" ? <Spinner className="h-5 w-5" /> : <ImageOff className="h-5 w-5" aria-hidden />}
              <span>{captured === "loading" ? "берём кадр из видео…" : "кадра «до» нет"}</span>
              <button type="button" className={s.baSeek} onClick={() => onSeek(beforeT)} aria-label={`Перейти к ${formatTimecode(beforeT)} в видео`}>
                <Crosshair className="h-3 w-3" aria-hidden /> к {formatTimecode(beforeT)}
              </button>
            </div>
          )}
          <figcaption className={s.baCap}>
            <span className={s.baWhen}>до</span>
            {serverBeforeIdx < 0 && capturedSrc && <span className={s.baWarn}>кадр из видео, без размытия</span>}
          </figcaption>
        </figure>
        <div className={s.baLink} aria-hidden>
          <ArrowRight className="h-4 w-4" />
          {gap !== null && gap > 0 && <span>+{formatGap(gap)}</span>}
        </div>
        <figure className={s.baCell}>
          {after ? (
            <button
              type="button"
              className={s.baTile}
              onClick={() => onOpen(after.i)}
              aria-label={`Открыть кадр «после» (${formatTimecode(after.f.timestamp_sec)})`}
            >
              <EvidenceImage
                src={imageSrc(after.f.image_base64)}
                boxes={highlight ? boxesForFrame(after.f, peakT, boxes) : []}
                alt="Кадр после начала эпизода"
                className="w-full"
              />
              <span className={s.baTime}>{formatTimecode(after.f.timestamp_sec)}</span>
            </button>
          ) : (
            <div className={cx(s.baTile, s.baEmpty)}>
              <ImageOff className="h-5 w-5" aria-hidden />
              <span>кадра «после» нет</span>
            </div>
          )}
          <figcaption className={s.baCap}>
            <span className={s.baWhen}>после</span>
          </figcaption>
        </figure>
      </div>
    </div>
  );
}
