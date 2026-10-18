import { useEffect, useState, type CSSProperties } from "react";
import { ArrowRight, Crosshair, ImageOff } from "lucide-react";
import type { EvidenceFrame, Violation } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { formatTimecode } from "../lib/format";
import { uiConfig } from "../lib/uiConfig";
import { cx, Spinner } from "./ui";
import s from "./MediaViewer.module.css";

/**
 * Кадр с необязательной подсветкой рамками (A12 shapes; координаты нормированы к кадру).
 * Если изображение не загрузилось (повреждённые данные, недоступный файл), вместо него — плашка с причиной:
 * ни заглушка, ни соседний кадр не подставляются, рамки не рисуются.
 */
export function EvidenceImage({
  src,
  boxes,
  alt,
  className,
  imgClassName,
  imgStyle,
  errorClassName,
  onImgLoad,
}: {
  src: string;
  boxes: HighlightBox[];
  alt: string;
  className?: string;
  imgClassName?: string;
  /** Явный размер изображения (лайтбокс вписывает кадр в окно). */
  imgStyle?: CSSProperties;
  errorClassName?: string;
  onImgLoad?: (img: HTMLImageElement) => void;
}) {
  // Ошибка привязана к конкретному src: при переходе к другому кадру плашка не «залипает».
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  if (failedSrc === src) {
    return (
      <div className={cx("relative", className ?? "block")}>
        <div
          className={cx(s.imgError, errorClassName)}
          role="img"
          aria-label={`${alt}: изображение не загрузилось — данные повреждены или файл недоступен`}
        >
          <ImageOff className="h-5 w-5 shrink-0" aria-hidden />
          <span className="font-semibold">Кадр не загрузился</span>
          <span className={cx("opacity-80", s.imgErrorMore)}>данные повреждены или файл недоступен</span>
        </div>
      </div>
    );
  }
  return (
    <div className={cx("relative", className ?? "block")}>
      <img
        src={src}
        alt={alt}
        draggable={false}
        style={imgStyle}
        onLoad={onImgLoad ? (e) => onImgLoad(e.currentTarget) : undefined}
        onError={() => setFailedSrc(src)}
        className={cx("block select-none", imgClassName ?? "h-auto w-full")}
      />
      {boxes.length > 0 && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
          {boxes.map((b, i) => (
            <rect
              key={i}
              x={b.bbox[0]}
              y={b.bbox[1]}
              width={b.bbox[2] - b.bbox[0]}
              height={b.bbox[3] - b.bbox[1]}
              fill="none"
              stroke={b.color}
              strokeWidth={2.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      )}
      {boxes.length > 0 && (
        <div className="pointer-events-none absolute left-1 top-1 flex max-w-[95%] flex-wrap gap-1">
          {boxes.slice(0, 3).map((b, i) =>
            b.label ? (
              <span key={i} className="rounded px-1 py-0.5 text-[10px] font-semibold leading-none text-white" style={{ background: b.color }}>
                {b.label}
              </span>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

/** Рамки A12 заданы для кадра t_peak — рисуем их только на нём (|ts − t_peak| ≤ maxDt), не на соседних кадрах. */
export function boxesForFrame(frame: EvidenceFrame, peakT: number, boxes: HighlightBox[], maxDt = uiConfig().highlightMaxDtS): HighlightBox[] {
  return !frame.missing && Math.abs(frame.timestamp_sec - peakT) <= maxDt ? boxes : [];
}

/**
 * Кадры-доказательства эпизода как последовательность по времени: номер шага, фактический таймкод кадра
 * (timestamp_sec сервера), отметка кадра в момент эпизода (|ts − t_peak| ≤ highlightMaxDtS). Кадр без изображения —
 * плашка «кадр не передан» с переходом к этому моменту в видео; ничем не подменяется. onOpen получает индекс в списке
 * кадров С изображением в порядке сервера (frames.filter(f => !f.missing)) — тот же список передаётся в Lightbox,
 * который сам листает его по времени.
 */
export function EvidenceSequence({
  frames,
  peakT,
  boxes,
  highlight,
  onOpen,
  onSeek,
}: {
  frames: EvidenceFrame[];
  /** t_peak, к которому относятся рамки. */
  peakT: number;
  boxes: HighlightBox[];
  highlight: boolean;
  onOpen: (shownIndex: number) => void;
  onSeek: (t: number) => void;
}) {
  const shown = frames.filter((f) => !f.missing);
  const maxDt = uiConfig().highlightMaxDtS;
  // Порядок показа — по времени кадра (сервер может прислать сначала кадр пика); при равенстве — как прислал сервер.
  const ordered = frames.map((f, k) => ({ f, k })).sort((a, b) => a.f.timestamp_sec - b.f.timestamp_sec || a.k - b.k);
  return (
    <ol className={s.seq} aria-label="Кадры-доказательства по времени">
      {ordered.map(({ f, k }, i) => {
        const tc = formatTimecode(f.timestamp_sec);
        const peak = !f.missing && Math.abs(f.timestamp_sec - peakT) <= maxDt;
        return (
          <li key={k} className={s.seqItem}>
            <div className={s.seqHead}>
              <span className={s.seqNum}>{i + 1}</span>
              <span className={s.seqTime}>{tc}</span>
            </div>
            {f.missing ? (
              <div className={s.seqMissing}>
                <ImageOff className="h-5 w-5 opacity-70" aria-hidden />
                <span>
                  <b>кадр не передан</b>
                  <span className="block opacity-80">сервер сообщил момент без изображения</span>
                </span>
                <button
                  type="button"
                  className={s.seqMissingBtn}
                  onClick={() => onSeek(f.timestamp_sec)}
                  aria-label={`Кадр ${i + 1} не передан — перейти к ${tc} в видео`}
                >
                  <Crosshair className="h-3.5 w-3.5" aria-hidden /> к {tc}
                </button>
              </div>
            ) : (
              <>
                <button
                  type="button"
                  className={cx(s.seqBtn, peak && s.seqBtnPeak)}
                  onClick={() => onOpen(shown.indexOf(f))}
                  aria-label={`Открыть кадр ${i + 1} · ${tc}${peak ? " · момент эпизода" : ""}`}
                >
                  <EvidenceImage
                    src={imageSrc(f.image_base64)}
                    boxes={highlight ? boxesForFrame(f, peakT, boxes, maxDt) : []}
                    alt={`Кадр ${tc}`}
                    imgClassName={s.thumbImg}
                  />
                </button>
                {peak && <div className={s.peakTag}>момент эпизода</div>}
              </>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/**
 * «До / после» для эпизодов-переходов («ушла и оставила», сейф оставлен открытым, сумка оставлена).
 * «До» — кадр сервера до начала эпизода, иначе кадр из видео в браузере (t_start − beforeS) с явной пометкой,
 * что это не кадр сервера; если видео не воспроизводится — причина и кнопка перехода к этому моменту.
 * «После» — только кадр сервера в момент/после начала; более ранний кадр не подставляется.
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

  const serverBefore = serverBeforeIdx >= 0 ? frames[serverBeforeIdx] : null;
  const capturedSrc = !serverBefore && captured && captured !== "loading" && captured !== "error" ? captured : null;
  const waiting = !serverBefore && !!capture && (captured === null || captured === "loading");
  const beforeReason = !capture
    ? "сервер не передал кадр до начала эпизода."
    : "сервер не передал кадр до начала эпизода, а взять его из видео в браузере не удалось: видео недоступно или не декодируется.";

  return (
    <div className="max-w-md">
      <div className="label mb-2">До и после начала эпизода · начало {formatTimecode(start)}</div>
      <div className={s.ba}>
        <figure className={s.baFig}>
          <div className={cx(s.baSlot, capturedSrc && s.baCaptured)}>
            {serverBefore ? (
              <button type="button" className={s.baSlotBtn} onClick={() => onOpen(serverBeforeIdx)} aria-label={`Открыть кадр «до» · ${formatTimecode(serverBefore.timestamp_sec)}`}>
                <EvidenceImage src={imageSrc(serverBefore.image_base64)} boxes={[]} alt="Кадр до начала эпизода" />
              </button>
            ) : capturedSrc ? (
              <button
                type="button"
                className={s.baSlotBtn}
                onClick={() => onSeek(beforeT)}
                aria-label={`Кадр из видео в браузере, не кадр сервера — перейти к ${formatTimecode(beforeT)} в видео`}
              >
                <EvidenceImage src={capturedSrc} boxes={[]} alt="Кадр из видео в браузере до начала эпизода" />
              </button>
            ) : (
              <div className={s.baEmpty} role={waiting ? "status" : undefined}>
                {waiting ? <Spinner className="h-5 w-5" /> : <ImageOff className="h-5 w-5" aria-hidden />}
                {waiting ? (
                  <span>Берём кадр из видео…</span>
                ) : (
                  <span>
                    <b className="text-white">кадра «до» нет</b> — {beforeReason}
                  </span>
                )}
                <button type="button" className="btn bg-white/10 px-2 py-1 text-[11px] text-white hover:bg-white/20" onClick={() => onSeek(beforeT)}>
                  <Crosshair className="h-3 w-3" aria-hidden /> к {formatTimecode(beforeT)}
                </button>
              </div>
            )}
          </div>
          <figcaption className={s.baCaption}>
            <b>До</b> · {formatTimecode(serverBefore ? serverBefore.timestamp_sec : beforeT)} ·{" "}
            {serverBefore ? "кадр сервера" : capturedSrc ? "кадр из видео" : "кадра нет"}
            {capturedSrc && <span className={s.baWarn}>не кадр сервера, без размытия</span>}
          </figcaption>
        </figure>
        <ArrowRight className={cx("h-5 w-5 shrink-0", s.baArrow)} aria-hidden />
        <figure className={s.baFig}>
          <div className={s.baSlot}>
            {after ? (
              <button type="button" className={s.baSlotBtn} onClick={() => onOpen(after.i)} aria-label={`Открыть кадр «после» · ${formatTimecode(after.f.timestamp_sec)}`}>
                <EvidenceImage src={imageSrc(after.f.image_base64)} boxes={highlight ? boxesForFrame(after.f, peakT, boxes) : []} alt="Кадр после начала эпизода" />
              </button>
            ) : (
              <div className={s.baEmpty}>
                <ImageOff className="h-5 w-5" aria-hidden />
                <span>
                  <b className="text-white">кадра «после» нет</b> — сервер не передал кадр в момент или после начала эпизода.
                </span>
              </div>
            )}
          </div>
          <figcaption className={s.baCaption}>
            <b>После</b> · {after ? `${formatTimecode(after.f.timestamp_sec)} · кадр сервера` : "кадра нет"}
          </figcaption>
        </figure>
      </div>
    </div>
  );
}
