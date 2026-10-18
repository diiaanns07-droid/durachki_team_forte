import { useContext, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, Crosshair, ImageOff, Minus, Plus, X } from "lucide-react";
import type { EvidenceFrame } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { AppContext } from "../context";
import { EvidenceImage } from "./Evidence";
import { formatTimecode } from "../lib/format";
import { cx } from "./ui";
import s from "./MediaViewer.module.css";

interface Props {
  frames: EvidenceFrame[];
  index: number;
  caption: string;
  onClose: () => void;
  onIndex: (i: number) => void;
  onSeek: (t: number) => void;
  boxesFor?: (f: EvidenceFrame) => HighlightBox[];
}

const ZOOM_MAX = 6;
/** Небольшие кадры увеличиваются до размера окна, но не больше чем втрое (дальше — только зум). */
const FIT_MAX = 3;
const PAD = 16;
const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Миниатюра для навигации; при ошибке загрузки — значок, без подстановки другого кадра. */
function Thumb({ f, n, on, onPick }: { f: EvidenceFrame; n: number; on: boolean; onPick: () => void }) {
  const [bad, setBad] = useState(false);
  const tc = formatTimecode(f.timestamp_sec);
  return (
    <button
      type="button"
      className={cx(s.lbThumb, on && s.lbThumbOn)}
      onClick={onPick}
      aria-label={`Кадр ${n} · ${tc}${bad ? " · не загрузился" : ""}`}
      aria-current={on ? "true" : undefined}
      title={tc}
    >
      {bad ? <ImageOff className={cx("h-4 w-4", s.lbThumbErr)} aria-hidden /> : <img src={imageSrc(f.image_base64)} alt="" draggable={false} onError={() => setBad(true)} />}
    </button>
  );
}

/**
 * Просмотр кадра-доказательства: кадр целиком по размеру окна, зум (колесо, кнопки, двойной клик, +/−/0),
 * панорама, кадры ←/→ и лента миниатюр, закрытие кнопкой и Esc. Фокус удерживается внутри и возвращается
 * на кнопку, открывшую просмотр.
 */
export default function Lightbox({ frames, index, caption, onClose, onIndex, onSeek, boxesFor }: Props) {
  const demo = useContext(AppContext)?.demo ?? false;
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  const [nat, setNat] = useState<{ src: string; w: number; h: number } | null>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const view = useRef<HTMLDivElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const closeBtn = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const metaId = useId();
  const f = frames[index];
  const src = f ? imageSrc(f.image_base64) : "";
  // Листаем по времени кадра (сервер может прислать сначала кадр пика); onIndex получает индекс в frames.
  const order = frames.map((_, i) => i).sort((a, b) => frames[a].timestamp_sec - frames[b].timestamp_sec || a - b);
  const pos = Math.max(0, order.indexOf(index));

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [index]);

  useEffect(() => {
    if (zoom <= 1) setPan({ x: 0, y: 0 });
  }, [zoom]);

  // Фокус: запоминаем, откуда открыли; при закрытии возвращаем туда же, не прокручивая страницу
  // (после «Перейти к … в видео» страница уже едет к плееру).
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
    closeBtn.current?.focus({ preventScroll: true });
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
      if (opener?.isConnected) opener.focus({ preventScroll: true });
    };
  }, []);

  const latest = useRef({ order, pos, onClose, onIndex });
  latest.current = { order, pos, onClose, onIndex };
  useEffect(() => {
    const trap = (e: KeyboardEvent) => {
      const el = root.current;
      if (!el) return;
      const items = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((x) => x.getClientRects().length > 0);
      const a = document.activeElement;
      if (!items.length) {
        e.preventDefault();
        el.focus();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      if (!a || !el.contains(a) || a === el) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      } else if (e.shiftKey && a === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && a === last) {
        e.preventDefault();
        first.focus();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      const { order, pos, onClose, onIndex } = latest.current;
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "Tab") {
        trap(e);
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        if (pos < order.length - 1) onIndex(order[pos + 1]);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (pos > 0) onIndex(order[pos - 1]);
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        setZoom((z) => Math.min(ZOOM_MAX, z * 1.25));
      } else if (e.key === "-") {
        e.preventDefault();
        setZoom((z) => Math.max(1, z / 1.25));
      } else if (e.key === "0") {
        setZoom(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Размер области просмотра: кадр вписывается целиком при любой высоте подписи и панелей.
  const hasFrame = !!f;
  useLayoutEffect(() => {
    const el = view.current;
    if (!el) return;
    const update = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    update();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    ro?.observe(el);
    window.addEventListener("resize", update);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [hasFrame]);

  // Активная миниатюра — в видимой части ленты.
  useEffect(() => {
    const bar = strip.current;
    const el = bar?.children[pos] as HTMLElement | undefined;
    if (!bar || !el) return;
    const b = bar.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (r.left < b.left) bar.scrollLeft -= b.left - r.left + 8;
    else if (r.right > b.right) bar.scrollLeft += r.right - b.right + 8;
  }, [pos]);

  if (!f) return null;
  if (typeof document === "undefined") return null;

  const tc = formatTimecode(f.timestamp_sec);
  const n = frames.length;
  const availW = box ? Math.max(40, box.w - PAD) : 0;
  const availH = box ? Math.max(40, box.h - PAD) : 0;
  const known = nat && nat.src === src && nat.w > 0 && nat.h > 0 ? nat : null;
  const scale = known && box ? Math.min(availW / known.w, availH / known.h, FIT_MAX) : 0;
  const imgStyle = known && box
    ? { width: Math.round(known.w * scale), height: Math.round(known.h * scale), maxWidth: "none" }
    : box
      ? { maxWidth: availW, maxHeight: availH, width: "auto", height: "auto" }
      : undefined;
  const zoomIn = () => setZoom((z) => Math.min(ZOOM_MAX, z * 1.25));
  const zoomOut = () => setZoom((z) => Math.max(1, z / 1.25));

  // Портал в body: fixed-оверлей не зависит от transform/overflow предков; обёртка ks-ws переносит токены
  // рабочей области (как у Modal U01).
  return createPortal(
    <div className="ks-ws">
      <div
        ref={root}
        className={cx("no-print", s.lb)}
        role="dialog"
        aria-modal="true"
        aria-label="Кадр-доказательство"
        aria-describedby={`${titleId} ${metaId}`}
        tabIndex={-1}
      >
        <div className={s.lbTop}>
          <div className={s.lbTitleBox}>
            <h2 id={titleId} className={s.lbTitle}>
              {caption}
            </h2>
            <p id={metaId} className={s.lbMeta}>
              {demo && <span className={s.lbDemo}>демо · синтетический кадр</span>}
              <span>
                Кадр {pos + 1} из {n} · {tc}
              </span>
              <span className={s.lbKeys} aria-hidden>
                ←/→ кадры · +/− масштаб · Esc — закрыть
              </span>
            </p>
          </div>
          <button ref={closeBtn} type="button" className={s.lbBtn} onClick={onClose} aria-label="Закрыть просмотр" title="Закрыть (Esc)">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div
          ref={view}
          className={s.lbView}
          onWheel={(e) => setZoom((z) => Math.min(ZOOM_MAX, Math.max(1, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))))}
          onDoubleClick={() => setZoom((z) => (z > 1 ? 1 : 2.5))}
          onPointerDown={(e) => {
            drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
            // Захват только для панорамы при увеличении: при 100 % клик по кадру не должен уходить на фон.
            if (zoom > 1) (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
          }}
          onPointerMove={(e) => {
            if (!drag.current || zoom <= 1) return;
            setPan({ x: drag.current.px + (e.clientX - drag.current.x), y: drag.current.py + (e.clientY - drag.current.y) });
          }}
          onPointerUp={() => (drag.current = null)}
          onPointerCancel={() => (drag.current = null)}
          onClick={(e) => e.target === e.currentTarget && zoom === 1 && onClose()}
        >
          <div className={s.lbZoom} style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: zoom > 1 ? "grab" : "zoom-in" }}>
            <EvidenceImage
              src={src}
              boxes={boxesFor ? boxesFor(f) : []}
              alt={`Кадр-доказательство ${tc}`}
              className="relative inline-block align-top"
              imgClassName="object-contain"
              imgStyle={imgStyle}
              errorClassName={s.lbImgError}
              onImgLoad={(img) => setNat({ src, w: img.naturalWidth, h: img.naturalHeight })}
            />
          </div>
        </div>

        <div className={s.lbBottom}>
          {n > 1 && (
            <div className={s.lbGroup}>
              <button type="button" className={s.lbBtn} onClick={() => onIndex(order[pos - 1])} disabled={pos === 0} aria-label="Предыдущий кадр" title="Предыдущий кадр (←)">
                <ChevronLeft className="h-5 w-5" />
              </button>
              <div ref={strip} className={s.lbStrip} role="group" aria-label="Кадры эпизода по времени">
                {order.map((fi, k) => (
                  <Thumb key={fi} f={frames[fi]} n={k + 1} on={fi === index} onPick={() => onIndex(fi)} />
                ))}
              </div>
              <button type="button" className={s.lbBtn} onClick={() => onIndex(order[pos + 1])} disabled={pos >= n - 1} aria-label="Следующий кадр" title="Следующий кадр (→)">
                <ChevronRight className="h-5 w-5" />
              </button>
            </div>
          )}
          <div className={s.lbGroup} role="group" aria-label="Масштаб">
            <button type="button" className={s.lbBtn} onClick={zoomOut} disabled={zoom <= 1} aria-label="Уменьшить" title="Уменьшить (−)">
              <Minus className="h-4 w-4" />
            </button>
            <span className={s.lbZoomValue}>{Math.round(zoom * 100)} %</span>
            <button type="button" className={s.lbBtn} onClick={zoomIn} disabled={zoom >= ZOOM_MAX} aria-label="Увеличить" title="Увеличить (+)">
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <button
            type="button"
            className={s.lbAction}
            onClick={() => {
              onSeek(f.timestamp_sec);
              onClose();
            }}
          >
            <Crosshair className="h-4 w-4" aria-hidden /> Перейти к {tc} в видео
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
