import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, Crosshair, EyeOff, Minus, Plus, X } from "lucide-react";
import type { EvidenceFrame } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { EvidenceImage } from "./Evidence";
import { formatTimecode } from "../lib/format";
import { cx } from "./ui";
import s from "./a15.module.css";

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
const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])';

function reducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Просмотр кадра-доказательства: зум (колесо/кнопки/двойной клик/«+ − 0»), панорама, ← →, лента кадров, Esc.
 * Фокус остаётся внутри диалога и возвращается на вызвавший элемент после закрытия.
 */
export default function Lightbox({ frames, index, caption, onClose, onIndex, onSeek, boxesFor }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const f = frames[index];

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [index]);

  // Фокус — в диалог при открытии, обратно на вызвавший элемент при закрытии.
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    root.current?.focus();
    return () => opener?.focus?.();
  }, []);

  // Текущая миниатюра — в поле зрения ленты.
  useEffect(() => {
    const el = strip.current?.querySelector<HTMLElement>('[aria-current="true"]');
    el?.scrollIntoView?.({ block: "nearest", inline: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  }, [index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "Tab" && root.current) {
        // Ловушка фокуса: Tab не уходит под оверлей.
        const items = [...root.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (e.shiftKey && (active === first || active === root.current || !root.current.contains(active))) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && (active === last || !root.current.contains(active))) {
          e.preventDefault();
          first.focus();
        }
        return;
      }
      if (e.key === "ArrowRight" && index < frames.length - 1) onIndex(index + 1);
      if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
      if (e.key === "+" || e.key === "=") setZoom((z) => Math.min(ZOOM_MAX, z * 1.25));
      if (e.key === "-") setZoom((z) => Math.max(1, z / 1.25));
      if (e.key === "0") {
        setZoom(1);
        setPan({ x: 0, y: 0 });
      }
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [index, frames.length, onClose, onIndex]);

  if (!f) return null;
  if (typeof document === "undefined") return null;
  const n = frames.length;
  const tc = formatTimecode(f.timestamp_sec);
  const pct = Math.round(zoom * 100);

  // Портал в body: fixed-оверлей не зависит от transform/overflow предков; ks-ws — токены рабочей области U01.
  return createPortal(
    <div
      ref={root}
      tabIndex={-1}
      className={cx("ks-ws no-print", s.scope, s.lb)}
      role="dialog"
      aria-modal="true"
      aria-label="Кадр-доказательство"
    >
      <div className={s.lbBar}>
        <div className={s.lbTitle}>
          <div className={s.lbCaption}>{caption}</div>
          <div className={s.lbMeta}>
            <span className={s.lbPill} aria-live="polite">
              Кадр {index + 1} из {n}
            </span>
            <span className={cx(s.lbPill, s.lbPillMono)}>{tc}</span>
            <span className={s.lbNote}>
              <EyeOff className="h-3.5 w-3.5" aria-hidden /> головы размыты на сервере
            </span>
          </div>
        </div>
        <button type="button" className={s.lbIconBtn} onClick={onClose} aria-label="Закрыть" title="Закрыть (Esc)">
          <X className="h-5 w-5" aria-hidden />
        </button>
      </div>
      <div
        className={s.lbStage}
        onWheel={(e) => setZoom((z) => Math.min(ZOOM_MAX, Math.max(1, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))))}
        onDoubleClick={() => {
          setZoom((z) => (z > 1 ? 1 : 2.5));
          setPan({ x: 0, y: 0 });
        }}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
          (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current || zoom <= 1) return;
          setPan({ x: drag.current.px + (e.clientX - drag.current.x), y: drag.current.py + (e.clientY - drag.current.y) });
        }}
        onPointerUp={() => (drag.current = null)}
        onClick={(e) => e.target === e.currentTarget && zoom === 1 && onClose()}
      >
        <div className={s.lbPan} style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: zoom > 1 ? "grab" : "zoom-in" }}>
          <EvidenceImage
            src={imageSrc(f.image_base64)}
            boxes={boxesFor ? boxesFor(f) : []}
            alt={`Кадр-доказательство ${tc}`}
            className={s.lbImgBox}
            imgClassName={s.lbImg}
          />
        </div>
        {n > 1 && (
          <>
            <button type="button" className={cx(s.lbIconBtn, s.lbPrev)} onClick={() => onIndex(index - 1)} disabled={index === 0} aria-label="Предыдущий кадр">
              <ChevronLeft className="h-5 w-5" aria-hidden />
            </button>
            <button type="button" className={cx(s.lbIconBtn, s.lbNext)} onClick={() => onIndex(index + 1)} disabled={index >= n - 1} aria-label="Следующий кадр">
              <ChevronRight className="h-5 w-5" aria-hidden />
            </button>
          </>
        )}
      </div>
      <div className={s.lbFoot}>
        {n > 1 && (
          <div ref={strip} className={s.lbStrip} role="group" aria-label="Кадры эпизода">
            {frames.map((fr, i) => (
              <button
                key={i}
                type="button"
                className={s.lbThumb}
                aria-current={i === index ? "true" : undefined}
                aria-label={`Кадр ${i + 1}: ${formatTimecode(fr.timestamp_sec)}`}
                onClick={() => onIndex(i)}
              >
                <img src={imageSrc(fr.image_base64)} alt="" draggable={false} />
              </button>
            ))}
          </div>
        )}
        <div className={s.lbControls}>
          <div className={s.lbZoom} role="group" aria-label="Масштаб">
            <button type="button" className={s.lbIconBtn} onClick={() => setZoom((z) => Math.max(1, z / 1.25))} aria-label="Уменьшить" disabled={zoom <= 1}>
              <Minus className="h-4 w-4" aria-hidden />
            </button>
            <button
              type="button"
              className={s.lbZoomValue}
              onClick={() => {
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
              aria-label={`Масштаб ${pct} % — сбросить`}
              title="Сбросить масштаб (0)"
            >
              {pct} %
            </button>
            <button type="button" className={s.lbIconBtn} onClick={() => setZoom((z) => Math.min(ZOOM_MAX, z * 1.25))} aria-label="Увеличить" disabled={zoom >= ZOOM_MAX}>
              <Plus className="h-4 w-4" aria-hidden />
            </button>
          </div>
          <button
            type="button"
            className={s.lbSeek}
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
