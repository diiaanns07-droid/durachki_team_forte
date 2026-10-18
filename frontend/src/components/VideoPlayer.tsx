import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, Info, Pause, Play, RotateCcw, RotateCw, VideoOff, X } from "lucide-react";
import type { TimelinePoint, Violation } from "../api/types";
import { demoSceneDataUri } from "../api/demo";
import { formatTimecode } from "../lib/format";
import { uiConfig } from "../lib/uiConfig";
import RiskTimeline from "./RiskTimeline";
import { cx, Spinner } from "./ui";
import s from "./MediaViewer.module.css";

export interface PlayerHandle {
  /** Переход к моменту (время анализа, сек от первого кадра). По умолчанию — пауза на кадре (uiConfig.seekPauses). */
  seek: (t: number, play?: boolean) => void;
  /** Проиграть интервал [from, to] и остановиться (для «Смотреть эпизод»). */
  playRange: (from: number, to: number) => void;
  /** Кадр из видео в браузере в момент t (JPEG data-URI) или null, если видео недоступно. */
  captureAt: (t: number) => Promise<string | null>;
}

export interface VideoSource {
  url: string;
  /** Запасные источники по порядку (напр. серверный маршрут с перекодированным preview.mp4, затем blob по ключу). */
  fallbacks?: (() => Promise<string>)[];
}

interface Props {
  source: VideoSource | null;
  /** Синтетическая сцена вместо видео (только демо). */
  synthetic?: boolean;
  duration: number;
  timeline: TimelinePoint[];
  violations: Violation[];
  hints?: { timecode_sec: number; type: string }[];
  onTime?: (t: number) => void;
  /**
   * Сдвиг: время медиа = время анализа + timeOffset (напр. PTS первого кадра, если браузер его не вычитает).
   * Задаётся флагом uiConfig.ptsOffsetMode; по умолчанию 0.
   */
  timeOffset?: number;
  /** Оверлей поверх видимой области кадра (object-fit: contain): (время анализа, размер области px). */
  overlay?: (t: number, box: { w: number; h: number }) => ReactNode;
  /** Выбор эпизода с шкалы — общий с навигатором и карточками. */
  onSelect?: (i: number) => void;
}

/** Видимая область кадра внутри сцены при object-fit: contain (поля по бокам или сверху/снизу). */
export function contentRect(stage: { w: number; h: number }, aspect: number): { left: number; top: number; w: number; h: number } {
  let w = stage.w;
  let h = aspect > 0 ? stage.w / aspect : stage.h;
  if (h > stage.h) {
    h = stage.h;
    w = stage.h * aspect;
  }
  return { left: (stage.w - w) / 2, top: (stage.h - h) / 2, w, h };
}

const SPEEDS = [0.5, 1, 2];
/** Пропорции синтетической сцены демо (портретная камера над кассой, 464×848). */
const DEMO_ASPECT = 464 / 848;
/** Минимальная ширина поля сбоку от кадра (px), чтобы подписи стояли в нём, а не поверх кадра. */
const MARGIN_MIN = 140;

type SeekMode = "jump" | "scrub" | "play";

/** Итог действия с видео: warn — действие не выполнено (с причиной), info — пояснение. */
interface Notice {
  tone: "warn" | "info";
  text: string;
  /** Переход ждёт загрузки видео; сообщение снимается, когда переход выполнен. */
  pending?: boolean;
}

/** Причина ошибки <video> по коду MediaError. */
function mediaErrorReason(code: number | undefined): string {
  switch (code) {
    case 1:
      return "загрузка видео прервана";
    case 2:
      return "ошибка чтения или загрузки видео (сеть недоступна или файл повреждён)";
    case 3:
      return "видео не удалось декодировать (файл повреждён или кодек не поддерживается)";
    case 4:
      return "формат или кодек не поддерживается браузером (например, HEVC/H.265), либо файл недоступен";
    default:
      return "браузер не сообщил причину";
  }
}

const VideoPlayer = forwardRef<PlayerHandle, Props>(function VideoPlayer(
  { source, synthetic, duration, timeline, violations, hints, onTime, timeOffset = 0, overlay, onSelect },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [src, setSrc] = useState<string | null>(source?.url ?? null);
  const [fallbackIdx, setFallbackIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!source);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [realDuration, setRealDuration] = useState(0);
  const [aspect, setAspect] = useState<number | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const stopAt = useRef<number | null>(null);
  /** Переход, запрошенный до загрузки метаданных, — выполняется в onLoadedMetadata. */
  const pending = useRef<{ t: number; mode: SeekMode } | null>(null);
  const d = realDuration > 0 ? realDuration : duration;
  const virtual = !source && !!synthetic;
  /** Почему перемотка и воспроизведение невозможны; null — возможны. */
  const blocked = virtual ? null : error ? "видео не воспроизводится в этом браузере" : !src ? "исходное видео недоступно для этой задачи" : null;

  // Размер сцены — чтобы понять, хватает ли полей по бокам портретного кадра для подписей.
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageBox, setStageBox] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    const el = stageRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setStageBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Актуальные значения для стабильных колбэков (ручка плеера, горячие клавиши).
  const live = useRef({ t, d, blocked, virtual, src, timeOffset, speed });
  live.current = { t, d, blocked, virtual, src, timeOffset, speed };

  useEffect(() => {
    setSrc(source?.url ?? null);
    setFallbackIdx(0);
    setError(null);
    setLoading(!!source);
    setAspect(null);
    setRealDuration(0);
    setNotice(null);
    pending.current = null;
  }, [source]);

  // виртуальные часы синтетической сцены
  useEffect(() => {
    if (!virtual || !playing) return;
    let last = performance.now();
    const id = window.setInterval(() => {
      const now = performance.now();
      setT((x) => {
        const nx = x + ((now - last) / 1000) * speed;
        if (stopAt.current !== null && nx >= stopAt.current) {
          stopAt.current = null;
          setPlaying(false);
          return nx;
        }
        if (nx >= d) {
          setPlaying(false);
          return d;
        }
        return nx;
      });
      last = now;
    }, 100);
    return () => window.clearInterval(id);
  }, [virtual, playing, speed, d]);

  useEffect(() => onTime?.(t), [t, onTime]);

  const say = useCallback((tone: Notice["tone"], text: string, isPending = false) => setNotice({ tone, text, pending: isPending }), []);

  const startPlay = useCallback(
    (v: HTMLVideoElement) => {
      void v.play().catch((e: unknown) => {
        const name = e instanceof DOMException ? e.name : "";
        if (name === "AbortError") return; // прервано паузой или новой перемоткой — не ошибка
        say(
          "warn",
          name === "NotAllowedError"
            ? "Браузер не разрешил запустить воспроизведение без нажатия — нажмите ▶."
            : "Браузер не смог запустить воспроизведение этого видео.",
        );
      });
    },
    [say],
  );

  /** time — время анализа; "jump" ставит паузу на кадре (если включено), "scrub" — нет, "play" — воспроизводит. */
  const seekTo = useCallback(
    (time: number, mode: SeekMode = "scrub"): boolean => {
      const { d, blocked, virtual, timeOffset } = live.current;
      const want = Math.max(0, Number.isFinite(time) ? time : 0);
      const tt = d > 0 ? Math.min(want, d) : want;
      stopAt.current = null;
      if (blocked) {
        // Видео нет: перемотки не было — так и сообщаем. Отмечаем только момент на шкале риска (данные сервера),
        // чтобы был виден выбранный эпизод; время подписано «видео недоступно».
        setT(tt);
        say("warn", `${mode === "play" ? "Воспроизведение не запущено" : "Перемотка не выполнена"}: ${blocked}. На шкале риска отмечен момент ${formatTimecode(tt)}.`);
        return false;
      }
      if (virtual) {
        setT(tt);
        if (mode === "play") setPlaying(true);
        else if (mode === "jump" && uiConfig().seekPauses) setPlaying(false);
      } else {
        const v = videoRef.current;
        if (!v || v.readyState < 1) {
          pending.current = { t: tt, mode };
          say("info", `Видео ещё загружается — переход к ${formatTimecode(tt)} выполнится, как только оно будет готово.`, true);
          return true;
        }
        v.currentTime = tt + timeOffset;
        if (mode === "play") startPlay(v);
        else if (mode === "jump" && uiConfig().seekPauses) v.pause();
        setT(tt);
      }
      if (mode !== "scrub" && d > 0 && want > d + 0.05) {
        say("info", `Момент ${formatTimecode(want)} за пределами записи (длительность ${formatTimecode(d)}) — показан конец записи.`);
      } else {
        setNotice(null);
      }
      return true;
    },
    [say, startPlay],
  );

  const seek = useCallback((time: number, play = false) => void seekTo(time, play ? "play" : "jump"), [seekTo]);

  const playRange = useCallback(
    (from: number, to: number) => {
      if (seekTo(from, "play")) stopAt.current = to;
    },
    [seekTo],
  );

  const captureAt = useCallback(async (time: number): Promise<string | null> => {
    const { virtual, blocked, src, timeOffset } = live.current;
    if (virtual) return demoSceneDataUri(Math.max(0, time));
    if (blocked || !src) return null;
    return new Promise((resolve) => {
      const v = document.createElement("video");
      let settled = false;
      const done = (r: string | null) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        v.removeAttribute("src");
        v.load();
        resolve(r);
      };
      const timer = window.setTimeout(() => done(null), 10000);
      v.muted = true;
      v.preload = "auto";
      v.playsInline = true;
      v.onerror = () => done(null);
      v.onloadeddata = () => {
        const target = Math.max(0, time + timeOffset);
        v.currentTime = Number.isFinite(v.duration) ? Math.min(target, Math.max(0, v.duration - 0.05)) : target;
      };
      v.onseeked = () => {
        try {
          const scale = Math.min(1, 640 / Math.max(1, v.videoWidth, v.videoHeight));
          const c = document.createElement("canvas");
          c.width = Math.max(1, Math.round(v.videoWidth * scale));
          c.height = Math.max(1, Math.round(v.videoHeight * scale));
          c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);
          done(c.toDataURL("image/jpeg", 0.85));
        } catch {
          done(null);
        }
      };
      v.src = src;
    });
  }, []);

  useImperativeHandle(ref, () => ({ seek, playRange, captureAt }), [seek, playRange, captureAt]);

  // Горячие клавиши (когда фокус не в поле ввода): Space — пауза, ←/→ — ±5 с, N/P — эпизоды.
  // Без видео они двигают только отмеченный момент на шкале, с сообщением, что видео недоступно.
  const toggleRef = useRef<() => void>(() => undefined);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(el.tagName) || el.getAttribute("role") === "slider")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (document.querySelector('[role="dialog"]')) return;
      const cur = live.current.t;
      const code = e.code;
      if (code === "Space") {
        e.preventDefault();
        toggleRef.current();
      } else if (code === "ArrowRight") {
        e.preventDefault();
        seekTo(cur + 5, "scrub");
      } else if (code === "ArrowLeft") {
        e.preventDefault();
        seekTo(cur - 5, "scrub");
      } else if (code === "KeyN" || code === "KeyP") {
        const peaks = violations.map((v) => v.timecode_sec).sort((a, b) => a - b);
        const target = code === "KeyN" ? peaks.find((x) => x > cur + 0.3) : [...peaks].reverse().find((x) => x < cur - 0.3);
        if (target !== undefined) seek(target);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [seek, seekTo, violations]);

  const toggle = () => {
    if (virtual) {
      if (t >= d) setT(0);
      setPlaying((p) => !p);
      return;
    }
    const v = videoRef.current;
    if (blocked) {
      say("warn", `Воспроизведение недоступно: ${blocked}.`);
      return;
    }
    if (!v) return;
    if (v.paused) startPlay(v);
    else v.pause();
  };
  toggleRef.current = toggle;

  const onVideoError = async (code: number | undefined) => {
    const list = source?.fallbacks ?? [];
    for (let i = fallbackIdx; i < list.length; i++) {
      setFallbackIdx(i + 1);
      try {
        const next = await list[i]();
        if (next && next !== src) {
          setLoading(true);
          setAspect(null);
          setSrc(next);
          return;
        }
      } catch {
        /* следующий источник */
      }
    }
    const p = pending.current;
    pending.current = null;
    stopAt.current = null;
    setLoading(false);
    setPlaying(false);
    setError(`Причина: ${mediaErrorReason(code)}.`);
    if (p) setT(p.t);
    setNotice(
      p
        ? { tone: "warn", text: `Перемотка не выполнена: видео не воспроизводится в этом браузере. На шкале риска отмечен момент ${formatTimecode(p.t)}.` }
        : null,
    );
  };

  const showVideo = !!src && !error;
  const shownAspect = virtual ? DEMO_ASPECT : showVideo ? aspect : null;
  const landscape = shownAspect !== null && shownAspect > 1;
  const compact = !virtual && !showVideo;
  const active = blocked ? -1 : violations.findIndex((v) => t >= (v.t_start ?? v.timecode_sec - 1) && t <= (v.t_end ?? v.timecode_sec + 1));
  const av = active >= 0 ? violations[active] : null;
  // Поля по бокам портретного кадра: если в них хватает места, подписи стоят там и не закрывают кадр.
  const side = stageBox && shownAspect && !landscape ? (stageBox.w - Math.min(stageBox.w, stageBox.h * shownAspect)) / 2 : 0;
  const inMargins = side >= MARGIN_MIN;
  // Поверх кадра — только номер и интервал (название — в навигаторе и карточке под плеером).
  const compactChip = !inMargins;
  const badge = virtual ? <span className={s.badge}>синтетическая сцена · демо</span> : null;
  const rect = overlay && showVideo && stageBox && shownAspect ? contentRect(stageBox, shownAspect) : null;
  const range = av
    ? av.t_start !== undefined && av.t_end !== undefined
      ? `${formatTimecode(av.t_start)}–${formatTimecode(av.t_end)}`
      : formatTimecode(av.timecode_sec)
    : "";
  const chip = av ? (
    <div className={cx(s.episodeChip, compactChip && s.episodeChipCompact)}>
      <span className={s.episodeNum}>{active + 1}</span>
      {compactChip ? (
        <span className={s.episodeTime}>{range}</span>
      ) : (
        <span className="min-w-0">
          {av.type}
          <span className={s.episodeTime}>{range}</span>
        </span>
      )}
    </div>
  ) : null;

  const noticeEl = notice && (
    <div className={cx(s.notice, notice.tone === "info" && s.noticeInfo, compact && s.noticeInline)} role={notice.tone === "warn" ? "alert" : "status"}>
      {notice.tone === "warn" ? <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden /> : <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />}
      <span className={s.noticeText}>{notice.text}</span>
      <button type="button" className={s.noticeClose} onClick={() => setNotice(null)} aria-label="Скрыть сообщение">
        <X className="h-4 w-4" />
      </button>
    </div>
  );

  return (
    <div>
      <div ref={stageRef} className={cx(s.stage, landscape && s.stageLandscape, compact && s.stageCompact)}>
        {showVideo ? (
          <video
            ref={videoRef}
            src={src ?? undefined}
            className={s.media}
            aria-label="Видео записи"
            playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              setLoading(false);
              if (v.videoWidth > 0 && v.videoHeight > 0) setAspect(v.videoWidth / v.videoHeight);
              v.playbackRate = live.current.speed;
              const dur = v.duration;
              const real = Number.isFinite(dur) && dur > 0 ? Math.max(0, dur - timeOffset) : 0;
              if (real > 0) setRealDuration(real);
              const p = pending.current;
              if (p) {
                pending.current = null;
                const tt = real > 0 ? Math.min(p.t, real) : p.t;
                v.currentTime = tt + timeOffset;
                if (p.mode === "play") startPlay(v);
                else if (p.mode === "jump" && uiConfig().seekPauses) v.pause();
                setT(tt);
                setNotice((n) => (n?.pending ? null : n));
              }
            }}
            onTimeUpdate={(e) => {
              const ct = e.currentTarget.currentTime - timeOffset;
              setT(Math.max(0, ct));
              if (stopAt.current !== null && ct >= stopAt.current) {
                stopAt.current = null;
                e.currentTarget.pause();
              }
            }}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onError={(e) => void onVideoError(e.currentTarget.error?.code)}
            onClick={toggle}
          />
        ) : virtual ? (
          <img src={demoSceneDataUri(t)} alt="Синтетическая сцена (демо)" className={s.media} draggable={false} onClick={toggle} />
        ) : (
          <div className={s.unavailable}>
            <span className={s.unavailableIcon}>
              {error ? <AlertTriangle className="h-6 w-6 text-amber-300" aria-hidden /> : <VideoOff className="h-6 w-6" aria-hidden />}
            </span>
            <div className={s.unavailableTitle}>{error ? "Браузер не смог воспроизвести видео" : "Исходное видео недоступно для этой задачи"}</div>
            {error && <p>{error}</p>}
            <p>Перемотка и кадры из видео недоступны. Кадры-доказательства сервера и таймкоды — в карточках эпизодов.</p>
            {noticeEl}
          </div>
        )}
        {rect && overlay && (
          <div className={s.overlayLayer} style={{ left: rect.left, top: rect.top, width: rect.w, height: rect.h }}>
            {overlay(t, { w: rect.w, h: rect.h })}
          </div>
        )}
        {loading && showVideo && (
          <div className={s.center} role="status" aria-label="Видео загружается">
            <Spinner className="h-7 w-7" />
          </div>
        )}
        {(badge || chip) &&
          (inMargins ? (
            <>
              {badge && (
                <div className={s.marginLeft} style={{ width: side - 24 }}>
                  {badge}
                </div>
              )}
              {chip && (
                <div className={s.marginRight} style={{ width: side - 24 }}>
                  {chip}
                </div>
              )}
            </>
          ) : (
            <div className={s.overlayTop}>
              {badge}
              {chip}
            </div>
          ))}
        {!compact && noticeEl}
      </div>

      <div className={s.controls}>
        <button
          type="button"
          className="btn-primary h-11 w-11 shrink-0 rounded-full p-0"
          onClick={toggle}
          aria-label={playing ? "Пауза" : "Воспроизвести"}
          title={playing ? "Пауза (Space)" : "Воспроизвести (Space)"}
          disabled={!!blocked}
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </button>
        <button type="button" className={s.iconBtn} onClick={() => seekTo(t - 5, "scrub")} aria-label="Назад 5 с" title="Назад 5 с (←)">
          <RotateCcw className="h-4 w-4" />
        </button>
        <button type="button" className={s.iconBtn} onClick={() => seekTo(t + 5, "scrub")} aria-label="Вперёд 5 с" title="Вперёд 5 с (→)">
          <RotateCw className="h-4 w-4" />
        </button>
        {/* font-mono text-sm — опора e2e (scripts/e2e-mock.mjs) */}
        <div className={cx(s.time, "font-mono text-sm")}>
          <span className="font-semibold">{formatTimecode(t)}</span>
          <span className={s.timeTotal}> / {d > 0 ? formatTimecode(d) : "—"}</span>
        </div>
        {blocked ? (
          <span className={s.timeNote}>видео недоступно — время по шкале</span>
        ) : (
          <span className={cx(s.hint, "hidden lg:ml-auto lg:inline")} title="Горячие клавиши">
            Space · ←/→ ±5 с · N/P эпизоды
          </span>
        )}
        <div className={cx(s.speeds, "ml-auto", !blocked && "lg:ml-2")} role="group" aria-label="Скорость воспроизведения">
          {SPEEDS.map((x) => (
            <button
              type="button"
              key={x}
              className={cx(s.speed, x === speed && s.speedOn)}
              aria-pressed={x === speed}
              disabled={!!blocked}
              onClick={() => {
                setSpeed(x);
                if (videoRef.current) videoRef.current.playbackRate = x;
              }}
            >
              {x}×
            </button>
          ))}
        </div>
      </div>
      {d > 0 ? (
        <div className="mt-5">
          <RiskTimeline duration={d} timeline={timeline} violations={violations} hints={hints} currentTime={t} onSeek={(x) => seekTo(x, "scrub")} onSelect={onSelect} />
        </div>
      ) : (
        <p className="mt-4 text-xs text-slate-500">Длительность записи неизвестна — шкала риска недоступна.</p>
      )}
    </div>
  );
});

export default VideoPlayer;
