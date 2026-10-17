import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { AlertTriangle, Pause, Play, RotateCcw, RotateCw, VideoOff } from "lucide-react";
import type { TimelinePoint, Violation } from "../api/types";
import { demoSceneDataUri } from "../api/demo";
import { formatTimecode } from "../lib/format";
import { uiConfig } from "../lib/uiConfig";
import RiskTimeline from "./RiskTimeline";
import { cx, Spinner } from "./ui";

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
}

const SPEEDS = [0.5, 1, 2];

const VideoPlayer = forwardRef<PlayerHandle, Props>(function VideoPlayer(
  { source, synthetic, duration, timeline, violations, hints, onTime, timeOffset = 0 },
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
  const d = realDuration > 0 ? realDuration : duration;
  const virtual = !source && !!synthetic;

  useEffect(() => {
    setSrc(source?.url ?? null);
    setFallbackIdx(0);
    setError(null);
    setLoading(!!source);
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

  const stopAt = useRef<number | null>(null);

  /** time — время анализа; mode "jump" ставит паузу на кадре (если включено), "scrub" — нет. */
  const seekTo = useCallback(
    (time: number, mode: "jump" | "scrub" | "play" = "scrub") => {
      const tt = Math.min(Math.max(0, time), d || time);
      const v = videoRef.current;
      if (mode !== "play") stopAt.current = null;
      if (v && src) {
        v.currentTime = tt + timeOffset;
        if (mode === "play") void v.play().catch(() => undefined);
        else if (mode === "jump" && uiConfig().seekPauses) v.pause();
      }
      setT(tt);
      if (virtual) {
        if (mode === "play") setPlaying(true);
        else if (mode === "jump" && uiConfig().seekPauses) setPlaying(false);
      }
    },
    [d, src, virtual, timeOffset],
  );

  const seek = useCallback((time: number, play = false) => seekTo(time, play ? "play" : "jump"), [seekTo]);

  const playRange = useCallback(
    (from: number, to: number) => {
      seekTo(from, "play");
      stopAt.current = to;
    },
    [seekTo],
  );

  const captureAt = useCallback(
    async (time: number): Promise<string | null> => {
      if (virtual) return demoSceneDataUri(Math.max(0, time));
      if (!src) return null;
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
    },
    [src, virtual, timeOffset],
  );

  useImperativeHandle(ref, () => ({ seek, playRange, captureAt }), [seek, playRange, captureAt]);

  // Горячие клавиши (когда фокус не в поле ввода): Space — пауза, ←/→ — ±5 с, N/P — следующий/предыдущий эпизод.
  const keyState = useRef({ t, d });
  keyState.current = { t, d };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(el.tagName) || el.getAttribute("role") === "slider")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (document.querySelector('[role="dialog"]')) return;
      const cur = keyState.current.t;
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

  const toggleRef = useRef<() => void>(() => undefined);
  const toggle = () => {
    const v = videoRef.current;
    if (virtual) {
      if (t >= d) setT(0);
      setPlaying((p) => !p);
      return;
    }
    if (!v) return;
    if (v.paused) void v.play().catch(() => undefined);
    else v.pause();
  };

  toggleRef.current = toggle;

  const onVideoError = async () => {
    const list = source?.fallbacks ?? [];
    for (let i = fallbackIdx; i < list.length; i++) {
      setFallbackIdx(i + 1);
      try {
        const next = await list[i]();
        if (next && next !== src) {
          setLoading(true);
          setSrc(next);
          return;
        }
      } catch {
        /* следующий источник */
      }
    }
    setLoading(false);
    setError(
      "Браузер не смог воспроизвести видео (возможно, кодек HEVC или файл недоступен). Кадры-доказательства и таймкоды доступны в карточках эпизодов.",
    );
  };

  const active = violations.findIndex(
    (v) => t >= (v.t_start ?? v.timecode_sec - 1) && t <= (v.t_end ?? v.timecode_sec + 1),
  );

  return (
    <div>
      <div className="relative mx-auto flex max-h-[68vh] min-h-[220px] items-center justify-center overflow-hidden rounded-xl bg-ink-950">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            className="max-h-[68vh] w-full object-contain"
            playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              setLoading(false);
              const dur = e.currentTarget.duration;
              if (Number.isFinite(dur) && dur > 0) setRealDuration(Math.max(0, dur - timeOffset));
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
            onError={() => void onVideoError()}
            onClick={toggle}
          />
        ) : virtual ? (
          <img src={demoSceneDataUri(t)} alt="Синтетическая сцена (демо)" className="max-h-[68vh] w-auto max-w-full object-contain" onClick={toggle} />
        ) : (
          <div className="flex flex-col items-center gap-2 p-8 text-center text-sm text-slate-400">
            <VideoOff className="h-8 w-8" />
            Исходное видео недоступно для этой задачи.
          </div>
        )}
        {loading && src && !error && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-slate-300">
            <Spinner className="h-7 w-7" />
          </div>
        )}
        {error && (
          <div className="absolute inset-x-3 bottom-3 flex items-start gap-2 rounded-lg bg-black/70 p-3 text-xs text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {error}
          </div>
        )}
        {virtual && (
          <div className="pointer-events-none absolute left-3 top-3 rounded bg-amber-400 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-950">
            синтетическая сцена
          </div>
        )}
        {active >= 0 && (
          <div className="pointer-events-none absolute right-3 top-3 max-w-[70%] animate-fade-in rounded-lg bg-brand-600/95 px-2.5 py-1.5 text-xs font-semibold text-white shadow-lift">
            Эпизод {active + 1}: {violations[active].type}
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button className="btn-primary h-10 w-10 rounded-full p-0" onClick={toggle} aria-label={playing ? "Пауза" : "Воспроизвести"} disabled={!src && !virtual}>
          {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </button>
        <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-white/10" onClick={() => seekTo(t - 5, "scrub")} aria-label="Назад 5 с" title="Назад 5 с">
          <RotateCcw className="h-4 w-4" />
        </button>
        <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-white/10" onClick={() => seekTo(t + 5, "scrub")} aria-label="Вперёд 5 с" title="Вперёд 5 с">
          <RotateCw className="h-4 w-4" />
        </button>
        <div className="ml-1 font-mono text-sm tnum">
          <span className="font-semibold">{formatTimecode(t)}</span>
          <span className="text-slate-400"> / {formatTimecode(d)}</span>
        </div>
        <span className="hidden text-[11px] text-slate-400 lg:ml-auto lg:inline" title="Горячие клавиши">
          Space · ←/→ ±5 с · N/P эпизоды
        </span>
        <div className="ml-auto flex overflow-hidden rounded-lg border border-slate-200 text-xs dark:border-white/10 lg:ml-3">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={cx("px-2.5 py-1.5 font-semibold tnum", s === speed ? "bg-ink-900 text-white dark:bg-white dark:text-ink-900" : "text-slate-500 hover:bg-slate-50 dark:hover:bg-white/5")}
              onClick={() => {
                setSpeed(s);
                if (videoRef.current) videoRef.current.playbackRate = s;
              }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
      {d > 0 ? (
        <div className="mt-5">
          <RiskTimeline duration={d} timeline={timeline} violations={violations} hints={hints} currentTime={t} onSeek={(x) => seekTo(x, "scrub")} />
        </div>
      ) : (
        <p className="mt-4 text-xs text-slate-500">Длительность записи неизвестна — шкала риска недоступна.</p>
      )}
    </div>
  );
});

export default VideoPlayer;
