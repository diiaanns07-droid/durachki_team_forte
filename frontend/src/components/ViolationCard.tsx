import { useState } from "react";
import { Ban, CheckCircle2, Cpu, Image as ImageIcon, Scale, Sparkles, SquareDashed, Timer } from "lucide-react";
import type { JobResult, Violation } from "../api/types";
import { imageSrc, notCheckedFor, overlayBoxes, overlayPeak, rulesWithStatus, violationFacts, violationSources, violationVlm } from "../api/normalize";
import { uiConfig } from "../lib/uiConfig";
import { BeforeAfter, boxesForFrame, EvidenceImage } from "./Evidence";
import { typeInfo, sourceLabel } from "../lib/catalog";
import { formatTimecode } from "../lib/format";
import Lightbox from "./Lightbox";
import NormCard from "./NormCard";
import { ConfidenceBadge, cx } from "./ui";

interface Props {
  v: Violation;
  index: number;
  job: JobResult;
  active: boolean;
  onSeek: (t: number) => void;
  /** Кадр из видео в браузере (для «до»), null — видео недоступно. */
  capture?: (t: number) => Promise<string | null>;
  /** Кандидат A01: найден, но не вошёл в вердикт. */
  candidate?: boolean;
}

export default function ViolationCard({ v, index, job, active, onSeek, capture, candidate }: Props) {
  const info = typeInfo(v.type_id);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const [highlight, setHighlight] = useState(true);
  const boxes = overlayBoxes(v, job);
  const peakT = overlayPeak(v, job);
  // Кадры с изображением — для лайтбокса и «до/после»; кадры без изображения показываются плашкой.
  const shownFrames = v.evidence_frames.filter((f) => !f.missing);
  const notChecked = notCheckedFor(v, job);
  const transition = !!info?.transition && uiConfig().beforeAfterTypes.includes((v.type_id ?? "").toUpperCase());
  const facts = violationFacts(v, job);
  const sources = violationSources(v, job);
  const vlm = violationVlm(v, job);
  const rules = rulesWithStatus(v, job);

  return (
    <article
      id={`episode-${index + 1}`}
      className={cx("card scroll-mt-28 overflow-hidden transition", active && "ring-2 ring-brand-500/60")}
    >
      <header className="flex flex-wrap items-start gap-3 border-b border-slate-100 p-4 dark:border-white/5 sm:p-5">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-600 text-sm font-bold text-white">{index + 1}</span>
        <div className="min-w-0 flex-1">
          <h3 className="text-[17px] font-bold leading-snug text-slate-900 dark:text-white">{v.type}</h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            {v.type_id && (
              <span className="chip border-slate-200 bg-white font-mono text-slate-600 dark:border-white/10 dark:bg-white/5 dark:text-slate-300" title={info?.name}>
                {v.type_id}
              </span>
            )}
            {candidate && (
              <span className="chip border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-300">
                не вошло в вердикт
              </span>
            )}
            <ConfidenceBadge value={v.confidence} />
            {v.t_start !== undefined && v.t_end !== undefined && (
              <span className="chip border-slate-200 text-slate-500 tnum dark:border-white/10">
                <Timer className="h-3.5 w-3.5" />
                {formatTimecode(v.t_start)}–{formatTimecode(v.t_end)}
              </span>
            )}
          </div>
        </div>
        <button
          className="btn-brand shrink-0 px-3 py-2 font-mono tnum"
          onClick={() => onSeek(v.timecode_sec)}
          title="Перемотать видео к моменту эпизода"
        >
          ▶ {formatTimecode(v.timecode_sec)}
        </button>
      </header>

      <div className="space-y-5 p-4 sm:p-5">
        {v.description && <p className="text-[15px] leading-relaxed text-slate-700 dark:text-slate-200">{v.description}</p>}

        {v.evidence_frames.length > 0 ? (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="label flex items-center gap-1.5">
                <ImageIcon className="h-3.5 w-3.5" /> Кадры-доказательства
              </div>
              {boxes.length > 0 && (
                <button
                  className={cx(
                    "chip no-print",
                    highlight
                      ? "border-brand-300 bg-brand-50 text-brand-700 dark:border-brand-400/30 dark:bg-brand-400/10 dark:text-brand-200"
                      : "border-slate-200 text-slate-500 dark:border-white/10",
                  )}
                  aria-pressed={highlight}
                  onClick={() => setHighlight((h) => !h)}
                  title="Рамки объектов, на которые опирается вывод (от сервера)"
                >
                  <SquareDashed className="h-3.5 w-3.5" /> подсветка {highlight ? "вкл" : "выкл"}
                </button>
              )}
            </div>
            {transition && (
              <BeforeAfter v={v} peakT={peakT} boxes={boxes} highlight={highlight} capture={capture} onSeek={onSeek} onOpen={setLightbox} />
            )}
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
              {v.evidence_frames.map((f, i) =>
                f.missing ? (
                  <button
                    key={i}
                    className="flex aspect-[3/4] flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-slate-300 p-2 text-center text-[11px] text-slate-500 dark:border-white/15"
                    onClick={() => onSeek(f.timestamp_sec)}
                    title="Сервер сообщил кадр, но не передал изображение — откройте момент в видео"
                  >
                    кадр не передан
                    <span className="font-mono tnum">{formatTimecode(f.timestamp_sec)}</span>
                  </button>
                ) : (
                  <button
                    key={i}
                    className="group relative overflow-hidden rounded-lg bg-ink-950 ring-1 ring-slate-200 dark:ring-white/10"
                    onClick={() => setLightbox(shownFrames.indexOf(f))}
                    aria-label={`Открыть кадр ${formatTimecode(f.timestamp_sec)}`}
                  >
                    <EvidenceImage
                      src={imageSrc(f.image_base64)}
                      boxes={highlight ? boxesForFrame(f, peakT, boxes) : []}
                      alt={`Кадр ${formatTimecode(f.timestamp_sec)}`}
                      imgClassName="h-auto w-full transition duration-300 group-hover:scale-[1.03]"
                    />
                    <span className="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-white tnum">
                      {formatTimecode(f.timestamp_sec)}
                    </span>
                  </button>
                ),
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-500 dark:bg-white/5">Сервер не передал кадры-доказательства для этого эпизода.</div>
        )}

        <div>
          <div className="label mb-2 flex items-center gap-1.5">
            <Scale className="h-3.5 w-3.5" /> Норма законодательства РК
          </div>
          {rules.length > 0 ? (
            <div className="space-y-2">
              {rules.map((r, i) => (
                <NormCard key={i} rule={r} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-500 dark:bg-white/5">
              Для этого типа норма в каталоге не найдена — квалификацию определяет проверяющий.
            </div>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 p-4 dark:border-white/10">
          <div className="label mb-3 flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5" /> Как система пришла к выводу
          </div>
          {facts.length > 0 && (
            <ul className="space-y-1.5">
              {facts.map((f, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-700 dark:text-slate-300">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
                  {f}
                </li>
              ))}
            </ul>
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-1 text-xs text-slate-500">
              <Cpu className="h-3.5 w-3.5" /> Источники:
            </span>
            {vlm && (
              <span
                className={cx(
                  "chip",
                  vlm.tone === "ok" && "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-300",
                  vlm.tone === "warn" && "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-300",
                  vlm.tone === "muted" && "border-slate-200 text-slate-500 dark:border-white/10",
                )}
                title="Результат проверки эпизода локальной визуально-языковой моделью"
              >
                {vlm.label}
              </span>
            )}
            {sources.length ? (
              sources.map((s) => (
                <span key={s} className="chip border-ink-200 bg-ink-50 text-ink-700 dark:border-white/10 dark:bg-white/5 dark:text-slate-200" title={sourceLabel(s).hint}>
                  {sourceLabel(s).label}
                </span>
              ))
            ) : (
              <span className="text-xs text-slate-400">не указаны сервером</span>
            )}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            Уверенность — внутренняя оценка модели, а не вероятность и не точность. Описание содержит только видимые на записи факты.
          </p>
        </div>

        {notChecked.length > 0 && (
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-white/[.03]">
            <div className="label mb-2 flex items-center gap-1.5">
              <Ban className="h-3.5 w-3.5" /> По видео не установить
            </div>
            <ul className="space-y-1 text-sm text-slate-600 dark:text-slate-300">
              {notChecked.map((x) => (
                <li key={x} className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-400" />
                  {x}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {lightbox !== null && (
        <Lightbox
          frames={shownFrames}
          boxesFor={(f) => (highlight ? boxesForFrame(f, peakT, boxes) : [])}
          index={lightbox}
          caption={`Эпизод ${index + 1}: ${v.type}`}
          onClose={() => setLightbox(null)}
          onIndex={setLightbox}
          onSeek={onSeek}
        />
      )}
    </article>
  );
}
