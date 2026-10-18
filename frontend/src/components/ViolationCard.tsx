import { useId, useState } from "react";
import { AlertTriangle, ChevronDown, Eye, Image as ImageIcon, ImageOff, Lightbulb, Play, Scale, SquareDashed } from "lucide-react";
import type { JobResult, Violation } from "../api/types";
import { imageSrc, notCheckedFor, overlayBoxes, overlayPeak, rulesWithStatus, violationFacts, violationSources, violationVlm } from "../api/normalize";
import { uiConfig } from "../lib/uiConfig";
import { BeforeAfter, boxesForFrame, EvidenceImage } from "./Evidence";
import { typeInfo, sourceLabel } from "../lib/catalog";
import { formatTimecode, plural } from "../lib/format";
import Lightbox from "./Lightbox";
import NormCard from "./NormCard";
import { cx } from "./ui";
import s from "./ViolationCard.module.css";

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

/** Описание длиннее — сворачивается до трёх строк (раскрывается кнопкой, при печати — полностью). */
const LONG_DESCRIPTION = 260;

/** Те же предложения, которые normalize.violationFacts подставляет, когда сервер не передал факты. */
function descriptionSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((x) => x.trim())
    .filter((x) => x.length > 3)
    .slice(0, 5);
}

/** Длительность интервала эпизода «13,5 с» (только если сервер передал t_start/t_end). */
function spanLabel(a?: number, b?: number): string | null {
  if (a === undefined || b === undefined || !Number.isFinite(a) || !Number.isFinite(b) || b < a) return null;
  return `${(Math.round((b - a) * 10) / 10).toLocaleString("ru-RU")} с`;
}

/** Оценка слияния как есть (0..1, без процентов и уровней): это внутренний скор, а не вероятность. */
function scoreLabel(c?: number): string | null {
  if (c === undefined || !Number.isFinite(c)) return null;
  return (Math.round(Math.min(1, Math.max(0, c)) * 100) / 100).toLocaleString("ru-RU", { minimumFractionDigits: 2 });
}

export default function ViolationCard({ v, index, job, active, onSeek, capture, candidate }: Props) {
  const info = typeInfo(v.type_id);
  const [lightbox, setLightbox] = useState<number | null>(null);
  const [highlight, setHighlight] = useState(true);
  const [descOpen, setDescOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const uid = useId();
  const titleId = `${uid}-title`;
  const descId = `${uid}-desc`;
  const detailsId = `${uid}-details`;

  const boxes = overlayBoxes(v, job);
  const peakT = overlayPeak(v, job);
  // Кадры с изображением — для лайтбокса и «до/после»; кадры без изображения показываются плашкой.
  const shownFrames = v.evidence_frames.filter((f) => !f.missing);
  const missingFrames = v.evidence_frames.length - shownFrames.length;
  const notChecked = notCheckedFor(v, job);
  const transition = !!info?.transition && uiConfig().beforeAfterTypes.includes((v.type_id ?? "").toUpperCase());
  const facts = violationFacts(v, job);
  const factsFromDescription = !(v.facts && v.facts.length) && facts.length > 0 && facts.join("\n") === descriptionSentences(v.description).join("\n");
  const sources = violationSources(v, job);
  const vlm = violationVlm(v, job);
  const rules = rulesWithStatus(v, job);
  const unverifiedRules = rules.filter((r) => r.verified !== true).length;
  const span = spanLabel(v.t_start, v.t_end);
  const score = scoreLabel(v.confidence);
  const longDescription = v.description.length > LONG_DESCRIPTION;
  const evidenceMeta =
    (shownFrames.length > 0 ? `${shownFrames.length} ${plural(shownFrames.length, ["кадр", "кадра", "кадров"])}` : "кадров нет") +
    (missingFrames > 0 ? ` · ${missingFrames} не ${plural(missingFrames, ["передан", "переданы", "передано"])}` : "");

  // Критические предупреждения — на первом уровне, без дополнительного клика.
  const alerts: { key: string; text: string }[] = [];
  if (candidate) alerts.push({ key: "candidate", text: "Кандидат — не вошло в вердикт: система нашла эпизод, но не подтвердила его. Решение за проверяющим." });
  if (vlm?.tone === "warn") alerts.push({ key: "vlm", text: "Локальная VLM не подтвердила этот эпизод — источники системы расходятся." });

  return (
    <article
      id={`episode-${index + 1}`}
      aria-labelledby={titleId}
      data-active={active ? "true" : "false"}
      className={cx("card scroll-mt-28 overflow-hidden", s.card, active && s.active)}
    >
      {/* 1. Что обнаружено → когда */}
      <header className={s.head}>
        <span className={s.number} aria-hidden>
          {index + 1}
        </span>
        <div className="min-w-0">
          <p className={s.kicker}>
            <span>Эпизод {index + 1}</span>
            <span aria-hidden>·</span>
            <span>гипотеза системы</span>
            {v.type_id && (
              <>
                <span aria-hidden>·</span>
                <span className="font-mono normal-case tracking-normal" title={info?.name}>
                  {v.type_id}
                </span>
              </>
            )}
          </p>
          <h3 id={titleId} className={cx(s.title, s.breakWords)}>
            {v.type}
          </h3>
          {/* Момент эпизода — на кнопке перемотки; здесь только интервал (если сервер его передал). */}
          {v.t_start !== undefined && v.t_end !== undefined && (
            <p className={s.when}>
              <span className="tnum">
                интервал {formatTimecode(v.t_start)}–{formatTimecode(v.t_end)}
                {span && ` · ${span}`}
              </span>
            </p>
          )}
        </div>
        <button
          type="button"
          className={cx("btn-brand tnum", s.seek)}
          onClick={() => onSeek(v.timecode_sec)}
          aria-label={`Перейти к моменту ${formatTimecode(v.timecode_sec)} в видео`}
          title="Перемотать видео к моменту эпизода"
        >
          <Play className="h-3.5 w-3.5 fill-current" aria-hidden />
          <span className="font-mono">{formatTimecode(v.timecode_sec)}</span>
        </button>
      </header>

      {alerts.length > 0 && (
        <ul className={s.alerts} aria-label="Важно">
          {alerts.map((a) => (
            <li key={a.key} className={s.alert}>
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <span>{a.text}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Краткое объяснение */}
      {v.description ? (
        <div className={s.section}>
          <p id={descId} className={cx(s.description, s.breakWords, longDescription && !descOpen && s.clamp3)}>
            {v.description}
          </p>
          {longDescription && (
            <button
              type="button"
              className={cx(s.linkButton, "no-print mt-1")}
              aria-expanded={descOpen}
              aria-controls={descId}
              onClick={() => setDescOpen((o) => !o)}
            >
              {descOpen ? "Свернуть описание" : "Показать описание полностью"}
            </button>
          )}
        </div>
      ) : (
        <div className={s.section}>
          <p className="text-sm text-slate-500 dark:text-slate-400">Описание сервер не передал.</p>
        </div>
      )}

      {/* 2. Какие доказательства доступны */}
      <section className={s.section} aria-label="Доказательства">
        <div className={s.sectionHead}>
          <h4 className={s.sectionTitle}>
            <ImageIcon className="h-3.5 w-3.5" aria-hidden /> Доказательства
          </h4>
          <span className={s.sectionMeta}>{evidenceMeta}</span>
          {boxes.length > 0 && shownFrames.length > 0 && (
            <button
              type="button"
              className={cx(s.toggle, "no-print")}
              aria-pressed={highlight}
              onClick={() => setHighlight((h) => !h)}
              title="Рамки объектов, на которые опирается вывод (от сервера)"
            >
              <SquareDashed className="h-3.5 w-3.5" aria-hidden /> Подсветка объектов: {highlight ? "вкл" : "выкл"}
            </button>
          )}
        </div>

        {v.evidence_frames.length === 0 ? (
          <p className={s.warnNote}>
            <ImageOff className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>Сервер не передал кадры-доказательства для этого эпизода. Проверьте момент в видео.</span>
          </p>
        ) : (
          <>
            {transition && (
              <div className="mb-3">
                <BeforeAfter v={v} peakT={peakT} boxes={boxes} highlight={highlight} capture={capture} onSeek={onSeek} onOpen={setLightbox} />
              </div>
            )}
            <ul className={s.frames} aria-label="Кадры-доказательства">
              {v.evidence_frames.map((f, i) => (
                <li key={i}>
                  {f.missing ? (
                    <button
                      type="button"
                      className={s.frameMissing}
                      onClick={() => onSeek(f.timestamp_sec)}
                      title="Сервер сообщил кадр, но не передал изображение — откройте момент в видео"
                      aria-label={`Кадр ${formatTimecode(f.timestamp_sec)} не передан — перейти к моменту в видео`}
                    >
                      <ImageOff className="h-4 w-4" aria-hidden />
                      кадр не передан
                      <span className="font-mono tnum">{formatTimecode(f.timestamp_sec)}</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      className={s.frame}
                      onClick={() => setLightbox(shownFrames.indexOf(f))}
                      aria-label={`Открыть кадр ${formatTimecode(f.timestamp_sec)}`}
                    >
                      <EvidenceImage
                        src={imageSrc(f.image_base64)}
                        boxes={highlight ? boxesForFrame(f, peakT, boxes) : []}
                        alt={`Кадр ${formatTimecode(f.timestamp_sec)}`}
                        imgClassName={cx("h-auto w-full", s.frameImg)}
                      />
                      <span className={s.frameTime}>{formatTimecode(f.timestamp_sec)}</span>
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      {/* 3. На какой источник ссылается система */}
      <section className={s.section} aria-label="Нормативное основание">
        <div className={s.sectionHead}>
          <h4 className={s.sectionTitle}>
            <Scale className="h-3.5 w-3.5" aria-hidden /> Нормативное основание
          </h4>
          {rules.length > 0 && unverifiedRules > 0 && (
            <span className={s.sectionMeta}>
              {unverifiedRules === rules.length ? "не сверено с первоисточником" : `не сверено: ${unverifiedRules} из ${rules.length}`}
            </span>
          )}
        </div>
        {rules.length > 0 ? (
          <div className={s.norms}>
            {rules.map((r, i) => (
              <NormCard key={i} rule={r} />
            ))}
          </div>
        ) : (
          <p className={s.warnNote}>
            <Scale className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>
              Нормативное основание не установлено: для этого типа норма в каталоге не найдена. Квалифицировать эпизод должен
              проверяющий.
            </span>
          </p>
        )}
      </section>

      {/* Подробности: наблюдения, источники, оценка, ограничения */}
      <div className={cx(s.section, s.detailsWrap)}>
        <button
          type="button"
          className={cx(s.disclosure, "no-print")}
          aria-expanded={detailsOpen}
          aria-controls={detailsId}
          onClick={() => setDetailsOpen((o) => !o)}
        >
          <Lightbulb className="h-4 w-4 shrink-0 text-brand-600 dark:text-brand-300" aria-hidden />
          <span className="flex-1 text-left">Как система пришла к выводу</span>
          <ChevronDown className={cx("h-4 w-4 shrink-0", s.chevron, detailsOpen && s.chevronOpen)} aria-hidden />
        </button>
        <h4 className="print-only mb-2 text-sm font-semibold">Как система пришла к выводу</h4>

        <div id={detailsId} className={cx(s.details, detailsOpen ? "block animate-fade-in" : "hidden print-expand")}>
          <div className={s.layer}>
            <div className={s.layerLabel}>
              <Eye className="h-3.5 w-3.5" aria-hidden /> Наблюдения системы
            </div>
            {facts.length > 0 && !factsFromDescription ? (
              <ul className={s.facts}>
                {facts.map((f, i) => (
                  <li key={i} className={s.breakWords}>
                    {f}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">Перечень наблюдений сервер не передал — см. описание выше.</p>
            )}
          </div>

          <div className={s.layer}>
            <div className={s.layerLabel}>
              <Lightbulb className="h-3.5 w-3.5" aria-hidden /> Гипотеза
            </div>
            <p className="text-sm text-slate-700 dark:text-slate-200">
              «<span className={s.breakWords}>{v.type}</span>» — предположение системы по наблюдениям, а не юридический вывод. Требуется проверка
              человеком.
            </p>
            <dl className={cx(s.meta, "mt-2")}>
              <div>
                <dt>Источники сигнала</dt>
                <dd>
                  {sources.length ? (
                    sources.map((x, i) => (
                      <span key={x} title={sourceLabel(x).hint}>
                        {i > 0 && ", "}
                        {sourceLabel(x).label}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">не указаны сервером</span>
                  )}
                </dd>
              </div>
              {vlm && (
                <div>
                  <dt>Проверка VLM</dt>
                  <dd className={cx(vlm.tone === "warn" && "font-medium text-amber-800 dark:text-amber-200")}>{vlm.label}</dd>
                </div>
              )}
              <div>
                <dt>Оценка системы</dt>
                <dd>
                  {score ? (
                    <>
                      <span className="font-mono tnum">{score}</span>{" "}
                      <span className="text-slate-500 dark:text-slate-400">из 1 — внутренний скор слияния, не вероятность и не точность</span>
                    </>
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">не передана</span>
                  )}
                </dd>
              </div>
            </dl>
          </div>

          {notChecked.length > 0 && (
            <div className={s.layer}>
              <div className={s.layerLabel}>
                <AlertTriangle className="h-3.5 w-3.5" aria-hidden /> По видео не установить
              </div>
              <ul className={s.facts}>
                {notChecked.map((x) => (
                  <li key={x} className={s.breakWords}>
                    {x}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="mt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            Описание содержит только видимые на записи факты: умысел, сумма, подлинность и допуск по видео не устанавливаются.
          </p>
        </div>
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
