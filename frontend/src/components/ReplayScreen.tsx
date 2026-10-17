import { useId, useState, type DragEvent } from "react";
import { AlertTriangle, FileJson, FileVideo, History, X } from "lucide-react";
import { rememberLocalVideo } from "../api/client";
import { normalizeJob } from "../api/normalize";
import type { JobResult } from "../api/types";
import { formatBytes } from "../lib/format";
import ResultView from "./ResultView";
import { cx } from "./ui";
import s from "./a15.module.css";

interface Report {
  job: JobResult;
  raw: unknown;
  name: string;
}

type Slot = "json" | "video";

const isJsonFile = (f: File) => /\.json$/i.test(f.name) || f.type === "application/json";
const isVideoFile = (f: File) => f.type.startsWith("video/") || /\.(mov|mp4|m4v|webm|mkv|avi)$/i.test(f.name);

/**
 * Повторный просмотр СОХРАНЁННОГО отчёта (JSON из CLI A03 / «Полный отчёт JSON») + необязательное локальное видео.
 * Всё остаётся в браузере; к серверу запросов нет. Всегда помечено: это не новый анализ и не подмена ошибки.
 * Отчёт открывается сразу после выбора JSON; видео можно выбрать до или после (или перетащить оба файла сразу).
 */
export default function ReplayScreen() {
  const [report, setReport] = useState<Report | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState<Slot | null>(null);
  // Видео, выбранное после открытия отчёта, подхватывается плеером при новом монтировании ResultView.
  const [mount, setMount] = useState(0);
  const titleId = useId();

  const attachVideo = (f: File, jobId?: string) => {
    setVideo(f);
    if (jobId) {
      rememberLocalVideo(jobId, f);
      setMount((m) => m + 1);
    }
  };

  const openJson = async (f: File, withVideo: File | null) => {
    try {
      const raw: unknown = JSON.parse(await f.text());
      const job = normalizeJob(raw, `replay-${f.name}`);
      if (withVideo) rememberLocalVideo(job.job_id, withVideo);
      setReport({ job, raw, name: f.name });
    } catch (e) {
      setError(e instanceof Error ? `Не удалось прочитать отчёт «${f.name}»: ${e.message}` : "Не удалось прочитать отчёт");
    }
  };

  // Файлы разбираются по типу, на какую бы зону их ни положили.
  const take = (files: FileList | null | undefined, slot: Slot) => {
    const list = [...(files ?? [])];
    if (!list.length) return;
    setError(null);
    const json = list.find(isJsonFile);
    const vid = list.find(isVideoFile) ?? null;
    if (vid) attachVideo(vid, json ? undefined : report?.job.job_id);
    if (json) void openJson(json, vid ?? video);
    if (!json && !vid) setError(slot === "json" ? `«${list[0].name}» — не JSON-отчёт.` : `«${list[0].name}» — не видеофайл (.mov, .mp4).`);
  };

  const dropProps = (slot: Slot) => ({
    onDragOver: (e: DragEvent) => {
      e.preventDefault();
      setOver(slot);
    },
    onDragLeave: (e: DragEvent) => {
      // dragleave приходит и при переходе на дочерний элемент зоны — гасим подсветку только при выходе из неё.
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(null);
    },
    onDrop: (e: DragEvent) => {
      e.preventDefault();
      setOver(null);
      take(e.dataTransfer.files, slot);
    },
  });

  const reset = () => {
    setReport(null);
    setVideo(null);
    setError(null);
  };

  const videoInput = (label: string) => (
    <input
      type="file"
      accept=".mov,.mp4,video/*"
      className="sr-only"
      aria-label={label}
      onChange={(e) => {
        take(e.target.files, "video");
        e.target.value = "";
      }}
    />
  );

  return (
    // Классы модуля — только на собственных блоках: ResultView (U01) не вложен в них.
    <div className="space-y-6">
      <div className={cx(s.scope, s.enter, s.replayNote)} role="note">
        <span className={s.replayTag}>
          <History className="h-3.5 w-3.5" aria-hidden /> Повторный просмотр
        </span>
        <div className={s.noticeBody}>
          <b>Повторный просмотр сохранённого отчёта.</b> Это не новый анализ: показывается содержимое выбранного JSON-файла
          {report ? ` «${report.name}»` : ""}. Данные не отправляются на сервер.
          {report && (
            <div className="mt-1 text-[13px]">
              {video ? (
                <>
                  Видео: <b className={s.fileName}>«{video.name}»</b> <span className="tnum">· {formatBytes(video.size)}</span>
                </>
              ) : (
                "Видео не выбрано — переход по таймкодам недоступен."
              )}
            </div>
          )}
        </div>
        {report && (
          <div className={cx(s.actions, s.noticeAction)}>
            <label className={cx("btn-ghost cursor-pointer px-3 py-1.5 text-xs", s.fileBtn)}>
              <FileVideo className="h-3.5 w-3.5" aria-hidden /> {video ? "Заменить видео" : "Добавить видео"}
              {videoInput("Видео для отчёта")}
            </label>
            <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={reset}>
              Открыть другой отчёт
            </button>
          </div>
        )}
      </div>

      {!report && (
        <section className={cx(s.scope, s.enter, s.hero)} aria-labelledby={titleId}>
          <div className={s.grid} aria-hidden />
          <div className={s.heroGlow} aria-hidden />
          <div className={s.heroInner}>
            <div className={s.eyebrow}>Сохранённый отчёт</div>
            <h1 id={titleId} className={s.heroTitle}>
              Открыть отчёт для повторного просмотра
            </h1>
            <p className={s.heroText}>
              JSON — из CLI <code className={s.code}>backend.pipeline run --out</code> или кнопки «Полный отчёт JSON». Видео необязательно: тот же
              ролик нужен только для перехода по таймкодам, его можно добавить и после открытия. Файлы остаются в браузере.
            </p>

            <div className={s.drops}>
              <div className={s.drop} data-over={over === "json"} {...dropProps("json")}>
                <span className={s.dropStep}>
                  <span className={s.dropNum}>1</span>
                  <span className={s.eyebrow}>обязательно</span>
                </span>
                <label className="flex w-full cursor-pointer flex-col items-center gap-1.5">
                  <span className={s.dropIcon}>
                    <FileJson className="h-6 w-6" aria-hidden />
                  </span>
                  <span className={s.dropTitle}>Отчёт JSON</span>
                  <span className={s.dropHint}>Перетащите файл сюда или нажмите, чтобы выбрать. Отчёт откроется сразу.</span>
                  <input
                    type="file"
                    accept=".json,application/json"
                    className="sr-only"
                    aria-label="Отчёт JSON"
                    onChange={(e) => {
                      take(e.target.files, "json");
                      e.target.value = "";
                    }}
                  />
                </label>
              </div>

              <div className={s.drop} data-filled={!!video} data-over={over === "video"} {...dropProps("video")}>
                <span className={s.dropStep}>
                  <span className={s.dropNum}>2</span>
                  <span className={s.eyebrow}>по желанию</span>
                </span>
                <label className="flex w-full cursor-pointer flex-col items-center gap-1.5">
                  <span className={s.dropIcon}>
                    <FileVideo className="h-6 w-6" aria-hidden />
                  </span>
                  <span className={s.dropTitle}>Видео</span>
                  <span className={s.dropHint}>{video ? "Выбрано — можно заменить" : "Тот же ролик (.mov, .mp4) — для перехода по таймкодам"}</span>
                  {videoInput("Видео для отчёта")}
                </label>
                {video && (
                  <span className={s.fileChip}>
                    <span className={s.fileName}>{video.name}</span>
                    <span className="shrink-0 tnum">{formatBytes(video.size)}</span>
                    <button type="button" className={s.fileRemove} onClick={() => setVideo(null)} aria-label="Убрать видео">
                      <X className="h-4 w-4" aria-hidden />
                    </button>
                  </span>
                )}
              </div>
            </div>

            {error && (
              <div className={cx(s.notice, "mt-4")} data-tone="event" role="alert">
                <AlertTriangle className={s.noticeIcon} aria-hidden />
                <div className={s.noticeBody}>{error}</div>
              </div>
            )}
          </div>
        </section>
      )}

      {report && <ResultView key={mount} job={report.job} raw={report.raw} fileName={report.name} replay />}
    </div>
  );
}
