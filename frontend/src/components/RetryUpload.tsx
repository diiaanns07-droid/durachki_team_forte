import { useId, useRef, useState } from "react";
import { FolderOpen, RotateCcw } from "lucide-react";
import { ApiError, loadHistory, localFile, rememberLocalVideo, saveHistoryItem } from "../api/client";
import { useApp } from "../context";
import { formatBytes } from "../lib/format";
import { navigate } from "../lib/router";
import { validateVideoFile } from "../lib/validate";
import { cx } from "./ui";
import s from "./u11.module.css";

/** Имя по умолчанию в истории, когда настоящее неизвестно (api/client saveHistoryItem). */
const UNKNOWN_NAME = "видео";

/**
 * Повторная отправка после сбоя (U11): новая задача с тем же файлом. Упавшие задачи сервер не дедуплицирует,
 * поэтому это реальное действие API (POST /analyze), а не «обновить страницу».
 * Файл ещё в памяти вкладки → одна кнопка. Иначе (страница перезагружалась) — выбор файла с подсказкой, КАКОЙ
 * файл нужен: имя и размер из истории этого браузера; другой файл — только после явного подтверждения.
 */
export default function RetryUpload({ jobId, className }: { jobId: string; className?: string }) {
  const { backend, toast } = useApp();
  const captionId = useId();
  const input = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [sending, setSending] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [mismatch, setMismatch] = useState<File | null>(null);
  if (backend.kind === "demo") return null; // в демо нет реального файла — повторять нечего

  const inMemory = localFile(jobId);
  const hist = loadHistory().find((h) => h.job_id === jobId);
  const expectedName = inMemory?.name ?? (hist?.file_name && hist.file_name !== UNKNOWN_NAME ? hist.file_name : undefined);
  const expectedSize = inMemory?.size ?? (typeof hist?.size === "number" && hist.size > 0 ? hist.size : undefined);
  const btnCls = cx(className ?? "btn-ghost", "max-w-full");

  const send = async (file: File) => {
    setProblem(null);
    setMismatch(null);
    setSending(file.name);
    setProgress(0);
    try {
      const id = await backend.upload(file, setProgress).promise;
      rememberLocalVideo(id, file);
      saveHistoryItem({ job_id: id, file_name: file.name, size: file.size, status: "processing" });
      navigate({ name: "job", id });
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Не удалось отправить файл повторно";
      toast(msg, "error");
      setProblem(`Файл не отправлен: ${msg}. Текущая задача и её отчёт не изменились — можно попробовать ещё раз.`);
    } finally {
      setProgress(null);
      setSending(null);
    }
  };

  const onPick = (f: File | undefined) => {
    if (input.current) input.current.value = ""; // тот же файл можно выбрать снова
    if (!f) return;
    const err = validateVideoFile(f);
    if (err) {
      setMismatch(null);
      setProblem(err);
      return;
    }
    const differs = (expectedName !== undefined && f.name !== expectedName) || (expectedSize !== undefined && f.size !== expectedSize);
    if (differs) {
      setProblem(null);
      setMismatch(f);
      return;
    }
    void send(f);
  };

  const expected = expectedName ? (
    <>
      <b className="font-semibold text-slate-800 dark:text-slate-100">«{expectedName}»</b>
      {expectedSize !== undefined && <span className="tnum"> · {formatBytes(expectedSize)}</span>}
    </>
  ) : null;

  return (
    <div className={s.retry}>
      {progress !== null ? (
        <div className="w-full min-w-[12rem]" role="status" aria-live="polite">
          <div className={cx("mb-1 text-xs text-slate-700 dark:text-slate-200", s.wrapAnywhere)}>
            Отправка «{sending}»… <span className="tnum">{Math.round(progress * 100)} %</span>
          </div>
          <div className={s.uploadBar}>
            <div className={s.uploadFill} style={{ width: `${Math.max(2, progress * 100)}%` }} />
          </div>
        </div>
      ) : mismatch ? (
        <div className="w-full rounded-xl border border-amber-300 bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-950 dark:border-amber-400/30 dark:bg-amber-400/10 dark:text-amber-100" role="alert">
          <p className={s.wrapAnywhere}>
            Выбран файл <b>«{mismatch.name}»</b> · <span className="tnum">{formatBytes(mismatch.size)}</span>, а в этой задаче был {expected}. Если это
            другая запись, она будет проверена как новая задача.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => void send(mismatch)}>
              Отправить выбранный файл
            </button>
            <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => input.current?.click()}>
              <FolderOpen className="h-3.5 w-3.5" aria-hidden /> Выбрать другой
            </button>
          </div>
        </div>
      ) : inMemory ? (
        <>
          <button className={btnCls} aria-describedby={captionId} onClick={() => void send(inMemory)}>
            <RotateCcw className="h-4 w-4 shrink-0" aria-hidden />
            Отправить этот файл повторно
          </button>
          <p id={captionId} className={cx("text-xs leading-relaxed text-slate-600 dark:text-slate-300", s.wrapAnywhere)}>
            {expected} — файл ещё открыт в этой вкладке. Будет создана новая задача; эта останется в истории.
          </p>
        </>
      ) : (
        <>
          <button className={btnCls} aria-describedby={captionId} onClick={() => input.current?.click()}>
            <FolderOpen className="h-4 w-4 shrink-0" aria-hidden />
            Выбрать файл для повторной отправки
          </button>
          <p id={captionId} className={cx("text-xs leading-relaxed text-slate-600 dark:text-slate-300", s.wrapAnywhere)}>
            {expected ? (
              <>Нужен тот же исходный файл: {expected}. </>
            ) : (
              <>Нужен исходный видеофайл этой записи (.mov или .mp4) — тот же, что отправлялся в эту задачу. </>
            )}
            Браузер не сохранил его после перезагрузки страницы — выберите его на диске. Будет создана новая задача.
          </p>
        </>
      )}
      {problem && (
        <p role="alert" className={cx("text-xs font-medium leading-relaxed text-rose-700 dark:text-rose-300", s.wrapAnywhere)}>
          {problem}
        </p>
      )}
      <input
        ref={input}
        type="file"
        accept=".mov,.mp4,video/quicktime,video/mp4"
        className="hidden"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => onPick(e.target.files?.[0])}
      />
    </div>
  );
}
