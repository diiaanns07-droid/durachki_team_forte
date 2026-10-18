import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileVideo,
  History,
  Lock,
  PlayCircle,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import { ApiError, loadHistory, rememberLocalVideo, saveHistoryItem, type HistoryItem, type UploadHandle } from "../api/client";
import { useApp } from "../context";
import { formatBytes, formatDateTime } from "../lib/format";
import { navigate } from "../lib/router";
import { MAX_UPLOAD_MB, validateVideoFile } from "../lib/validate";
import { historyView } from "../lib/verdict";
import { batch } from "../lib/batch";
import BatchList from "./BatchList";
import { cx, LocalBadge, ProgressBar } from "./ui";

export default function UploadScreen() {
  const { backend, demo, serverOk, toast, openSettings, authenticated, refreshSession } = useApp();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(() => loadHistory());
  const inputRef = useRef<HTMLInputElement>(null);
  const handleRef = useRef<UploadHandle | null>(null);
  const startedAt = useRef(0);
  const [speed, setSpeed] = useState<number | null>(null);

  useEffect(() => () => handleRef.current?.abort(), []);

  // История: локальная (этот браузер) + серверная (GET /ui/v1/jobs, A02) — с именами файлов.
  useEffect(() => {
    if (!authenticated || !backend.listJobs) return;
    let alive = true;
    void backend.listJobs(20).then((server) => {
      if (!alive || server.length === 0) return;
      const local = loadHistory();
      const byId = new Map(local.map((h) => [h.job_id, h]));
      for (const sv of server) byId.set(sv.job_id, { ...sv, ...(byId.get(sv.job_id) ?? {}) });
      const merged = [...byId.values()].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
      setHistory(merged.slice(0, 20));
    });
    return () => {
      alive = false;
    };
  }, [authenticated, backend]);

  const pick = useCallback((f: File | undefined | null) => {
    if (!f) return;
    const err = validateVideoFile(f);
    setError(err);
    setFile(err ? null : f);
  }, []);

  /** Один файл — обычный путь с кнопкой «Начать проверку»; несколько — пакетная проверка. */
  const pickMany = useCallback(
    (list: FileList | null | undefined) => {
      const files = list ? Array.from(list) : [];
      if (files.length === 0) return;
      if (files.length === 1) return pick(files[0]);
      setError(null);
      setFile(null);
      batch.add(files);
      toast(`Пакетная проверка: ${files.length} файлов поставлено в очередь`);
    },
    [pick, toast],
  );

  const start = useCallback(async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    startedAt.current = performance.now();
    const h = backend.upload(file, (p) => {
      setProgress(p);
      const dt = (performance.now() - startedAt.current) / 1000;
      if (dt > 0.3) setSpeed((file.size * p) / dt);
    });
    handleRef.current = h;
    try {
      const jobId = await h.promise;
      rememberLocalVideo(jobId, file);
      saveHistoryItem({ job_id: jobId, file_name: file.name, size: file.size, status: "processing", demo: backend.kind === "demo" });
      navigate({ name: "job", id: jobId });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        toast("Загрузка отменена");
      } else {
        const msg = e instanceof ApiError ? e.message : "Не удалось загрузить файл.";
        if (e instanceof ApiError && e.status === 401) {
          setError("Сессия истекла — войдите снова и повторите загрузку.");
          void refreshSession();
        } else {
          setError(msg);
        }
      }
      setProgress(null);
    } finally {
      handleRef.current = null;
    }
  }, [backend, file, toast, refreshSession]);

  const uploading = progress !== null;

  return (
    <div className="animate-fade-in">
      <section className="relative overflow-hidden rounded-3xl border border-slate-200/70 bg-white px-5 py-8 shadow-card dark:border-white/[.06] dark:bg-ink-900 sm:px-10 sm:py-12">
        <div className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top_right,black,transparent_70%)]" />
        <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-brand-500/10 blur-3xl" />
        <div className="relative grid gap-8 lg:grid-cols-[1.05fr_1fr] lg:items-center">
          <div>
            <LocalBadge />
            <h1 className="mt-4 text-3xl font-extrabold leading-[1.1] text-slate-900 dark:text-white sm:text-[42px]">
              Проверка видеозаписи
              <br />
              <span className="bg-gradient-to-r from-brand-600 to-ink-600 bg-clip-text text-transparent dark:from-brand-300 dark:to-slate-200">
                кассового узла
              </span>
            </h1>
            <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-slate-600 dark:text-slate-300">
              Загрузите запись с потолочной камеры кассы. Система найдёт эпизоды с признаками возможных нарушений,
              покажет таймкод, кадры-доказательства и норму законодательства РК. Решение остаётся за проверяющим.
            </p>
            <ul className="mt-6 grid gap-2.5 text-sm text-slate-600 dark:text-slate-300 sm:grid-cols-2">
              {[
                [Lock, "Видео не покидает локальный сервер"],
                [Clock3, "Целевое время: ≤ 1 мин на запись 2,5 мин"],
                [ShieldCheck, "Лица не распознаются — только поведение"],
                [CheckCircle2, "Только видимые факты, без домыслов"],
              ].map(([Icon, text], i) => {
                const I = Icon as typeof Lock;
                return (
                  <li key={i} className="flex items-center gap-2">
                    <I className="h-4 w-4 shrink-0 text-ks-info" />
                    {text as string}
                  </li>
                );
              })}
            </ul>
          </div>

          <div>
            {demo ? (
              <DemoStart />
            ) : serverOk && !authenticated ? (
              <LoginCard />
            ) : (
            <div
              className={cx(
                "group relative flex min-h-[260px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition",
                drag
                  ? "border-[rgb(var(--ks-focus))] bg-ks-accent/5"
                  : "border-slate-300 bg-slate-50/60 hover:border-[rgb(var(--ks-focus))] hover:bg-ks-accent/[.03] dark:border-white/15 dark:bg-white/[.02] dark:hover:border-[rgb(var(--ks-focus))]",
                uploading && "pointer-events-none",
              )}
              onClick={() => !uploading && inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                if (!uploading) pickMany(e.dataTransfer.files);
              }}
              role="button"
              tabIndex={0}
              aria-label="Выбрать видеофайл"
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".mov,.mp4,video/quicktime,video/mp4"
                multiple
                className="hidden"
                onChange={(e) => {
                  pickMany(e.target.files);
                  e.target.value = "";
                }}
              />
              {!file ? (
                <>
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-ks-accent shadow-card ring-1 ring-slate-200 transition group-hover:scale-105 dark:bg-ink-800 dark:ring-white/10">
                    <UploadCloud className="h-7 w-7" />
                  </div>
                  <div className="text-base font-semibold text-slate-900 dark:text-white">Перетащите видео сюда</div>
                  <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                    или <span className="font-semibold text-ks-accent underline-offset-2 group-hover:underline">выберите файл</span>
                  </div>
                  <div className="mt-4 text-xs text-slate-400">.mov или .mp4 · до {MAX_UPLOAD_MB} МБ · можно несколько файлов сразу</div>
                </>
              ) : (
                <div className="w-full max-w-sm" onClick={(e) => e.stopPropagation()}>
                  <div className="flex items-center gap-3 rounded-xl bg-white p-3 text-left shadow-card ring-1 ring-slate-200 dark:bg-ink-800 dark:ring-white/10">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-ks-surface-2 text-ks-accent">
                      <FileVideo className="h-6 w-6" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold">{file.name}</div>
                      <div className="text-xs text-slate-500 tnum">{formatBytes(file.size)}</div>
                    </div>
                    {!uploading && (
                      <button className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-white/10" onClick={() => setFile(null)} aria-label="Убрать файл">
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                  {uploading ? (
                    <div className="mt-5 space-y-2">
                      <div className="flex justify-between text-xs text-slate-500 tnum">
                        <span>{progress! < 1 ? "Загрузка на сервер…" : "Создаём задачу…"}</span>
                        <span>
                          {Math.round(progress! * 100)} %{speed ? ` · ${formatBytes(speed)}/с` : ""}
                        </span>
                      </div>
                      <ProgressBar value={progress!} />
                      <button className="btn-ghost mt-2 w-full" onClick={() => handleRef.current?.abort()}>
                        Отменить
                      </button>
                    </div>
                  ) : (
                    <button className="btn-brand mt-5 w-full py-3 text-[15px]" onClick={() => void start()}>
                      <Sparkles className="h-4 w-4" /> Начать проверку
                    </button>
                  )}
                </div>
              )}
            </div>
            )}
            {error && (
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-400/20 dark:bg-rose-400/10 dark:text-rose-300" role="alert">
                <span>{error}</span>
                {file && !uploading && (
                  <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => void start()}>
                    Повторить загрузку
                  </button>
                )}
              </div>
            )}
            {serverOk === false && !demo && (
              <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-200">
                Сервер анализа не ответил. Проверьте, что он запущен, или укажите API-ключ в{" "}
                <button className="font-semibold underline" onClick={openSettings}>
                  настройках
                </button>
                . Посмотреть интерфейс можно в{" "}
                <a className="font-semibold underline" href="?demo=1#/">
                  демо-режиме
                </a>
                .
              </div>
            )}
          </div>
        </div>
      </section>

      {!demo && <BatchList />}

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="card p-5">
          <h2 className="mb-3 flex items-center gap-2 text-base font-bold">
            <History className="h-4 w-4 text-slate-400" /> Недавние проверки
          </h2>
          {history.length === 0 ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">Пока пусто — история хранится только в этом браузере.</p>
          ) : (
            <ul className="-mx-2 divide-y divide-slate-100 dark:divide-white/5">
              {history.slice(0, 8).map((h) => (
                <li key={h.job_id}>
                  <a href={`${h.demo && !demo ? "?demo=1" : ""}#/job/${encodeURIComponent(h.job_id)}`} className="flex items-center gap-3 rounded-lg px-2 py-2.5 hover:bg-slate-50 dark:hover:bg-white/5">
                    <span
                      className={cx(
                        "h-2.5 w-2.5 shrink-0 rounded-full",
                        {
                          danger: "bg-rose-500",
                          success: "bg-emerald-500",
                          warning: "bg-amber-500",
                          neutral: "bg-slate-300 dark:bg-slate-600",
                        }[historyView(h).tone],
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium">{h.file_name}</span>
                      <span className="block text-xs text-slate-500 tnum">
                        {formatDateTime(h.created_at)} · {historyView(h).label}
                      </span>
                    </span>
                    <ChevronRight className="h-4 w-4 text-slate-400" />
                  </a>
                </li>
              ))}
            </ul>
          )}
          {history.length > 0 && (
            <button className="mt-2 text-xs text-slate-500 hover:text-slate-800 dark:hover:text-slate-200" onClick={() => setHistory(loadHistory())}>
              обновить
            </button>
          )}
        </section>
        <section className="card p-5">
          <h2 className="mb-3 flex items-center gap-2 text-base font-bold">
            <PlayCircle className="h-4 w-4 text-slate-400" /> Как это работает
          </h2>
          <ol className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
            {[
              ["Загрузка", "Запись с камеры над кассой (.mov/.mp4). Поворот и ракурс учитываются автоматически."],
              ["Анализ", "Детекция людей и предметов, состояние сейфа и двери, правила, временная модель и локальная VLM."],
              ["Отчёт", "Вердикт, эпизоды с таймкодами, кадры-доказательства с размытыми головами, нормы РК со ссылкой на adilet."],
            ].map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink-900 text-xs font-bold text-white dark:bg-white dark:text-ink-900">{i + 1}</span>
                <span>
                  <b className="text-slate-900 dark:text-white">{t}.</b> {d}
                </span>
              </li>
            ))}
          </ol>
          <div className="mt-4 flex flex-wrap gap-2">
            <a className="btn-ghost" href="?demo=1#/job/demo">
              Открыть демо-отчёт
            </a>
            <a className="btn-ghost" href="#/about">
              Архитектура и модели
            </a>
            <a className="btn-ghost" href="#/replay" title="Просмотр сохранённого JSON-отчёта (повторный просмотр, не новый анализ)">
              Открыть сохранённый отчёт
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}

function DemoStart() {
  const { backend } = useApp();
  const [busy, setBusy] = useState(false);
  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center rounded-2xl border-2 border-dashed border-amber-300 bg-amber-50/60 p-6 text-center dark:border-amber-400/30 dark:bg-amber-400/5">
      <div className="text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300">Синтетическое демо</div>
      <p className="mt-2 max-w-sm text-sm text-slate-600 dark:text-slate-300">
        В демо-режиме реальные файлы не загружаются и не анализируются. Можно посмотреть синтетический пример отчёта.
      </p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        <button
          className="btn-brand"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            const id = await backend.upload(new File([], "synthetic_demo.mov"), () => undefined).promise;
            saveHistoryItem({ job_id: id, file_name: "synthetic_demo.mov", demo: true });
            navigate({ name: "job", id });
          }}
        >
          <Sparkles className="h-4 w-4" /> Симуляция обработки
        </button>
        <a className="btn-ghost" href="?demo=1#/job/demo">
          Отчёт с нарушениями
        </a>
        <a className="btn-ghost" href="?demo=1#/job/demo-clean">
          Отчёт без нарушений
        </a>
      </div>
      <a className="mt-4 text-xs font-semibold text-slate-500 underline-offset-2 hover:underline" href="./#/">
        Выйти из демо и загрузить реальное видео
      </a>
    </div>
  );
}

export function LoginCard({ onDone }: { onDone?: () => void } = {}) {
  const { login } = useApp();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <form
      className="flex min-h-[260px] flex-col justify-center rounded-2xl border border-slate-200 bg-white p-6 shadow-card dark:border-white/10 dark:bg-ink-800"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        setErr(null);
        try {
          await login(key);
          setKey("");
          onDone?.();
        } catch (ex) {
          setErr(ex instanceof ApiError ? ex.message : "Не удалось войти.");
        } finally {
          setBusy(false);
        }
      }}
    >
      <div className="flex items-center gap-2 text-base font-bold">
        <Lock className="h-4 w-4 text-ks-info" /> Вход для проверяющего
      </div>
      <p className="mt-1 text-sm text-slate-500">Введите ключ доступа. Сервер откроет защищённую сессию; ключ не сохраняется в браузере.</p>
      <input
        className="input mt-4 font-mono"
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder="ключ доступа"
        value={key}
        onChange={(e) => setKey(e.target.value)}
        aria-label="Ключ доступа"
      />
      {err && <div className="mt-2 text-sm text-rose-600 dark:text-rose-400">{err}</div>}
      <button className="btn-primary mt-4" type="submit" disabled={busy || !key.trim()}>
        {busy ? "Проверяем…" : "Войти"}
      </button>
    </form>
  );
}
