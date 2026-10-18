import { useEffect, useRef, useState } from "react";
import { AlertTriangle, RefreshCw, Upload } from "lucide-react";
import { ApiError, loadHistory, saveHistoryItem } from "../api/client";
import type { JobResult } from "../api/types";
import { useApp } from "../context";
import ProcessingView from "./ProcessingView";
import ResultView from "./ResultView";
import RetryUpload from "./RetryUpload";
import { LoginCard } from "./UploadScreen";
import { uiConfig } from "../lib/uiConfig";
import { analyzeCompleteness } from "../lib/completeness";
import { Spinner } from "./ui";

export default function JobScreen({ jobId }: { jobId: string }) {
  const { backend, demo, setLastResult, refreshSession } = useApp();
  const [job, setJob] = useState<JobResult | null>(null);
  const [raw, setRaw] = useState<unknown>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [needLogin, setNeedLogin] = useState(false);
  const [netFailures, setNetFailures] = useState(0);
  const [paused, setPaused] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const fileName = useRef(loadHistory().find((h) => h.job_id === jobId)?.file_name);

  // Опрос: интервал из ui-config; при сетевых сбоях — экспоненциальная пауза (до pollMaxIntervalMs) без обрыва;
  // 401 → форма входа здесь же (восстановление по job_id в URL); 403/404 → ошибка; done/failed → стоп;
  // скрытая вкладка → пауза до возврата (pausePollingWhenHidden).
  useEffect(() => {
    const cfg = uiConfig();
    let alive = true;
    let timer: number | undefined;
    let failures = 0;
    let waitingVisible = false;
    setError(null);
    setNeedLogin(false);
    const schedule = (ms: number) => {
      timer = window.setTimeout(() => void tick(), ms);
    };
    const tick = async () => {
      if (!alive) return;
      if (cfg.pausePollingWhenHidden && typeof document !== "undefined" && document.hidden) {
        waitingVisible = true;
        setPaused(true);
        return;
      }
      setPaused(false);
      try {
        const r = await backend.getJob(jobId);
        if (!alive) return;
        failures = 0;
        setNetFailures(0);
        setJob(r.job);
        setRaw(r.raw);
        setError(null);
        if (r.job.status === "done" || r.job.status === "failed") {
          const rep = analyzeCompleteness(r.job);
          saveHistoryItem({
            job_id: jobId,
            status: r.job.status,
            verdict: r.job.status === "done" ? r.job.verdict : null,
            n_violations: r.job.violations.length,
            demo: demo || undefined,
            completeness: rep.state,
            mock: rep.mock,
          });
          if (r.job.status === "done") setLastResult(r.job);
          return; // опрос остановлен
        }
        schedule(cfg.pollIntervalMs);
      } catch (e) {
        if (!alive) return;
        const err = e instanceof ApiError ? e : new ApiError(0, "Не удалось получить статус задачи.");
        if (err.status === 401) {
          setNeedLogin(true);
          void refreshSession();
          return;
        }
        if (err.status === 403 || err.status === 404) {
          setError(err);
          return;
        }
        failures++;
        setNetFailures(failures);
        schedule(Math.min(cfg.pollMaxIntervalMs, cfg.pollIntervalMs * 2 ** Math.min(failures, 6)));
      }
    };
    const onVisible = () => {
      if (!document.hidden && waitingVisible && alive) {
        waitingVisible = false;
        void tick();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    void tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [backend, jobId, attempt, demo, setLastResult, refreshSession]);

  if (needLogin) {
    return (
      <div className="mx-auto max-w-md animate-fade-in space-y-3">
        <p className="text-center text-sm text-slate-600 dark:text-slate-300">
          Сессия истекла или не начата. Войдите — проверка задачи <span className="font-mono">{jobId}</span> продолжится.
        </p>
        <LoginCard onDone={() => setAttempt((a) => a + 1)} />
      </div>
    );
  }
  if (error && (!job || job.status !== "done")) {
    return <StateCard title="Не удалось получить результат" text={error.message} onRetry={() => setAttempt((a) => a + 1)} jobId={jobId} />;
  }
  const netBanner =
    netFailures > 0 ? (
      <div className="mx-auto mb-4 max-w-3xl rounded-xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-900 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-200" role="status">
        Нет связи с сервером (попыток: {netFailures}). Продолжаем опрашивать с увеличивающейся паузой — задача на сервере не теряется.
      </div>
    ) : paused ? (
      <div className="mx-auto mb-4 max-w-3xl rounded-xl bg-slate-100 px-4 py-2 text-xs text-slate-600 dark:bg-white/5 dark:text-slate-300" role="status">
        Вкладка неактивна — опрос приостановлен и продолжится при возврате.
      </div>
    ) : null;
  if (!job) {
    return (
      <>
        {netBanner}
        <div className="flex min-h-[40vh] items-center justify-center gap-3 text-slate-500">
          <Spinner className="h-5 w-5" /> Получаем статус задачи…
        </div>
      </>
    );
  }
  const pr = job.partialResult;
  if (
    job.status === "failed" &&
    pr &&
    (pr.violations.length > 0 || pr.timeline.length > 0 || (pr.candidates?.length ?? 0) > 0 || typeof pr.meta.stages === "object")
  ) {
    // A02 отдаёт для UI частичный результат упавшего анализа: показываем найденное, баннер — «Анализ не завершён».
    const shown: JobResult = { ...pr, job_id: job.job_id, status: "failed", error: job.error };
    return <ResultView job={shown} raw={raw} fileName={fileName.current} />;
  }
  if (job.status === "failed" && ((job.candidates?.length ?? 0) > 0 || job.timeline.length > 0 || typeof job.meta.stages === "object")) {
    // A01 UI-report: у «failed» есть meta (стадии, coverage) и кандидаты — показываем подробный отчёт с жёлтым баннером.
    return <ResultView job={job} raw={raw} fileName={fileName.current} />;
  }
  if (job.status === "failed") {
    return (
      <StateCard
        title="Анализ не завершён"
        text={`${job.error ? `Сервер сообщил: «${job.error}». ` : ""}Вывод об отсутствии нарушений сделать нельзя — просмотрите запись вручную или отправьте её повторно.`}
        onRetry={() => setAttempt((a) => a + 1)}
        jobId={job.job_id}
      />
    );
  }
  if (job.status !== "done")
    return (
      <>
        {netBanner}
        <ProcessingView job={job} fileName={fileName.current} />
      </>
    );
  return <ResultView job={job} raw={raw} fileName={fileName.current} />;
}

function StateCard({ title, text, onRetry, jobId }: { title: string; text: string; onRetry: () => void; jobId?: string }) {
  const { demo } = useApp();
  return (
    <div className="mx-auto max-w-xl animate-fade-in">
      <div className="card overflow-hidden">
        <div className="flex items-center gap-3 bg-amber-400 px-6 py-4 text-amber-950">
          <AlertTriangle className="h-6 w-6" />
          <h1 className="text-lg font-extrabold">{title}</h1>
        </div>
        <div className="space-y-4 p-6">
          <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-200">{text}</p>
          {jobId && <p className="font-mono text-xs text-slate-500">задача {jobId}</p>}
          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" onClick={onRetry}>
              <RefreshCw className="h-4 w-4" /> Обновить статус
            </button>
            {jobId && <RetryUpload jobId={jobId} />}
            <a className="btn-primary" href={demo ? "?demo=1#/" : "#/"}>
              <Upload className="h-4 w-4" /> Загрузить видео
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
