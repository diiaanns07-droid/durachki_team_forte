import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, KeyRound, PauseCircle, RefreshCw, ServerCrash, Settings, Upload, WifiOff } from "lucide-react";
import { ApiError, loadHistory, saveHistoryItem } from "../api/client";
import type { JobResult } from "../api/types";
import { useApp } from "../context";
import ProcessingView from "./ProcessingView";
import ResultView from "./ResultView";
import RetryUpload from "./RetryUpload";
import { LoginCard } from "./UploadScreen";
import { uiConfig } from "../lib/uiConfig";
import { analyzeCompleteness } from "../lib/completeness";
import { cx, Spinner } from "./ui";
import s from "./a15.module.css";

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
  // Счётчик сетевых сбоев переживает ручной повтор («Проверить сейчас»): пауза не сбрасывается до первого ответа.
  const failuresRef = useRef(0);

  // Опрос: интервал из ui-config; при сетевых сбоях — экспоненциальная пауза (до pollMaxIntervalMs) без обрыва;
  // 401 → форма входа здесь же (восстановление по job_id в URL); 403/404 → ошибка; done/failed → стоп;
  // скрытая вкладка → пауза до возврата (pausePollingWhenHidden).
  useEffect(() => {
    const cfg = uiConfig();
    let alive = true;
    let timer: number | undefined;
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
        failuresRef.current = 0;
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
        const failures = ++failuresRef.current;
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

  const retryNow = () => setAttempt((a) => a + 1);

  if (needLogin) {
    return (
      <div className="mx-auto max-w-md space-y-4">
        <div className={cx(s.scope, s.enter, s.notice)} role="status">
          <KeyRound className={s.noticeIcon} aria-hidden />
          <div className={s.noticeBody}>
            <div className={s.noticeTitle}>Нужен вход</div>
            Сессия истекла или не начата. Войдите — проверка задачи <span className={s.code}>{jobId}</span> продолжится.
          </div>
        </div>
        <LoginCard onDone={retryNow} />
      </div>
    );
  }
  if (error && (!job || job.status !== "done")) {
    return (
      <StateCard
        tone="error"
        eyebrow="Результат недоступен"
        title="Не удалось получить результат"
        what={error.message}
        meaning="Пока результат не получен, вывода о нарушениях по этой записи нет."
        onRetry={retryNow}
        jobId={jobId}
        access={error.status === 403}
      />
    );
  }
  const netBanner =
    netFailures > 0 ? (
      <div className={cx(s.scope, s.notice, "mx-auto mb-4 max-w-3xl")} data-tone="warning" role="status">
        <WifiOff className={s.noticeIcon} aria-hidden />
        <div className={s.noticeBody}>
          <span className={s.noticeTitle}>Нет связи с сервером</span> <span className="tnum">· попыток: {netFailures}</span>
          <div>Продолжаем опрашивать с увеличивающейся паузой — задача на сервере не теряется.</div>
        </div>
        <button className={cx("btn-ghost px-3 py-1.5 text-xs", s.noticeAction)} onClick={retryNow}>
          <RefreshCw className="h-3.5 w-3.5" aria-hidden /> Проверить сейчас
        </button>
      </div>
    ) : paused ? (
      <div className={cx(s.scope, s.notice, "mx-auto mb-4 max-w-3xl")} role="status">
        <PauseCircle className={s.noticeIcon} aria-hidden />
        <div className={s.noticeBody}>Вкладка неактивна — опрос приостановлен и продолжится при возврате.</div>
      </div>
    ) : null;
  if (!job) {
    return (
      <>
        {netBanner}
        <LoadingCard jobId={jobId} fileName={fileName.current} />
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
        tone="warning"
        eyebrow="Статус анализа"
        title="Анализ не завершён"
        what={job.error ? `Сервер сообщил: «${job.error}».` : "Сервер завершил задачу с ошибкой и не сообщил причину."}
        meaning="Вывод об отсутствии нарушений сделать нельзя — просмотрите запись вручную или отправьте её повторно."
        onRetry={retryNow}
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

/** Пока нет первого ответа сервера: та же шапка, что у экрана обработки, — без скачка компоновки. */
function LoadingCard({ jobId, fileName }: { jobId: string; fileName?: string }) {
  return (
    <div className={cx(s.scope, s.enter, "mx-auto max-w-3xl")}>
      <div className={s.panel}>
        <div className={s.band}>
          <div className={s.bandGrid} aria-hidden />
          <div className={s.bandScan} aria-hidden />
          <div className={s.bandRow}>
            <span className={s.pulse} aria-hidden>
              <span className={s.pulseCore}>
                <Spinner className="h-5 w-5" />
              </span>
            </span>
            <div className="min-w-0">
              <div className={s.bandEyebrow}>Подключение к задаче</div>
              <h1 className={s.bandTitle}>{fileName ?? "Видеозапись"}</h1>
              <div className={s.bandMono}>задача {jobId}</div>
            </div>
          </div>
        </div>
        <div className={s.body}>
          <p className={s.lead} role="status" aria-live="polite">
            Получаем статус задачи…
          </p>
          <div className={s.skeleton} aria-hidden>
            {[0, 1, 2].map((i) => (
              <div key={i} className={s.skeletonRow}>
                <span className={s.skeletonDot} />
                <span className={s.skeletonLines}>
                  <span className={s.skeletonLine} />
                  <span className={s.skeletonLine} />
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Ошибка получения результата или незавершённый анализ: что произошло → что это значит → что можно сделать. */
function StateCard({
  tone,
  eyebrow,
  title,
  what,
  meaning,
  onRetry,
  jobId,
  access,
}: {
  /** error — результат не получен (красный: ошибка); warning — анализ не завершён (янтарный, как во всём UI). */
  tone: "error" | "warning";
  eyebrow: string;
  title: string;
  what: ReactNode;
  meaning: ReactNode;
  onRetry: () => void;
  jobId?: string;
  /** 403: дело в доступе — главное действие «Настройки доступа»; повторная отправка файла не поможет. */
  access?: boolean;
}) {
  const { demo, openSettings } = useApp();
  const titleId = useId();
  const Icon = tone === "error" ? ServerCrash : AlertTriangle;
  return (
    <div className={cx(s.scope, s.enter, "mx-auto max-w-2xl")}>
      <section className={cx(s.panel, s.state)} data-tone={tone} aria-labelledby={titleId}>
        <div className={s.stateHead}>
          <div className={s.grid} aria-hidden />
          <span className={s.stateIcon}>
            <Icon className="h-6 w-6" aria-hidden />
          </span>
          <div className={s.stateText}>
            <div className={s.eyebrow}>{eyebrow}</div>
            <h1 id={titleId} className={s.stateTitle}>
              {title}
            </h1>
          </div>
        </div>
        <dl className={s.facts}>
          <div className={s.fact}>
            <dt>Что произошло</dt>
            <dd>{what}</dd>
          </div>
          <div className={s.fact}>
            <dt>Что это значит</dt>
            <dd>{meaning}</dd>
          </div>
          <div className={s.fact}>
            <dt>Что можно сделать</dt>
            <dd>
              <div className={s.actionsStack}>
                {jobId && !access && <RetryUpload jobId={jobId} className="btn-primary" />}
                <div className={s.actions}>
                  {access && (
                    <button className="btn-primary" onClick={openSettings}>
                      <Settings className="h-4 w-4" aria-hidden /> Настройки доступа
                    </button>
                  )}
                  <button className="btn-ghost" onClick={onRetry}>
                    <RefreshCw className="h-4 w-4" aria-hidden /> Обновить статус
                  </button>
                  <a className="btn-ghost" href={demo ? "?demo=1#/" : "#/"}>
                    <Upload className="h-4 w-4" aria-hidden /> Загрузить видео
                  </a>
                </div>
              </div>
            </dd>
          </div>
        </dl>
        {jobId && (
          <div className={s.stateFoot}>
            <span className={s.mono}>задача {jobId}</span>
          </div>
        )}
      </section>
    </div>
  );
}
