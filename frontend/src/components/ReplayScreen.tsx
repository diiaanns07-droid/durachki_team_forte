import { useState } from "react";
import { FileJson, FileVideo, History } from "lucide-react";
import { rememberLocalVideo } from "../api/client";
import { normalizeJob } from "../api/normalize";
import type { JobResult } from "../api/types";
import ResultView from "./ResultView";

/**
 * Повторный просмотр СОХРАНЁННОГО отчёта (JSON из CLI A03 / «Полный отчёт JSON») + необязательное локальное видео.
 * Всё остаётся в браузере; к серверу запросов нет. Всегда помечено: это не новый анализ и не подмена ошибки.
 */
export default function ReplayScreen() {
  const [state, setState] = useState<{ job: JobResult; raw: unknown; name: string } | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadJson = async (f: File | undefined) => {
    if (!f) return;
    setError(null);
    try {
      const raw: unknown = JSON.parse(await f.text());
      const job = normalizeJob(raw, `replay-${f.name}`);
      if (video) rememberLocalVideo(job.job_id, video);
      setState({ job, raw, name: f.name });
    } catch (e) {
      setError(e instanceof Error ? `Не удалось прочитать отчёт: ${e.message}` : "Не удалось прочитать отчёт");
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-xl border-2 border-violet-300 bg-violet-50 px-4 py-3 text-sm text-violet-900 dark:border-violet-400/40 dark:bg-violet-400/10 dark:text-violet-200" role="note">
        <b className="uppercase tracking-wider">Повторный просмотр сохранённого отчёта.</b> Это не новый анализ: показывается содержимое
        выбранного JSON-файла{state ? ` «${state.name}»` : ""}. Данные не отправляются на сервер.
      </div>
      {!state && (
        <section className="card grid gap-4 p-6 sm:grid-cols-2">
          <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 p-6 text-center hover:border-brand-400 dark:border-white/15">
            <FileJson className="h-7 w-7 text-brand-600 dark:text-brand-300" />
            <span className="text-sm font-semibold">Отчёт JSON</span>
            <span className="text-xs text-slate-500">CLI `backend.pipeline run --out` или «Полный отчёт JSON»</span>
            <input type="file" accept=".json,application/json" className="sr-only" onChange={(e) => void loadJson(e.target.files?.[0])} aria-label="Отчёт JSON" />
          </label>
          <label className="flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 p-6 text-center hover:border-brand-400 dark:border-white/15">
            <FileVideo className="h-7 w-7 text-slate-500" />
            <span className="text-sm font-semibold">Видео (необязательно)</span>
            <span className="text-xs text-slate-500">{video ? video.name : "тот же ролик — для перехода по таймкодам"}</span>
            <input type="file" accept=".mov,.mp4,video/*" className="sr-only" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} aria-label="Видео для отчёта" />
          </label>
          {error && <div className="text-sm text-rose-600 dark:text-rose-400 sm:col-span-2">{error}</div>}
        </section>
      )}
      {state && <ResultView job={state.job} raw={state.raw} fileName={state.name} replay />}
      {!state && (
        <p className="flex items-center gap-2 text-xs text-slate-500">
          <History className="h-3.5 w-3.5" /> Сначала выберите видео (если нужно), затем JSON — видео привяжется к отчёту.
        </p>
      )}
    </div>
  );
}
