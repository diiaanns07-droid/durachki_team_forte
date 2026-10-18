// Пакетная проверка: несколько файлов загружаются по очереди, статусы опрашиваются в фоне.
// Состояние — в модуле (переживает переходы между экранами), React читает его через useSyncExternalStore.
import { ApiError, rememberLocalVideo, saveHistoryItem, type Backend } from "../api/client";
import { analyzeCompleteness, type Completeness } from "./completeness";
import type { Verdict } from "../api/types";
import { validateVideoFile } from "./validate";

export type BatchState = "waiting" | "uploading" | "processing" | "done" | "failed" | "error";

export interface BatchItem {
  key: string;
  name: string;
  size: number;
  state: BatchState;
  /** Доля загрузки 0..1 (XHR). */
  upload: number;
  jobId?: string;
  stage?: string;
  /** Прогресс анализа от сервера, если он его присылает. */
  progress?: number;
  verdict?: Verdict | null;
  /** Полнота анализа: complete | partial | unknown (зелёный — только complete). */
  completeness?: Completeness;
  /** Результат mock/fake-пайплайна. */
  mock?: boolean;
  /** @deprecated совместимость: completeness === "partial". */
  partial?: boolean;
  violations?: { type: string; timecode_sec: number }[];
  error?: string;
}

const POLL_MS = 2000;

export class BatchStore {
  private items: BatchItem[] = [];
  private files = new Map<string, File>();
  private listeners = new Set<() => void>();
  private running = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private seq = 0;

  constructor(
    private backendRef: () => Backend | null,
    private pollMs = POLL_MS,
  ) {}

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getSnapshot = (): BatchItem[] => this.items;

  private set(key: string, patch: Partial<BatchItem>) {
    this.items = this.items.map((it) => (it.key === key ? { ...it, ...patch } : it));
    this.listeners.forEach((l) => l());
  }

  add(files: File[]) {
    const added: BatchItem[] = files.map((f) => {
      const key = `b${++this.seq}`;
      const err = validateVideoFile(f);
      if (!err) this.files.set(key, f);
      return { key, name: f.name, size: f.size, state: err ? "error" : "waiting", upload: 0, error: err ?? undefined };
    });
    this.items = [...this.items, ...added];
    this.listeners.forEach((l) => l());
    void this.runQueue();
  }

  /** Убрать завершённые/ошибочные строки (идущие загрузки и анализы остаются). */
  clearFinished() {
    this.items = this.items.filter((it) => it.state === "waiting" || it.state === "uploading" || it.state === "processing");
    this.listeners.forEach((l) => l());
  }

  busy(): boolean {
    return this.items.some((it) => it.state === "waiting" || it.state === "uploading" || it.state === "processing");
  }

  /** Сводка пакета для экспорта (без кадров): файл, job_id, вердикт, эпизоды. */
  summary() {
    return this.items.map((it) => ({
      file: it.name,
      job_id: it.jobId ?? null,
      status: it.state,
      verdict: it.verdict ?? null,
      completeness: it.completeness ?? null,
      mock: it.mock ?? false,
      violations: it.violations ?? [],
      error: it.error ?? null,
    }));
  }

  private async runQueue() {
    if (this.running) return;
    this.running = true;
    try {
      for (;;) {
        const next = this.items.find((it) => it.state === "waiting");
        const backend = this.backendRef();
        if (!next || !backend) break;
        const file = this.files.get(next.key);
        if (!file) {
          this.set(next.key, { state: "error", error: "Файл недоступен" });
          continue;
        }
        this.set(next.key, { state: "uploading", upload: 0 });
        try {
          const jobId = await backend.upload(file, (p) => this.set(next.key, { upload: p })).promise;
          rememberLocalVideo(jobId, file);
          saveHistoryItem({ job_id: jobId, file_name: file.name, size: file.size, status: "processing" });
          this.set(next.key, { state: "processing", upload: 1, jobId });
          this.schedulePoll();
        } catch (e) {
          this.set(next.key, { state: "error", error: e instanceof ApiError ? e.message : "Не удалось загрузить файл" });
        } finally {
          this.files.delete(next.key);
        }
      }
    } finally {
      this.running = false;
    }
  }

  private schedulePoll() {
    if (this.timer !== null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.pollOnce();
    }, this.pollMs);
  }

  private async pollOnce() {
    const backend = this.backendRef();
    if (!backend) return;
    const active = this.items.filter((it) => it.state === "processing" && it.jobId);
    for (const it of active) {
      try {
        const { job } = await backend.getJob(it.jobId!);
        if (job.status === "done" || job.status === "failed") {
          const rep = analyzeCompleteness(job);
          this.set(it.key, {
            state: job.status,
            verdict: job.status === "done" ? job.verdict : null,
            completeness: rep.state,
            mock: rep.mock,
            partial: rep.state === "partial",
            violations: job.violations.map((v) => ({ type: v.type, timecode_sec: v.timecode_sec })),
            error: job.error,
          });
          saveHistoryItem({
            job_id: it.jobId!,
            status: job.status,
            verdict: job.status === "done" ? job.verdict : null,
            n_violations: job.violations.length,
            completeness: rep.state,
            mock: rep.mock,
          });
        } else {
          this.set(it.key, { stage: job.stage, progress: job.progress });
        }
      } catch (e) {
        if (e instanceof ApiError && [401, 403, 404].includes(e.status)) {
          this.set(it.key, { state: "error", error: e.message });
        }
      }
    }
    if (this.items.some((x) => x.state === "processing")) this.schedulePoll();
  }
}

let currentBackend: Backend | null = null;
export function setBatchBackend(b: Backend) {
  currentBackend = b;
}
/** Общий для приложения пакет (переживает переходы между экранами). */
export const batch = new BatchStore(() => currentBackend);
