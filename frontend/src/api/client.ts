// HTTP-клиент к бэкенду (A02). Все пути — в ENDPOINTS, чтобы интегратор правил в одном месте.
//
// Авторизация UI: GET /ui/v1/session → токен → заголовок X-UI-Token для /ui/v1/*.
// Запасной путь: API-ключ из настроек → X-API-Key для /api/v1/*.
import { normalizeJob } from "./normalize";
import { parseObservations, type Observations } from "./observations";
import type { AskResponse, JobResult, SessionInfo } from "./types";

export const ENDPOINTS = {
  session: "/ui/v1/session",
  uiAnalyze: "/ui/v1/analyze",
  uiJobs: "/ui/v1/jobs",
  apiJobs: "/api/v1/jobs",
  uiJob: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}`,
  uiAsk: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}/ask`,
  /** A14 (через A02): статус Copilot и текстовое заключение (Markdown, работает без LLM). */
  uiCopilotStatus: "/ui/v1/copilot/status",
  apiCopilotStatus: "/api/v1/copilot/status",
  uiReport: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}/report`,
  apiReport: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/report`,
  apiAnalyze: "/api/v1/analyze",
  apiJob: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}`,
  apiAsk: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/ask`,
  /** Видео для плеера: защищённый cookie UI-маршрут с поддержкой Range. */
  uiVideo: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}/video`,
  /** Запасной путь (только с API-ключом в памяти вкладки → blob). */
  apiVideo: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/video`,
  /** Наблюдения правил ml.seqrules (профиль pose_rules): скелеты, предметы, зоны — расширение контракта. */
  uiObservations: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}/observations`,
  apiObservations: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/observations`,
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class CopilotUnavailable extends Error {
  constructor() {
    super("Copilot недоступен на этом сервере");
    this.name = "CopilotUnavailable";
  }
}

// ---------- настройки (localStorage; всё в try/catch — приватный режим) ----------
const LS = {
  get(k: string): string | null {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string | null) {
    try {
      if (v === null || v === "") localStorage.removeItem(k);
      else localStorage.setItem(k, v);
    } catch {
      /* noop */
    }
  },
};

// API-ключ живёт только в памяти вкладки: не в localStorage, не в URL, не в бандле.
let memoryApiKey = "";

export const settings = {
  get apiKey(): string {
    return memoryApiKey;
  },
  set apiKey(v: string) {
    memoryApiKey = v.trim();
  },
  get apiBase(): string {
    const env = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE ?? "";
    return (LS.get("ks.apiBase") ?? env).replace(/\/+$/, "");
  },
  set apiBase(v: string) {
    LS.set("ks.apiBase", v.trim().replace(/\/+$/, ""));
  },
};

export function messageForStatus(status: number, fallback = ""): string {
  if (status === 401 || status === 403) return "Нет доступа: проверьте API-ключ в настройках или обновите страницу.";
  if (status === 404) return "Задача не найдена на сервере (возможно, сервер перезапускался).";
  if (status === 413) return "Файл слишком большой для сервера.";
  if (status === 415 || status === 422) return fallback || "Сервер не принял файл: нужен видеофайл .mov или .mp4.";
  if (status === 429) return "Слишком много запросов. Повторите через минуту.";
  if (status === 503) return fallback || "Сервер перегружен: очередь задач заполнена или модели ещё загружаются. Повторите позже.";
  if (status === 507) return "На сервере недостаточно места для загрузки.";
  if (status >= 500) return fallback || "Ошибка сервера при обработке запроса.";
  if (status === 0) return "Нет связи с сервером.";
  return fallback || `Ошибка запроса (HTTP ${status}).`;
}

async function errorText(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: unknown; error?: unknown; message?: unknown };
    const d = j.detail ?? j.error ?? j.message;
    if (typeof d === "string") return d;
  } catch {
    /* noop */
  }
  return "";
}

export interface UploadHandle {
  promise: Promise<string>;
  abort: () => void;
}

export interface Backend {
  kind: "http" | "demo";
  /** Проверка сессии (HttpOnly cookie). Публичной выдачи сессии без ключа нет. */
  init(): Promise<SessionInfo>;
  /** Вход проверяющего: ключ/секрет → сервер ставит HttpOnly cookie. Ключ не сохраняется. */
  login(key: string): Promise<SessionInfo>;
  logout(): Promise<void>;
  upload(file: File, onProgress: (fraction: number) => void): UploadHandle;
  getJob(id: string): Promise<{ job: JobResult; raw: unknown }>;
  listJobs?(limit?: number): Promise<HistoryItem[]>;
  /** null — сервер не сообщает статус (тогда ориентируемся на 501 от /ask). */
  copilotStatus?(): Promise<{ available: boolean; disclaimer?: string } | null>;
  /** Текстовое заключение (Markdown) от A14; бросает ApiError, если недоступно. */
  report?(id: string): Promise<string>;
  ask(id: string, question: string, history: { role: "user" | "assistant"; text: string }[]): Promise<AskResponse>;
  /** URL исходного видео для <video>; fallbacks — запасные источники (blob с X-API-Key и т. п.). */
  videoSource(id: string): { url: string; fallbacks?: (() => Promise<string>)[] } | null;
  /** Наблюдения правил для оверлеев; null — сервер их не отдаёт (профиль без seqrules). */
  observations?(id: string): Promise<Observations | null>;
}

function sessionFromJson(info: Record<string, unknown>, authenticated: boolean): SessionInfo {
  const f = info.features;
  const features: SessionInfo["features"] = typeof f === "object" && f !== null ? { ...(f as SessionInfo["features"]) } : {};
  if (typeof info.copilot === "boolean") features.copilot = info.copilot;
  if (typeof info.password_login === "boolean") features.password_login = info.password_login;
  const auth = typeof info.authenticated === "boolean" ? info.authenticated : authenticated;
  return {
    token: null,
    reachable: true,
    authenticated: auth,
    features,
    version: typeof info.version === "string" ? info.version : undefined,
  };
}

async function readJson(res: Response): Promise<Record<string, unknown>> {
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("json")) return {};
  try {
    const j = (await res.json()) as unknown;
    return typeof j === "object" && j !== null ? (j as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

export class HttpBackend implements Backend {
  kind = "http" as const;
  /** Необязательный токен из тела ответа сессии (если A02 его отдаёт) — только в памяти. */
  private token: string | null = null;
  /** CSRF-токен из GET/POST /ui/v1/session (A02) — заголовок X-CSRF-Token для POST в /ui/v1/*. */
  private csrf: string | null = null;
  private features: SessionInfo["features"] = {};
  private uiAvailable = true;

  private url(path: string): string {
    return `${settings.apiBase}${path}`;
  }

  private headers(path: string, extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json", ...extra };
    if (this.token && path.startsWith("/ui/")) h["X-UI-Token"] = this.token;
    if (this.csrf && path.startsWith("/ui/")) h["X-CSRF-Token"] = this.csrf;
    if (path.startsWith("/api/") && settings.apiKey) h["X-API-Key"] = settings.apiKey;
    return h;
  }

  async init(): Promise<SessionInfo> {
    try {
      const res = await fetch(this.url(ENDPOINTS.session), {
        credentials: "same-origin",
        headers: this.headers(ENDPOINTS.session),
      });
      if (res.ok) {
        const info = await readJson(res);
        this.csrf = typeof info.csrf_token === "string" ? info.csrf_token : this.csrf;
        const s = sessionFromJson(info, true);
        this.features = { ...this.features, ...s.features };
        this.uiAvailable = true;
        return { ...s, features: this.features };
      }
      if (res.status === 401 || res.status === 403) {
        this.uiAvailable = true;
        this.csrf = null;
        const s = sessionFromJson(await readJson(res), false);
        // Сессии нет; работать можно только с API-ключом в памяти вкладки (/api/v1/*).
        return { ...s, authenticated: !!settings.apiKey };
      }
      // 404/405 — UI-маршруты выключены (ui_enabled: false): работаем только через API-ключ.
      // 429/5xx — временная проблема: UI-маршруты не отключаем.
      if (res.status === 404 || res.status === 405) this.uiAvailable = false;
      return { token: null, reachable: true, authenticated: !!settings.apiKey, features: {} };
    } catch {
      return { token: null, reachable: false, authenticated: false, features: {} };
    }
  }

  async login(key: string): Promise<SessionInfo> {
    const k = key.trim();
    if (!k) throw new ApiError(401, "Введите ключ доступа.");
    let res: Response;
    try {
      res = await fetch(this.url(ENDPOINTS.session), {
        method: "POST",
        credentials: "same-origin",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        // A02: вход по API-ключу ИЛИ паролю UI — одно поле формы, сервер проверяет оба варианта.
        body: JSON.stringify({ api_key: k, password: k }),
      });
    } catch {
      throw new ApiError(0, messageForStatus(0));
    }
    if (res.status === 404 || res.status === 405) {
      // UI-сессий нет — запасной путь: ключ в памяти вкладки для /api/v1/*. Сначала проверяем ключ.
      let check: Response;
      try {
        check = await fetch(this.url(`${ENDPOINTS.apiJobs}?limit=1`), {
          headers: { Accept: "application/json", "X-API-Key": k },
          credentials: "same-origin",
        });
      } catch {
        throw new ApiError(0, messageForStatus(0));
      }
      if (check.status === 401 || check.status === 403) throw new ApiError(401, "Неверный ключ доступа.");
      settings.apiKey = k;
      this.uiAvailable = false;
      return { token: null, reachable: true, authenticated: true, features: {} };
    }
    if (!res.ok) {
      if (res.status === 401) throw new ApiError(res.status, "Неверный ключ доступа.");
      if (res.status === 403) throw new ApiError(res.status, "Сервер отклонил вход (запрос не с того же адреса).");
      throw new ApiError(res.status, messageForStatus(res.status, await errorText(res)));
    }
    const info = await readJson(res);
    this.csrf = typeof info.csrf_token === "string" ? info.csrf_token : null;
    const t = info.token ?? info.ui_token;
    this.token = typeof t === "string" ? t : null;
    const s = sessionFromJson(info, true);
    this.features = s.features;
    this.uiAvailable = true;
    return s;
  }

  async logout(): Promise<void> {
    this.token = null;
    this.csrf = null;
    settings.apiKey = "";
    try {
      await fetch(this.url(ENDPOINTS.session), { method: "DELETE", credentials: "same-origin" });
    } catch {
      /* noop */
    }
  }

  /** /ui/v1/* (cookie); если UI-маршруты выключены или не реализованы — /api/v1/* с ключом из памяти. */
  private async fetchWithFallback(uiPath: string, apiPath: string, init: RequestInit = {}): Promise<Response> {
    const doFetch = (p: string) =>
      fetch(this.url(p), {
        ...init,
        credentials: "same-origin",
        headers: this.headers(p, (init.headers as Record<string, string>) ?? {}),
      });
    if (this.uiAvailable || !settings.apiKey) {
      let res = await doFetch(uiPath);
      if (res.status === 403 && init.method && init.method !== "GET") {
        await this.init(); // CSRF-токен мог устареть (перезапуск сервера) — обновляем и повторяем
        res = await doFetch(uiPath);
      }
      const tryApi = (res.status === 404 || res.status === 405 || res.status === 401) && !!settings.apiKey;
      if (!tryApi) return res;
    }
    return doFetch(apiPath);
  }

  upload(file: File, onProgress: (fraction: number) => void): UploadHandle {
    let xhr: XMLHttpRequest | null = null;
    let aborted = false;
    const send = (path: string): Promise<{ status: number; body: string }> =>
      new Promise((resolve, reject) => {
        xhr = new XMLHttpRequest();
        xhr.open("POST", this.url(path));
        xhr.withCredentials = true;
        for (const [k, v] of Object.entries(this.headers(path))) xhr.setRequestHeader(k, v);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) onProgress(Math.min(1, e.loaded / e.total));
        };
        xhr.onload = () => resolve({ status: xhr!.status, body: xhr!.responseText });
        xhr.onerror = () => reject(new ApiError(0, messageForStatus(0)));
        xhr.onabort = () => reject(new DOMException("Загрузка отменена", "AbortError"));
        const fd = new FormData();
        fd.append("file", file, file.name);
        xhr.send(fd);
      });
    const promise = (async () => {
      let r: { status: number; body: string };
      if (this.uiAvailable || !settings.apiKey) {
        r = await send(ENDPOINTS.uiAnalyze);
        if (r.status === 403 && !aborted) {
          await this.init();
          onProgress(0);
          r = await send(ENDPOINTS.uiAnalyze);
        }
        if ((r.status === 404 || r.status === 405 || r.status === 401) && settings.apiKey && !aborted) {
          onProgress(0);
          r = await send(ENDPOINTS.apiAnalyze);
        }
      } else {
        r = await send(ENDPOINTS.apiAnalyze);
      }
      if (r.status < 200 || r.status >= 300) {
        let detail = "";
        try {
          const j = JSON.parse(r.body) as { detail?: unknown };
          if (typeof j.detail === "string") detail = j.detail;
        } catch {
          /* noop */
        }
        throw new ApiError(r.status, messageForStatus(r.status, detail));
      }
      let j: { job_id?: unknown; id?: unknown } = {};
      try {
        j = JSON.parse(r.body) as typeof j;
      } catch {
        /* noop */
      }
      const id = j.job_id ?? j.id;
      if (typeof id !== "string" && typeof id !== "number") throw new ApiError(500, "Сервер не вернул job_id");
      onProgress(1);
      return String(id);
    })();
    return {
      promise,
      abort: () => {
        aborted = true;
        (xhr as XMLHttpRequest | null)?.abort();
      },
    };
  }

  /** GET /ui/v1/jobs — последние задачи на сервере (A02). Ошибки → пустой список. */
  async listJobs(limit = 20): Promise<HistoryItem[]> {
    try {
      const res = await this.fetchWithFallback(`${ENDPOINTS.uiJobs}?limit=${limit}`, `${ENDPOINTS.apiJobs}?limit=${limit}`);
      if (!res.ok) return [];
      const j = (await res.json()) as { jobs?: unknown };
      if (!Array.isArray(j.jobs)) return [];
      const out: HistoryItem[] = [];
      for (const x of j.jobs as Record<string, unknown>[]) {
        if (!x || (typeof x.job_id !== "string" && typeof x.job_id !== "number")) continue;
        const created = typeof x.created_at === "number" ? new Date(x.created_at < 1e12 ? x.created_at * 1000 : x.created_at) : new Date(String(x.created_at ?? ""));
        out.push({
          job_id: String(x.job_id),
          file_name: typeof x.filename === "string" && x.filename ? x.filename : "видео",
          created_at: Number.isNaN(created.getTime()) ? new Date().toISOString() : created.toISOString(),
          status: typeof x.status === "string" ? x.status : undefined,
        });
      }
      return out;
    } catch {
      return [];
    }
  }

  async copilotStatus(): Promise<{ available: boolean; disclaimer?: string } | null> {
    try {
      const res = await this.fetchWithFallback(ENDPOINTS.uiCopilotStatus, ENDPOINTS.apiCopilotStatus);
      if (!res.ok) return null;
      const j = await readJson(res);
      const available = j.available !== false && j.enabled !== false;
      if (!available) this.features.copilot = false;
      return { available, disclaimer: typeof j.disclaimer === "string" ? j.disclaimer : undefined };
    } catch {
      return null;
    }
  }

  async report(id: string): Promise<string> {
    const res = await this.fetchWithFallback(ENDPOINTS.uiReport(id), ENDPOINTS.apiReport(id));
    if (!res.ok) throw new ApiError(res.status, res.status === 404 ? "Заключение недоступно на этом сервере." : messageForStatus(res.status));
    return res.text();
  }

  async observations(id: string): Promise<Observations | null> {
    try {
      const res = await this.fetchWithFallback(ENDPOINTS.uiObservations(id), ENDPOINTS.apiObservations(id));
      if (!res.ok) return null;
      return parseObservations(await res.json());
    } catch {
      return null;
    }
  }

  async getJob(id: string): Promise<{ job: JobResult; raw: unknown }> {
    let res: Response;
    try {
      res = await this.fetchWithFallback(ENDPOINTS.uiJob(id), ENDPOINTS.apiJob(id));
    } catch {
      throw new ApiError(0, messageForStatus(0));
    }
    if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status, await errorText(res)));
    const raw: unknown = await res.json();
    return { job: normalizeJob(raw, id), raw };
  }

  async ask(
    id: string,
    question: string,
    history: { role: "user" | "assistant"; text: string }[],
  ): Promise<AskResponse> {
    if (this.features.copilot === false) throw new CopilotUnavailable();
    const res = await this.fetchWithFallback(ENDPOINTS.uiAsk(id), ENDPOINTS.apiAsk(id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (res.status === 501) {
      this.features.copilot = false;
      throw new CopilotUnavailable();
    }
    if (res.status === 503) throw new ApiError(503, "Copilot занят (идёт анализ видео или другой запрос). Повторите чуть позже.");
    if (res.status === 504) throw new ApiError(504, "Copilot не успел ответить (таймаут). Попробуйте задать вопрос короче.");
    if (res.status === 409) throw new ApiError(409, "Анализ ещё не завершён — Copilot ответит после формирования отчёта.");
    if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status, await errorText(res)));
    const j = (await res.json()) as Record<string, unknown>;
    const answer = j.answer ?? j.text ?? j.response ?? j.message;
    const sources = Array.isArray(j.sources) ? j.sources.map((s) => (typeof s === "string" ? s : JSON.stringify(s))) : [];
    return { answer: typeof answer === "string" ? answer : JSON.stringify(j), sources };
  }

  videoSource(id: string): { url: string; fallbacks?: (() => Promise<string>)[] } {
    // Основной путь — cookie UI-маршрут с Range: <video> шлёт cookie сам, ключ в URL не попадает.
    // A02 отдаёт по нему preview.mp4 (если пайплайн перекодировал исходник), иначе оригинал.
    const apiPath = ENDPOINTS.apiVideo(id);
    return {
      url: this.url(ENDPOINTS.uiVideo(id)),
      fallbacks: settings.apiKey
        ? [
            async () => {
              const res = await fetch(this.url(apiPath), { headers: this.headers(apiPath), credentials: "same-origin" });
              if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status));
              return URL.createObjectURL(await res.blob());
            },
          ]
        : [],
    };
  }
}

// ---------- локальные видео (только что загруженные файлы: играем без сервера) ----------
const localVideos = new Map<string, string>();
const localFiles = new Map<string, File>();
export function rememberLocalVideo(jobId: string, file: File) {
  localFiles.set(jobId, file);
  try {
    localVideos.set(jobId, URL.createObjectURL(file));
  } catch {
    /* SSR/тесты */
  }
}
/** Файл задачи, если он ещё в памяти вкладки (для повторной отправки после ошибки). */
export function localFile(jobId: string): File | undefined {
  return localFiles.get(jobId);
}
export function localVideo(jobId: string): string | undefined {
  return localVideos.get(jobId);
}

// ---------- история проверок ----------
export interface HistoryItem {
  job_id: string;
  file_name: string;
  size?: number;
  created_at: string;
  verdict?: "violation" | "no_violation" | null;
  n_violations?: number;
  status?: string;
  demo?: boolean;
  /** Полнота анализа на момент завершения: зелёный в истории — только "complete" и не mock. */
  completeness?: "complete" | "partial" | "unknown";
  mock?: boolean;
}

const HKEY = "ks.history";
export function loadHistory(): HistoryItem[] {
  try {
    const v = JSON.parse(LS.get(HKEY) ?? "[]") as unknown;
    return Array.isArray(v) ? (v as HistoryItem[]).filter((x) => x && typeof x.job_id === "string") : [];
  } catch {
    return [];
  }
}
export function saveHistoryItem(item: Partial<HistoryItem> & { job_id: string }) {
  const list = loadHistory();
  const i = list.findIndex((x) => x.job_id === item.job_id);
  const merged: HistoryItem = {
    file_name: "видео",
    created_at: new Date().toISOString(),
    ...(i >= 0 ? list[i] : {}),
    ...item,
  } as HistoryItem;
  if (i >= 0) list.splice(i, 1);
  list.unshift(merged);
  LS.set(HKEY, JSON.stringify(list.slice(0, 20)));
}
export function clearHistory() {
  LS.set(HKEY, null);
}
