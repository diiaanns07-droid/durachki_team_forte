// U11: «Полнота анализа» и повторная отправка — SSR-рендер на синтетических данных (без браузера и бэкенда).
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { DemoBackend } from "../src/api/demo";
import { rememberLocalVideo, saveHistoryItem } from "../src/api/client";
import type { JobResult } from "../src/api/types";
import { AppContext, type AppContextValue } from "../src/context";
import { analyzeCompleteness } from "../src/lib/completeness";
import CompletenessPanel from "../src/components/CompletenessPanel";
import RetryUpload from "../src/components/RetryUpload";

const LONG_ERROR =
  "Traceback (most recent call last):\n" +
  '  File "ml/detect/backend.py", line 412, in detect\n' +
  "    out = self.model(batch)\n" +
  "torch.OutOfMemoryError: CUDA out of memory. ".repeat(40) +
  "/very/long/path/without/spaces/" + "x".repeat(400);

function job(over: Partial<JobResult>): JobResult {
  return { job_id: "u11-synth", status: "done", verdict: "no_violation", violations: [], timeline: [], meta: {}, ...over };
}

function panel(j: JobResult, extra: { violationsFound?: number; failed?: boolean } = {}): string {
  return renderToString(<CompletenessPanel report={analyzeCompleteness(j)} {...extra} />);
}

const text = (html: string) => html.replace(/<!-- -->/g, "").replace(/<[^>]+>/g, " ").replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ");

describe("CompletenessPanel (U11)", () => {
  it("полный анализ: проверено целиком, «нет нарушений» — результат, а не нехватка данных", () => {
    const html = panel(
      job({ meta: { complete: true, coverage: 1, required_stages: ["ingest", "detect"], stages: { ingest: { status: "ok", seconds: 1.2 }, detect: { status: "ok", seconds: 8.4 } } } }),
      { violationsFound: 0 },
    );
    const t = text(html);
    expect(t).toContain("Полнота анализа");
    expect(t).toContain("Запись проверена целиком");
    expect(t).toContain("100 %");
    expect(html).toContain('role="meter"');
    expect(t).toContain("это результат проверки всей записи, а не нехватка данных");
    expect(t).not.toContain("Что можно сделать");
    expect(t).toContain("Обязательные стадии: 2 из 2 выполнены");
  });

  it("неполный анализ: границы результата, понятные причины, длинная ошибка в раскрываемом блоке", () => {
    const html = panel(
      job({
        meta: {
          coverage: 0.62,
          deadline_exceeded_s: 14,
          required_stages: ["ingest", "detect", "fusion"],
          stages: {
            ingest: { status: "ok", seconds: 2.1 },
            detect: { status: "timeout", error: LONG_ERROR },
            fusion: { status: "pending" },
            vlm_verify: { status: "skipped" },
          },
        },
      }),
      { violationsFound: 0 },
    );
    const t = text(html);
    expect(t).toContain("Проверена только часть записи");
    expect(t).toContain("62 %");
    expect(t).toContain("Что произошло");
    expect(t).toContain("Превышен бюджет времени анализа");
    expect(t).toContain("Нарушений не найдено только в проверенной части");
    expect(t).toContain("не выполнены: Поиск людей и объектов — таймаут; Формирование отчёта — не запускалась");
    expect(t).toContain("Просмотрите запись вручную");
    expect(t).not.toContain("Отправьте тот же файл повторно"); // done: сервер вернул бы ту же задачу (дедуп)
    expect(html).toContain("<pre"); // текст ошибки целиком, с переносом
    expect(html).toContain("CUDA out of memory");
    expect(html).toMatch(/<details[^>]*open/); // обязательные стадии упали — список раскрыт
  });

  it("ошибка задачи: предлагается повторная отправка (новая задача)", () => {
    const t = text(panel(job({ status: "failed", verdict: null, error: "timeout", meta: {} })));
    expect(t).toContain("Отправьте тот же файл повторно");
  });

  it("неизвестная полнота: не выглядит как 100 %, технический хвост — в подробностях", () => {
    const html = panel(job({ meta: {} }), { violationsFound: 0 });
    const t = text(html);
    expect(t).toContain("Полнота не подтверждена");
    expect(t).toContain("неизвестно");
    expect(t).toContain("это не «100 %»");
    expect(html).not.toContain('role="meter"');
    expect(t).not.toMatch(/Проверено записи\s+100 %/); // неизвестное ≠ 100 %
    expect(t).toContain("Нарушений не найдено, но данных недостаточно");
    expect(t).toContain("Сведений о стадиях анализа сервер не прислал");
    // основная формулировка без «meta.», полная — в «Технических подробностях»
    const main = t.split("Технические подробности")[0];
    expect(main).toContain("Сервер не подтвердил полноту анализа");
    expect(main).not.toContain("meta.complete");
    expect(t).toContain("Технические подробности (для администратора)");
    expect(t).toContain("нет meta.complete / meta.coverage");
  });

  it("A01: полнота не подтверждена при известной доле; жаргон заменён", () => {
    const t = text(
      panel(job({ analysisComplete: false, coverageProblem: "оркестратор сообщил незавершённый анализ (meta.complete=false)", meta: { coverage: 1 } })),
    );
    const main = t.split("Технические подробности")[0];
    expect(main).toContain("Сервер анализа сообщил незавершённый анализ");
    expect(main).not.toContain("оркестратор");
    expect(main).not.toContain("meta.complete");
    expect(t).toContain("(meta.complete=false)"); // полный текст сервера — в подробностях
  });

  it("mock: модели не запускались — заметно", () => {
    const t = text(panel(job({ meta: { mock: true, complete: true, coverage: 1 } })));
    expect(t).toContain("Тестовый результат: модели не запускались");
    expect(t).toContain("ничего не говорят о записи");
  });
});

// ---------------------------------------------------------------- RetryUpload

const store = new Map<string, string>();
beforeEach(() => {
  store.clear();
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
  };
});
afterEach(() => {
  delete (globalThis as { localStorage?: unknown }).localStorage;
});

function ctx(kind: "http" | "demo"): AppContextValue {
  const backend = new DemoBackend() as unknown as AppContextValue["backend"];
  (backend as { kind: string }).kind = kind;
  return {
    backend,
    demo: kind === "demo",
    session: { token: null, reachable: true, authenticated: true, features: {} },
    serverOk: true,
    authenticated: true,
    login: async () => undefined,
    logout: async () => undefined,
    copilotEnabled: false,
    disableCopilot: () => undefined,
    refreshSession: async () => undefined,
    toast: () => undefined,
    openSettings: () => undefined,
    lastResult: null,
    setLastResult: () => undefined,
  };
}
const wrap = (node: ReactNode, kind: "http" | "demo" = "http") =>
  renderToString(<AppContext.Provider value={ctx(kind)}>{node}</AppContext.Provider>);

describe("RetryUpload (U11)", () => {
  it("файл в памяти вкладки: одна кнопка и какой файл уйдёт", () => {
    rememberLocalVideo("j-mem", new File([new Uint8Array(2048)], "synthetic_cam1.mov", { type: "video/quicktime" }));
    const t = text(wrap(<RetryUpload jobId="j-mem" />));
    expect(t).toContain("Отправить этот файл повторно");
    expect(t).toContain("«synthetic_cam1.mov»");
    expect(t).toContain("2,0 КБ");
    expect(t).toContain("Будет создана новая задача");
  });

  it("после перезагрузки: объясняет, какой файл выбрать (имя и размер из истории)", () => {
    saveHistoryItem({ job_id: "j-hist", file_name: "synthetic_cam2.mov", size: 29_800_000 });
    const t = text(wrap(<RetryUpload jobId="j-hist" />));
    expect(t).toContain("Выбрать файл для повторной отправки");
    expect(t).toContain("Нужен тот же исходный файл: «synthetic_cam2.mov» · 28,4 МБ");
    expect(t).toContain("выберите его на диске");
  });

  it("нет ни файла, ни истории: общая подсказка без выдуманного имени", () => {
    const t = text(wrap(<RetryUpload jobId="j-none" />));
    expect(t).toContain("Нужен исходный видеофайл этой записи (.mov или .mp4)");
    expect(t).not.toContain("«");
  });

  it("демо: повторная отправка не показывается (реального файла нет)", () => {
    expect(wrap(<RetryUpload jobId="j-demo" />, "demo")).toBe("");
  });
});
