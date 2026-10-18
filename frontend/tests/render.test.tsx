// SSR smoke-тесты экранов: рендер без браузера и без бэкенда (ловит падения при отрисовке).
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { DemoBackend, buildDemoClean, buildDemoResult } from "../src/api/demo";
import { normalizeJob } from "../src/api/normalize";
import type { JobResult } from "../src/api/types";
import { AppContext, type AppContextValue } from "../src/context";
import ResultView from "../src/components/ResultView";
import ProcessingView from "../src/components/ProcessingView";
import AboutScreen from "../src/components/AboutScreen";

function ctx(over: Partial<AppContextValue> = {}): AppContextValue {
  return {
    backend: new DemoBackend(),
    demo: false,
    session: { token: null, reachable: true, authenticated: true, features: {} },
    serverOk: true,
    authenticated: true,
    login: async () => undefined,
    logout: async () => undefined,
    copilotEnabled: true,
    disableCopilot: () => undefined,
    refreshSession: async () => undefined,
    toast: () => undefined,
    openSettings: () => undefined,
    lastResult: null,
    setLastResult: () => undefined,
    ...over,
  };
}

const wrap = (node: ReactNode, over?: Partial<AppContextValue>) =>
  renderToString(<AppContext.Provider value={ctx(over)}>{node}</AppContext.Provider>);

describe("ResultView", () => {
  it("нарушения: баннер ТЗ, таймкоды, нормы, «что не проверяет»", () => {
    const html = wrap(<ResultView job={buildDemoResult()} raw={{}} />, { demo: true });
    expect(html).toContain("Обнаружены признаки возможных нарушений");
    expect(html).toContain("00:24.6");
    expect(html).toContain("00:47.8");
    expect(html).toContain("цитата не сверена");
    expect(html).toContain("без пункта");
    expect(html).not.toContain("цитата сверена"); // в демо нет сверенных норм — бейдж не показывается
    expect(html).toContain("Что система не проверяет");
    expect(html).toContain("Как система пришла к выводу");
  });
  it("без нарушений — формулировка ТЗ; пограничные моменты отдельно и не меняют вердикт", () => {
    const html = wrap(<ResultView job={buildDemoClean()} raw={{}} />);
    expect(html).toContain("Нарушение не обнаружено");
    expect(html).toContain("Стоит просмотреть");
    expect(html).toContain("00:37.2");
    expect(html).not.toContain("Обнаружены признаки возможных нарушений");
  });
  it("неполный анализ без нарушений — не зелёный «не обнаружено»", () => {
    const job: JobResult = { ...buildDemoClean(), meta: { partial: true } };
    const html = wrap(<ResultView job={job} raw={{}} />);
    expect(html).toContain("Анализ неполный");
    expect(html).not.toContain("Нарушение не обнаружено");
  });
  it("failed с частичным результатом — «Анализ не завершён»", () => {
    const job: JobResult = { ...buildDemoResult(), status: "failed", error: "timeout" };
    const html = wrap(<ResultView job={job} raw={{}} />);
    expect(html).toContain("Анализ не завершён");
    expect(html).toContain("Найдено до сбоя анализа");
    expect(html).not.toContain("Нарушение не обнаружено");
  });
  it("A01 failed с кандидатами: жёлтый «Анализ не завершён», кандидаты помечены, зелёного нет", () => {
    const job = normalizeJob({
      job_id: "u",
      status: "failed",
      verdict: null,
      violations: [],
      verdict_state: "unknown",
      analysis_complete: false,
      coverage_problem: "meta.coverage.ratio не задан",
      error: "«нет нарушения» не подтверждено: meta.coverage.ratio не задан",
      candidates: [{ type: "Кандидат", type_id: "V05", timecode_sec: 7, description: "Описание.", evidence_frames: [] }],
      meta: { stages: { ingest: { status: "ok", required: true } } },
    });
    const html = wrap(<ResultView job={job} raw={{}} />);
    expect(html).toContain("Анализ не завершён");
    expect(html).toContain("Кандидаты — не подтверждены, в вердикт не вошли");
    expect(html).toContain("не вошло в вердикт");
    expect(html).not.toContain("Нарушение не обнаружено");
  });
  it("ответ сервера в формате A01 рендерится", () => {
    const job = normalizeJob({
      job_id: "x",
      status: "done",
      verdict: "violation",
      violations: [{ type: "Тип", timecode_sec: 22, description: "Описание.", evidence_frames: [{ timestamp_sec: 22, image_base64: "/9j/4AAQSkZJRgABAQAAAQABAAD" }] }],
    });
    const html = wrap(<ResultView job={job} raw={{}} />);
    expect(html).toContain("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD");
    expect(html).toContain("норма в каталоге не найдена");
  });
});

describe("ProcessingView / AboutScreen", () => {
  it("без progress — нет выдуманных процентов", () => {
    const html = wrap(<ProcessingView job={{ job_id: "p", status: "processing", violations: [], timeline: [], meta: {}, stage: "detect" }} />);
    expect(html).toContain("прогресс не передаётся");
    expect(html).not.toMatch(/\d+ %/);
  });
  it("с progress — процент и очередь", () => {
    const html = wrap(
      <ProcessingView job={{ job_id: "p", status: "processing", violations: [], timeline: [], meta: {}, stage: "queued", progress: 0, queuePosition: 2 }} />,
    );
    expect(html).toContain("позиция в очереди: 2");
  });
  it("О системе: типы и лицензии", () => {
    const html = wrap(<AboutScreen />);
    expect(html).toContain("V01");
    expect(html).toContain("Apache-2.0");
  });
});

describe("JobScreen-логика: failed с partial.json A03 (стадии, coverage)", () => {
  it("partial без нарушений, но со стадиями → подробный жёлтый отчёт с панелью полноты", () => {
    const job = normalizeJob({
      job_id: "f",
      status: "failed",
      error: "pipeline_failed",
      partial: {
        status: "processing",
        verdict: null,
        violations: [],
        coverage_problem: "оркестратор сообщил незавершённый анализ (meta.complete=false)",
        meta: { complete: false, stages: { ingest: { status: "ok", required: true }, temporal: { status: "ok" } }, coverage: { ingest: 1, temporal: 0 } },
      },
    });
    expect(job.partialResult?.meta.stages).toBeTruthy();
    const shown = { ...job.partialResult!, job_id: job.job_id, status: "failed" as const, error: job.error };
    const html = wrap(<ResultView job={shown} raw={{}} />);
    expect(html).toContain("Анализ не завершён");
    expect(html).toContain("Полнота анализа");
    expect(html).toContain("Временная модель");
    expect(html).not.toContain("Нарушение не обнаружено");
  });
});
