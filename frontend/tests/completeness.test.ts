import { describe, expect, it } from "vitest";
import { normalizeJob } from "../src/api/normalize";
import { analyzeCompleteness, readCoverage, readStages } from "../src/lib/completeness";

const base = { job_id: "c", status: "done", verdict: "no_violation", violations: [] };

describe("analyzeCompleteness (complete / partial / unknown)", () => {
  it("unknown, если сервер не подтвердил полноту — unknown ≠ «нарушений нет»", () => {
    const r = analyzeCompleteness(normalizeJob(base));
    expect(r.state).toBe("unknown");
    expect(r.reasons[0]).toMatch(/не подтвердил полноту/);
  });
  it("complete по meta.complete=true (A03)", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true } })).state).toBe("complete");
  });
  it("complete по coverage ≥ 0.98 и всем обязательным стадиям ok/cached", () => {
    const meta = { coverage: 0.99, required_stages: ["ingest"], stages: { ingest: { status: "ok" }, vlm_verify: { status: "skipped" } } };
    expect(analyzeCompleteness(normalizeJob({ ...base, meta })).state).toBe("complete");
  });
  it("partial: coverage < 0.98 (A04), с процентом", () => {
    const r = analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, coverage: 0.874 } }));
    expect(r.state).toBe("partial");
    expect(r.reasons.join(" ")).toMatch(/87 %/);
  });
  it("partial: обязательная стадия не выполнена (A03 stages)", () => {
    const meta = { stages: { detect: { status: "timeout", required: true, seconds: 30 } } };
    const r = analyzeCompleteness(normalizeJob({ ...base, meta }));
    expect(r.state).toBe("partial");
    expect(r.reasons.join(" ")).toMatch(/Поиск людей и объектов.*таймаут/);
  });
  it("partial: complete=false / degraded / deadline / failed", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: false } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, degraded: true } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, deadline_exceeded_s: 3.2 } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, status: "failed", verdict: null })).state).toBe("partial");
  });
  it("mock: meta.mock, origin=mock у стадии, FAKE PIPELINE в предупреждениях", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { mock: true } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { stages: { detect: { status: "ok", origin: "mock" } } } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { warnings: ["FAKE PIPELINE: result is synthetic"] } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true } })).mock).toBe(false);
  });
  it("кэш-стадии перечисляются", () => {
    const r = analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, stages: { detect: { status: "cached" } } } }));
    expect(r.state).toBe("complete");
    expect(r.cached).toEqual(["Поиск людей и объектов"]);
  });
});

describe("readCoverage / readStages", () => {
  it("форматы coverage", () => {
    expect(readCoverage({ coverage: 0.5 })).toBe(0.5);
    expect(readCoverage({ coverage: "87%" })).toBeCloseTo(0.87);
    expect(readCoverage({ coverage: { ratio: 0.9 } })).toBe(0.9);
    expect(readCoverage({ video: { coverage: 1 } })).toBe(1);
    expect(readCoverage({ coverage: "abc" })).toBeUndefined();
    expect(readCoverage({})).toBeUndefined();
  });
  it("стадии: подписи и статусы по-русски, обязательность из required_stages", () => {
    const st = readStages({ required_stages: ["fusion"], stages: { fusion: { status: "crashed", error: "x" }, temporal: { status: "disabled" } } });
    expect(st.find((x) => x.name === "fusion")).toMatchObject({ label: "Формирование отчёта", statusLabel: "сбой", required: true, ok: false });
    expect(st.find((x) => x.name === "temporal")).toMatchObject({ statusLabel: "выключена", required: false });
  });
});

describe("A01 UI-report и A03 coverage по стадиям (интеграция)", () => {
  it("A01: analysis_complete=true → complete; coverage_problem «нет meta.coverage» → unknown", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, analysis_complete: true })).state).toBe("complete");
    const r = analyzeCompleteness(normalizeJob({ ...base, analysis_complete: false, coverage_problem: "полнота анализа не подтверждена: нет meta.coverage" }));
    expect(r.state).toBe("unknown");
    expect(r.reasons[0]).toMatch(/нет meta.coverage/);
  });
  it("A01: coverage_problem «проанализировано 87 %» → partial", () => {
    const r = analyzeCompleteness(
      normalizeJob({ ...base, status: "done", analysis_complete: false, coverage_problem: "проанализировано 87% видео (< 98%): неизвестные участки ≠ «нарушений нет»" }),
    );
    expect(r.state).toBe("partial");
  });
  it("A03: meta.coverage = {стадия: доля}; обязательная стадия < 0.9 → partial; общая доля = ingest", () => {
    const meta = {
      complete: true,
      required_stages: ["ingest", "detect"],
      stages: { ingest: { status: "ok" }, detect: { status: "ok" }, temporal: { status: "ok" } },
      coverage: { ingest: 1.0, detect: 0.6, temporal: 0.0, vlm_discover: null },
    };
    const r = analyzeCompleteness(normalizeJob({ ...base, meta }));
    expect(r.state).toBe("partial");
    expect(r.coverage).toBe(1);
    expect(r.reasons.join(" ")).toMatch(/Поиск людей и объектов.*60 %/);
    expect(r.stages.find((x) => x.name === "temporal")?.coverage).toBe(0);
  });
  it("A01 UI-report: candidates и coverage верхнего уровня разбираются", () => {
    const j = normalizeJob({
      job_id: "u",
      status: "failed",
      verdict: null,
      violations: [],
      verdict_state: "unknown",
      analysis_complete: false,
      coverage_problem: "meta.coverage.ratio не задан",
      candidates: [{ type: "Кандидат", type_id: "V05", timecode_sec: 7, description: "", evidence_frames: [] }],
      warnings: ["«нет нарушения» не подтверждено"],
      coverage: { ingest: 1 },
      meta: {},
    });
    expect(j.candidates).toHaveLength(1);
    expect(j.verdictState).toBe("unknown");
    expect(j.meta.coverage).toEqual({ ingest: 1 });
    expect(j.warnings).toHaveLength(1);
    expect(analyzeCompleteness(j).state).toBe("partial"); // failed → неполный
  });
});
