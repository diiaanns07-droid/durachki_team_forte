// Регрессии по аудиту интеграции 2055fc4 (синтетические фикстуры в формате A02/A01/A03/A12).
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DemoBackend } from "../src/api/demo";
import { isImagePayload, normalizeJob, overlayPeak, toJuryJson } from "../src/api/normalize";
import type { EvidenceFrame } from "../src/api/types";
import { AppContext, type AppContextValue } from "../src/context";
import { BeforeAfter, boxesForFrame } from "../src/components/Evidence";
import ResultView from "../src/components/ResultView";
import { analyzeCompleteness } from "../src/lib/completeness";
import { DEFAULT_UI_CONFIG } from "../src/lib/uiConfig";
import { historyView } from "../src/lib/verdict";

const ctx: AppContextValue = {
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
};
const render = (n: JSX.Element) => renderToString(<AppContext.Provider value={ctx}>{n}</AppContext.Provider>);
const JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD";

describe("кадры без изображения не подменяются", () => {
  it("isImagePayload: base64/URL — да; заглушки — нет", () => {
    expect(isImagePayload(JPEG)).toBe(true);
    expect(isImagePayload("data:image/jpeg;base64,xx")).toBe(true);
    expect(isImagePayload("/ui/v1/jobs/1/frames/0.jpg")).toBe(true);
    for (const bad of ["<jpeg omitted>", "omitted", "", "null", "N/A"]) expect(isImagePayload(bad)).toBe(false);
  });
  it("«<jpeg omitted>» (A02 у failed) → missing, момент сохранён; в JSON жюри не попадает", () => {
    const j = normalizeJob({
      status: "done",
      verdict: "violation",
      violations: [{ type: "A", timecode_sec: 5, description: "", evidence_frames: [{ timestamp_sec: 5, image_base64: "<jpeg omitted>" }, { timestamp_sec: 6, image_base64: JPEG }] }],
    });
    expect(j.violations[0].evidence_frames[0]).toEqual({ timestamp_sec: 5, image_base64: "", missing: true });
    const jury = toJuryJson(j) as { violations: { evidence_frames: EvidenceFrame[] }[] };
    expect(jury.violations[0].evidence_frames).toEqual([{ timestamp_sec: 6, image_base64: JPEG }]);
  });
  it("карточка: плашка «кадр не передан», без <img> с заглушкой", () => {
    const j = normalizeJob({
      status: "done",
      verdict: "violation",
      violations: [{ type: "A", timecode_sec: 5, description: "", evidence_frames: [{ timestamp_sec: 5, image_base64: "<jpeg omitted>" }] }],
      meta: { complete: true },
    });
    const html = render(<ResultView job={j} raw={{}} />);
    expect(html).toContain("кадр не передан");
    expect(html).not.toContain("jpeg omitted");
    expect(html).not.toMatch(/<img[^>]+base64,(&lt;|<)/);
  });
  it("«до/после»: кадр раньше начала эпизода НЕ подставляется как «после»", () => {
    const v = { type: "B", type_id: "V02", timecode_sec: 30, t_start: 30, t_end: 40, description: "", evidence_frames: [{ timestamp_sec: 25, image_base64: JPEG }] };
    const html = renderToString(<BeforeAfter v={v} peakT={30} boxes={[]} highlight onSeek={() => undefined} onOpen={() => undefined} />);
    expect(html).toContain("кадра «после» нет");
  });
});

describe("подсветка A12 — только на кадре t_peak", () => {
  const boxes = [{ label: "x", color: "#ff0000", bbox: [0.1, 0.1, 0.2, 0.2] as [number, number, number, number] }];
  it("рамки на кадре пика есть, на соседнем (0,4 с) — нет", () => {
    expect(DEFAULT_UI_CONFIG.highlightMaxDtS).toBe(0.1);
    expect(boxesForFrame({ timestamp_sec: 30.03, image_base64: JPEG }, 30, boxes)).toHaveLength(1);
    expect(boxesForFrame({ timestamp_sec: 30.4, image_base64: JPEG }, 30, boxes)).toHaveLength(0);
    expect(boxesForFrame({ timestamp_sec: 30, image_base64: "", missing: true }, 30, boxes)).toHaveLength(0);
  });
  it("якорь — t_peak из оверлея (таймкод может быть началом эпизода)", () => {
    const j = normalizeJob({
      status: "done",
      verdict: "violation",
      violations: [{ type: "A", type_id: "V02", timecode_sec: 28, description: "", evidence_frames: [] }],
      meta: { ui: { evidence_overlays: [{ violation_index: 0, type_id: "V02", timecode_sec: 30.5 }] } },
    });
    expect(overlayPeak(j.violations[0], j)).toBe(30.5);
  });
});

describe("A02 2055fc4: failed + partial-обёртка", () => {
  const wrapped = {
    job_id: "w",
    status: "failed",
    verdict: null,
    violations: [],
    error: "pipeline_failed: incomplete analysis (meta.completeness/coverage): no_violation is not issued",
    partial: {
      job_id: "w",
      complete: false,
      code: "pipeline_failed",
      partial: { status: "processing", meta: { complete: false, stages: { ingest: { status: "ok" } } } },
      rejected_result: {
        status: "failed",
        verdict: null,
        verdict_state: "no_violation",
        analysis_complete: true,
        violations: [],
        candidates: [{ type: "K", type_id: "V05", timecode_sec: 3, description: "", evidence_frames: [{ timestamp_sec: 3, image_base64: "<jpeg omitted>" }] }],
        timeline: [[0, 0.1], [3, 0.4]],
        meta: { complete: true, stages: { ingest: { status: "ok", required: true }, detect: { status: "ok", required: true } }, coverage: { ratio: 1, complete: true, by_stage: { ingest: 1, detect: 1 } } },
      },
    },
  };
  it("разворачивает rejected_result (полный meta), кадры-заглушки → missing", () => {
    const j = normalizeJob(wrapped);
    expect(j.partialResult?.meta.stages).toBeTruthy();
    expect(j.partialResult?.candidates?.[0].evidence_frames[0].missing).toBe(true);
  });
  it("без rejected_result берёт partial.partial (partial.json A03)", () => {
    const { rejected_result: _r, ...rest } = wrapped.partial;
    void _r;
    const j = normalizeJob({ ...wrapped, partial: rest });
    expect(j.partialResult?.meta.complete).toBe(false);
  });
  it("показ: жёлтый «Анализ не завершён», не зелёный, кандидат с плашкой кадра", () => {
    const j = normalizeJob(wrapped);
    const shown = { ...j.partialResult!, job_id: j.job_id, status: "failed" as const, error: j.error };
    expect(analyzeCompleteness(shown).state).toBe("partial");
    const html = render(<ResultView job={shown} raw={{}} />);
    expect(html).toContain("Анализ не завершён");
    expect(html).toContain("кадр не передан");
    expect(html).not.toContain("Нарушение не обнаружено");
  });
});

describe("канонический meta.coverage A01/A03: by_stage и complete=false", () => {
  it("доли стадий из by_stage; coverage.complete=false → неполный", () => {
    const meta = {
      complete: true,
      required_stages: ["detect"],
      stages: { detect: { status: "ok" } },
      coverage: { ratio: 1, complete: false, by_stage: { detect: 0.5 } },
    };
    const r = analyzeCompleteness(normalizeJob({ status: "done", verdict: "no_violation", violations: [], meta }));
    expect(r.state).toBe("partial");
    expect(r.stages[0].coverage).toBe(0.5);
    expect(r.reasons.join(" ")).toMatch(/покрытие записи отмечено сервером как неполное/);
  });
});

describe("«Недавние проверки»: зелёный только для подтверждённо полного не-mock", () => {
  it("complete → success; unknown/старые → neutral; mock → neutral; partial → warning", () => {
    expect(historyView({ verdict: "no_violation", completeness: "complete" }).tone).toBe("success");
    expect(historyView({ verdict: "no_violation" }).tone).toBe("neutral");
    expect(historyView({ verdict: "no_violation", completeness: "unknown" }).label).toMatch(/полнота не подтверждена/);
    expect(historyView({ verdict: "no_violation", completeness: "complete", mock: true }).tone).toBe("neutral");
    expect(historyView({ verdict: "violation", n_violations: 2, mock: true }).label).toMatch(/mock/);
    expect(historyView({ verdict: "no_violation", completeness: "partial" }).tone).toBe("warning");
    expect(historyView({ status: "failed" }).tone).toBe("warning");
  });
});
