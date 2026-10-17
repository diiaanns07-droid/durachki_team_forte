import { describe, expect, it } from "vitest";
import {
  imageSrc,
  inferDuration,
  isPartial,
  normalizeJob,
  normalizeProgress,
  normalizeStatus,
  normalizeTimeline,
  normalizeVerdict,
  toJuryJson,
  violationFacts,
  violationSources,
  violationVlm,
  reviewHints,
} from "../src/api/normalize";

// Формы ответов — как в contracts/examples (A01).
const processing = { job_id: "j1", status: "processing", verdict: null, violations: [], progress: 0.35, stage: "detect" };
const failed = { job_id: "j5", status: "failed", verdict: null, violations: [], error: "не удалось декодировать видео" };
const clean = { job_id: "j4", status: "done", verdict: "no_violation", violations: [] };
const full = {
  job_id: "j2",
  status: "done",
  verdict: "violation",
  violations: [
    {
      type: "Посторонний в кассовом помещении",
      timecode_sec: 22.0,
      description: "На 22.0 с в зоне за кассовым столом находится второй человек. Признаки возможного нарушения.",
      evidence_frames: [{ timestamp_sec: 22.0, image_base64: "/9j/AAAA" }],
      type_id: "V01",
      confidence: 0.87,
      t_start: 21.2,
      t_end: 26.8,
      rule: { document: "Док", clause: "п. 1", url: "https://adilet.zan.kz/rus/docs/X", quote: "цитата", verified: false },
    },
  ],
  progress: 1.0,
  stage: "done",
  timeline: [
    [0.0, 0.02],
    [22.0, 0.93],
    [26.8, 0.4],
  ],
  meta: { version: "0.1.0", duration_sec: 48.3, timings: { total: 31.5 }, models: ["x"], warnings: [] },
};

describe("normalizeJob", () => {
  it("processing: прогресс и стадия", () => {
    const j = normalizeJob(processing);
    expect(j.status).toBe("processing");
    expect(j.progress).toBeCloseTo(0.35);
    expect(j.stage).toBe("detect");
    expect(j.verdict).toBeNull();
  });
  it("failed: ошибка и без вердикта", () => {
    const j = normalizeJob(failed);
    expect(j.status).toBe("failed");
    expect(j.error).toContain("декодировать");
    expect(j.verdict).toBeNull();
  });
  it("done без нарушений", () => {
    const j = normalizeJob(clean);
    expect(j.verdict).toBe("no_violation");
    expect(j.violations).toHaveLength(0);
    expect(j.timeline).toEqual([]);
  });
  it("done c нарушением: rule (ед. ч.) → rules[], timeline, длительность", () => {
    const j = normalizeJob(full);
    expect(j.verdict).toBe("violation");
    expect(j.violations[0].rules).toHaveLength(1);
    expect(j.violations[0].rules?.[0].clause).toBe("п. 1");
    expect(j.timeline).toHaveLength(3);
    expect(inferDuration(j)).toBeCloseTo(48.3);
  });
  it("устойчив к строковым числам и мусору", () => {
    const j = normalizeJob({
      job_id: 7,
      status: "DONE",
      verdict: "Violation",
      violations: [null, 5, { type: "X", timecode_sec: "12,5", evidence_frames: ["/9j/4AAQSkZJRgABAQAAAQABAAD", { foo: 1 }, "abc"], confidence: 3 }],
      timeline: [[1, 2], { t: 0, risk: 0.1 }, "bad", [-1, 0.5]],
    });
    expect(j.job_id).toBe("7");
    expect(j.violations).toHaveLength(1);
    expect(j.violations[0].timecode_sec).toBe(12.5);
    expect(j.violations[0].evidence_frames).toHaveLength(1);
    expect(j.violations[0].confidence).toBe(1);
    expect(j.timeline).toEqual([
      [0, 0.1],
      [1, 1],
    ]);
  });
  it("бросает на не-объекте", () => {
    expect(() => normalizeJob("oops")).toThrow();
    expect(() => normalizeJob([1, 2])).toThrow();
  });
  it("done без вердикта → по наличию нарушений", () => {
    expect(normalizeJob({ status: "done", violations: [] }).verdict).toBe("no_violation");
    expect(normalizeJob({ status: "done", violations: [{ type: "A", timecode_sec: 1, description: "", evidence_frames: [] }] }).verdict).toBe("violation");
  });
  it("нарушения сортируются по времени", () => {
    const j = normalizeJob({
      status: "done",
      verdict: "violation",
      violations: [
        { type: "B", timecode_sec: 40, description: "", evidence_frames: [] },
        { type: "A", timecode_sec: 5, description: "", evidence_frames: [] },
      ],
    });
    expect(j.violations.map((v) => v.type)).toEqual(["A", "B"]);
  });
});

describe("вспомогательные", () => {
  it("normalizeStatus", () => {
    expect(normalizeStatus("completed")).toBe("done");
    expect(normalizeStatus("error")).toBe("failed");
    expect(normalizeStatus("queued")).toBe("queued");
    expect(normalizeStatus("whatever")).toBe("processing");
  });
  it("normalizeVerdict", () => {
    expect(normalizeVerdict("no-violation")).toBe("no_violation");
    expect(normalizeVerdict(true)).toBe("violation");
    expect(normalizeVerdict("")).toBeNull();
  });
  it("normalizeProgress 0..100 → 0..1", () => {
    expect(normalizeProgress(35)).toBeCloseTo(0.35);
    expect(normalizeProgress(0.5)).toBe(0.5);
    expect(normalizeProgress(undefined)).toBeUndefined();
  });
  it("normalizeTimeline сортирует", () => {
    expect(normalizeTimeline([[3, 0.1], [1, 0.2]])).toEqual([
      [1, 0.2],
      [3, 0.1],
    ]);
  });
  it("imageSrc распознаёт форматы", () => {
    expect(imageSrc("/9j/abc")).toBe("data:image/jpeg;base64,/9j/abc");
    expect(imageSrc("iVBORw0")).toBe("data:image/png;base64,iVBORw0");
    expect(imageSrc("PHN2ZyB4")).toBe("data:image/svg+xml;base64,PHN2ZyB4");
    expect(imageSrc("data:image/jpeg;base64,xx")).toBe("data:image/jpeg;base64,xx");
  });
  it("facts/sources: из нарушения, meta.events или описания", () => {
    const j = normalizeJob(full);
    expect(violationFacts(j.violations[0], j).length).toBeGreaterThan(0);
    expect(violationSources(j.violations[0], j)).toEqual([]);
    const withEvents = normalizeJob({ ...full, meta: { events: [{ type_id: "V01", t_peak: 22, source: "rules", facts: ["факт"] }] } });
    expect(violationSources(withEvents.violations[0], withEvents)).toEqual(["rules"]);
    expect(violationFacts(withEvents.violations[0], withEvents)).toEqual(["факт"]);
  });
});

describe("неполный анализ и JSON жюри", () => {
  it("isPartial по флагам meta", () => {
    expect(isPartial(normalizeJob(clean))).toBe(false);
    expect(isPartial(normalizeJob({ ...clean, meta: { partial: true } }))).toBe(true);
    expect(isPartial(normalizeJob({ ...clean, meta: { skipped_stages: ["vlm"] } }))).toBe(true);
    expect(isPartial(normalizeJob({ ...clean, meta: { warnings: ["vlm skipped: timeout"] } }))).toBe(true);
    expect(isPartial(normalizeJob({ ...clean, meta: { warnings: ["video rotated 90"] } }))).toBe(false);
  });
  it("toJuryJson — только обязательные поля ТЗ", () => {
    const out = toJuryJson(normalizeJob(full)) as { violations: Record<string, unknown>[] } & Record<string, unknown>;
    expect(Object.keys(out).sort()).toEqual(["job_id", "status", "verdict", "violations"]);
    expect(Object.keys(out.violations[0]).sort()).toEqual(["description", "evidence_frames", "timecode_sec", "type"]);
  });
});

describe("imageSrc: пути и base64", () => {
  it("JPEG base64 '/9j/' не путается с путём", () => {
    expect(imageSrc("/9j/4AAQSkZJRgABAQ==")).toMatch(/^data:image\/jpeg;base64,/);
  });
  it("URL/пути остаются как есть", () => {
    expect(imageSrc("/ui/v1/jobs/1/frames/0.jpg")).toBe("/ui/v1/jobs/1/frames/0.jpg");
    expect(imageSrc("https://x/y.jpg")).toBe("https://x/y.jpg");
    expect(imageSrc("blob:abc")).toBe("blob:abc");
  });
});

describe("поля A02", () => {
  it("processing: queue_position и elapsed_s", () => {
    const j = normalizeJob({ job_id: "q", status: "processing", stage: "queued", progress: 0, queue_position: 3, elapsed_s: 4.2 });
    expect(j.queuePosition).toBe(3);
    expect(j.elapsedSec).toBeCloseTo(4.2);
  });
  it("failed + partial → partialResult, но статус остаётся failed", () => {
    const j = normalizeJob({
      job_id: "f",
      status: "failed",
      verdict: null,
      violations: [],
      error: "timeout",
      partial: { job_id: "f", verdict: "violation", violations: [{ type: "A", timecode_sec: 3, description: "", evidence_frames: [] }] },
    });
    expect(j.status).toBe("failed");
    expect(j.verdict).toBeNull();
    expect(j.partialResult?.violations).toHaveLength(1);
  });
});

describe("A12: meta.ui.evidence_overlays", () => {
  const raw = {
    job_id: "o",
    status: "done",
    verdict: "violation",
    violations: [
      { type: "Б", type_id: "V02", timecode_sec: 50, description: "Описание Б.", evidence_frames: [] },
      { type: "А", type_id: "V01", timecode_sec: 10, description: "Описание А.", evidence_frames: [] },
    ],
    meta: {
      ui: {
        evidence_overlays: [
          { violation_index: 0, type_id: "V02", timecode_sec: 50, facts: ["факт Б"], sources: { temporal: 0.4, rules: 0.9 }, vlm: "reject_weak" },
          { violation_index: 1, type_id: "V01", timecode_sec: 10, facts: ["факт А"], sources: { vlm: 0.7 }, vlm: "confirm" },
        ],
      },
    },
  };
  it("сопоставляет по исходному индексу даже после сортировки по времени", () => {
    const j = normalizeJob(raw);
    expect(j.violations.map((v) => v.type_id)).toEqual(["V01", "V02"]);
    expect(violationFacts(j.violations[0], j)).toEqual(["факт А"]);
    expect(violationFacts(j.violations[1], j)).toEqual(["факт Б"]);
  });
  it("sources-словарь → имена по убыванию веса; состояние VLM", () => {
    const j = normalizeJob(raw);
    expect(violationSources(j.violations[1], j)).toEqual(["rules", "temporal"]);
    expect(violationVlm(j.violations[0], j)?.tone).toBe("ok");
    expect(violationVlm(j.violations[1], j)?.label).toBe("VLM не подтвердила");
  });
});

describe("A12: review_hints", () => {
  it("разбор и сортировка, причины по-русски", () => {
    const j = normalizeJob({
      status: "done",
      verdict: "no_violation",
      violations: [],
      meta: { ui: { review_hints: [{ type: "B", timecode_sec: 9, reasons: ["below_threshold"], reasons_ru: ["ниже порога"] }, { type_id: "V05", timecode_sec: 2 }, { type: "bad" }] } },
    });
    const h = reviewHints(j);
    expect(h.map((x) => x.timecode_sec)).toEqual([2, 9]);
    expect(h[1].reasons).toEqual(["ниже порога"]);
    expect(h[0].type).toBe("V05");
    expect(j.verdict).toBe("no_violation");
  });
});
