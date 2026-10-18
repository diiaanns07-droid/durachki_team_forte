import { describe, expect, it } from "vitest";
import { normalizeJob, notCheckedFor, overlayBoxes, rulesWithStatus } from "../src/api/normalize";
import { typeInfo, TYPE_CATALOG } from "../src/lib/catalog";
import { DEFAULT_UI_CONFIG, mergeUiConfig } from "../src/lib/uiConfig";

const job = normalizeJob({
  job_id: "k",
  status: "done",
  verdict: "violation",
  violations: [
    {
      type: "Кассир покинул рабочее место, оставив наличные без присмотра",
      type_id: "V02",
      timecode_sec: 30,
      description: "",
      evidence_frames: [],
      rules: [{ document: "Док", clause: null, url: "https://adilet.zan.kz/rus/docs/X", quote: "q", verified: false }],
    },
  ],
  meta: {
    ui: {
      evidence_overlays: [
        {
          violation_index: 0,
          type_id: "V02",
          timecode_sec: 30,
          shapes: [
            { kind: "box", key: "cash", label: "наличные", color: "#f5a524", bbox: [0.1, 0.2, 0.3, 0.4] },
            { kind: "blur", bbox: [0, 0, 0.1, 0.1] },
            { kind: "box", bbox: [0.5, 0.5, 0.4, 0.6] },
            { kind: "box", label: "x", color: "red; background:url(x)", bbox: [0.2, 0.2, 1.4, 0.3] },
          ],
          norms: [{ document: "Док", clause: null, status_ru: "НЕ подтверждено: требует сверки юристом" }],
        },
      ],
    },
  },
});

describe("карточка: подсветка, нормы, «не проверяется»", () => {
  it("overlayBoxes: только box, валидные bbox, безопасный цвет", () => {
    const b = overlayBoxes(job.violations[0], job);
    expect(b).toHaveLength(2);
    expect(b[0]).toEqual({ label: "наличные", color: "#f5a524", bbox: [0.1, 0.2, 0.3, 0.4] });
    expect(b[1].color).toBe("#ff3b5c"); // CSS-инъекция в цвете отброшена
    expect(b[1].bbox[2]).toBe(1); // клип к 0..1
  });
  it("rulesWithStatus: status_ru из оверлея A12", () => {
    expect(rulesWithStatus(job.violations[0], job)[0].status_ru).toMatch(/НЕ подтверждено/);
  });
  it("notCheckedFor: из каталога A13 по type_id, если сервер не прислал", () => {
    expect(notCheckedFor(job.violations[0], job)).toContain("заперты ли ящики стола, не видимые камере");
  });
  it("каталог: канонические названия A13, V09, переходные типы", () => {
    expect(typeInfo("V01")?.name).toBe("Посторонний в зоне кассира");
    expect(typeInfo("v09")?.name).toBe("Личные вещи на кассовом столе");
    expect(TYPE_CATALOG.filter((t) => t.transition).map((t) => t.id)).toEqual(["V02", "V03", "V07"]);
  });
});

describe("ui-config.json", () => {
  it("битые значения игнорируются, валидные применяются", () => {
    const c = mergeUiConfig(DEFAULT_UI_CONFIG, {
      pollIntervalMs: 5,
      seekPauses: "yes",
      ptsOffsetMode: "meta_t0",
      beforeAfterTypes: ["v02"],
      episodePrerollS: 3,
    });
    expect(c.pollIntervalMs).toBe(DEFAULT_UI_CONFIG.pollIntervalMs);
    expect(c.seekPauses).toBe(true);
    expect(c.ptsOffsetMode).toBe("meta_t0");
    expect(c.beforeAfterTypes).toEqual(["V02"]);
    expect(c.episodePrerollS).toBe(3);
    expect(mergeUiConfig(DEFAULT_UI_CONFIG, "garbage")).toEqual(DEFAULT_UI_CONFIG);
  });
  it("по умолчанию сдвиг PTS выключен (гипотеза до замера L2)", () => {
    expect(DEFAULT_UI_CONFIG.ptsOffsetMode).toBe("none");
  });
});
