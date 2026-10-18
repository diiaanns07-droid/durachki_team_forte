// Наблюдения правил (ml.seqrules): разбор ответа, связь с нарушением, кадр по времени, оверлей и объяснение.
// Только синтетические данные.
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { episodesForViolation, frameAt, parseObservations } from "../src/api/observations";
import type { Violation } from "../src/api/types";
import ObservationOverlay, { DEFAULT_LAYERS } from "../src/components/ObservationOverlay";
import RuleObservations from "../src/components/RuleObservations";
import { contentRect } from "../src/components/VideoPlayer";

const kp = Array.from({ length: 17 }, (_, i) => [0.2 + 0.01 * i, 0.2 + 0.02 * i, 0.9]);
const RAW = {
  job_id: "synthetic-1",
  items: [
    {
      kind: "meta",
      work_zone: "cashier_area",
      zones: [{ name: "cashier_area", polygon: [[0, 0], [0.5, 0], [0.5, 0.6], [0, 0.6]], conf: 0.9 }],
      rules: { phone: { status: "ok", episodes: 2 }, safe: { status: "unavailable", reason: "рабочая зона не построена" } },
      warnings: [],
    },
    {
      kind: "episode", id: "A-1-1", rule: "phone", type_id: "V05", status: "event", track_id: 1, role: "cashier",
      t_start: 1.0, t_end: 2.0, t_anchor: 1.0, counts: { observed: 5, unknown: 1 },
      frames: [
        { t: 1.4, state: "unknown", person: [0.1, 0.1, 0.4, 0.55] },
        { t: 1.0, state: "observed", person: [0.1, 0.1, 0.4, 0.55], object: [0.33, 0.4, 0.36, 0.42], object_label: "cell_phone",
          object_score: 0.62, joint: "right_wrist", joint_xy: [0.33, 0.42], kp },
        { t: "битый", state: "observed" },
      ],
    },
    { kind: "episode", id: "A-1-2", rule: "phone", type_id: "V05", status: "too_short", track_id: 1, t_start: 5, t_end: 5.4, frames: [] },
    { kind: "episode", id: "C-2-1", rule: "entry", type_id: "V01", status: "event", track_id: 2, t_start: 1.2, t_end: 3, frames: [] },
  ],
};
const V05: Violation = { type: "Телефон", type_id: "V05", timecode_sec: 1.0, t_start: 1.0, t_end: 2.0, description: "", evidence_frames: [] };

describe("parseObservations", () => {
  it("meta + эпизоды; битые кадры отброшены, кадры по времени", () => {
    const o = parseObservations(RAW)!;
    expect(o.workZone).toBe("cashier_area");
    expect(o.rules.safe).toEqual({ status: "unavailable", reason: "рабочая зона не построена", episodes: undefined });
    const ep = o.episodes.find((e) => e.id === "A-1-1")!;
    expect(ep.frames.map((f) => f.t)).toEqual([1.0, 1.4]);
    expect(ep.frames[0].kp?.length).toBe(17);
  });
  it("не наблюдения → null, а не «пусто»", () => {
    expect(parseObservations(null)).toBeNull();
    expect(parseObservations({ items: [] })).toBeNull();
    expect(parseObservations({ items: [{ kind: "episode" }] })).toBeNull();
  });
});

describe("episodesForViolation / frameAt", () => {
  const o = parseObservations(RAW)!;
  it("только события того же типа, пересекающиеся по времени", () => {
    expect(episodesForViolation(o, V05).map((e) => e.id)).toEqual(["A-1-1"]);
    expect(episodesForViolation(o, { ...V05, t_start: 10, t_end: 12, timecode_sec: 10 })).toEqual([]);
    expect(episodesForViolation(null, V05)).toEqual([]);
  });
  it("ближайший кадр в допуске; вне эпизода — ничего", () => {
    const ep = o.episodes[0];
    expect(frameAt(ep, 1.1)?.t).toBe(1.0);
    expect(frameAt(ep, 1.35)?.t).toBe(1.4);
    expect(frameAt(ep, 3.0)).toBeNull();
  });
});

describe("ObservationOverlay", () => {
  const o = parseObservations(RAW)!;
  const sel = new Set(["A-1-1"]);
  it("зона, человек с номером трека, скелет, сустав и предмет", () => {
    const html = renderToString(
      <ObservationOverlay obs={o} episodes={[o.episodes[0]]} selectedIds={sel} t={1.0} width={300} height={500} layers={DEFAULT_LAYERS} />,
    );
    expect(html).toContain('data-zone="cashier_area"');
    expect(html).toContain("рабочая зона");
    expect(html).toContain("#1");
    expect(html).toContain("кассир");
    expect(html).toContain('data-joint="right_wrist"');
    expect(html).toContain("телефон 0.62");
    expect((html.match(/<line /g) ?? []).length).toBeGreaterThan(5);
  });
  it("слои выключаются; «неизвестно» подписано; вне эпизода — пусто", () => {
    const off = renderToString(
      <ObservationOverlay obs={o} episodes={[o.episodes[0]]} selectedIds={sel} t={1.0} width={300} height={500}
        layers={{ skeletons: false, objects: false, zones: false, all: false }} />,
    );
    expect(off).not.toContain("#1");
    expect(off).not.toContain("data-zone");
    expect(off).not.toContain("телефон");
    const unk = renderToString(
      <ObservationOverlay obs={o} episodes={[o.episodes[0]]} selectedIds={sel} t={1.4} width={300} height={500} layers={DEFAULT_LAYERS} />,
    );
    expect(unk).toContain('data-state="unknown"');
    expect(unk).toContain("неизвестно");
    const none = renderToString(
      <ObservationOverlay obs={o} episodes={[o.episodes[0]]} selectedIds={sel} t={9} width={300} height={500} layers={DEFAULT_LAYERS} />,
    );
    expect(none).not.toContain("data-episode");
  });
});

describe("contentRect: оверлей совпадает с видимой областью кадра (object-fit: contain)", () => {
  it("портретный кадр в широкой сцене — поля по бокам", () => {
    const r = contentRect({ w: 1000, h: 800 }, 464 / 848);
    expect(r.h).toBe(800);
    expect(r.w).toBeCloseTo(800 * (464 / 848));
    expect(r.left).toBeCloseTo((1000 - r.w) / 2);
    expect(r.top).toBe(0);
  });
  it("горизонтальный кадр в высокой сцене — поля сверху и снизу", () => {
    const r = contentRect({ w: 400, h: 800 }, 16 / 9);
    expect(r.w).toBe(400);
    expect(r.top).toBeCloseTo((800 - 225) / 2);
  });
});

describe("RuleObservations", () => {
  const o = parseObservations(RAW)!;
  it("статусы правил с причиной и состав выбранного эпизода", () => {
    const html = renderToString(<RuleObservations obs={o} selected={episodesForViolation(o, V05)} violation={V05} onSeek={() => undefined} />);
    expect(html).toContain("проверено");
    expect(html).toContain("неприменимо — рабочая зона не построена");
    expect(html).toContain("Человек #1");
    expect(html).toContain("наблюдалось 5");
    expect(html).toContain("неизвестно 1");
  });
  it("нарушение без наблюдений правил — так и сказано", () => {
    const html = renderToString(<RuleObservations obs={o} selected={[]} violation={V05} onSeek={() => undefined} />);
    expect(html).toContain("наблюдений правил нет");
  });
});
