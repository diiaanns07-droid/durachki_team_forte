import { describe, expect, it } from "vitest";
import {
  confidenceLevel,
  estimateEta,
  formatBytes,
  formatDuration,
  formatPercent,
  formatTimecode,
  plural,
  riskColor,
} from "../src/lib/format";

describe("formatTimecode (mm:ss.s)", () => {
  it("форматирует секунды", () => {
    expect(formatTimecode(0)).toBe("00:00.0");
    expect(formatTimecode(22)).toBe("00:22.0");
    expect(formatTimecode(65.43)).toBe("01:05.4");
    expect(formatTimecode(145)).toBe("02:25.0");
  });
  it("переносит разряды при округлении", () => {
    expect(formatTimecode(59.96)).toBe("01:00.0");
    expect(formatTimecode(119.95)).toBe("02:00.0");
  });
  it("часы", () => {
    expect(formatTimecode(3600)).toBe("1:00:00.0");
    expect(formatTimecode(3725.25)).toBe("1:02:05.3");
  });
  it("мусор → 00:00.0", () => {
    expect(formatTimecode(Number.NaN)).toBe("00:00.0");
    expect(formatTimecode(-3)).toBe("00:00.0");
    expect(formatTimecode(undefined)).toBe("00:00.0");
    expect(formatTimecode(null)).toBe("00:00.0");
  });
});

describe("прочие форматтеры", () => {
  it("formatDuration", () => {
    expect(formatDuration(42.4)).toBe("42 с");
    expect(formatDuration(125)).toBe("2 мин 05 с");
    expect(formatDuration(Number.NaN)).toBe("—");
  });
  it("formatPercent ограничивает 0..1", () => {
    expect(formatPercent(0.874)).toBe("87 %");
    expect(formatPercent(1.7)).toBe("100 %");
    expect(formatPercent(undefined)).toBe("—");
  });
  it("formatBytes", () => {
    expect(formatBytes(512)).toBe("512 Б");
    expect(formatBytes(29 * 1024 * 1024)).toBe("29,0 МБ");
  });
  it("plural", () => {
    expect(plural(1, ["эпизод", "эпизода", "эпизодов"])).toBe("эпизод");
    expect(plural(3, ["эпизод", "эпизода", "эпизодов"])).toBe("эпизода");
    expect(plural(11, ["эпизод", "эпизода", "эпизодов"])).toBe("эпизодов");
    expect(plural(22, ["эпизод", "эпизода", "эпизодов"])).toBe("эпизода");
  });
  it("estimateEta только при реальном прогрессе", () => {
    expect(estimateEta(10, 0)).toBeNull();
    expect(estimateEta(10, 0.5)).toBeCloseTo(10);
    expect(estimateEta(10, 1)).toBe(0);
  });
  it("confidenceLevel", () => {
    expect(confidenceLevel(0.9).tone).toBe("high");
    expect(confidenceLevel(0.6).tone).toBe("medium");
    expect(confidenceLevel(0.2).tone).toBe("low");
    expect(confidenceLevel(undefined).tone).toBe("unknown");
  });
  it("riskColor возвращает hsla", () => {
    expect(riskColor(0)).toMatch(/^hsla\(/);
    expect(riskColor(1, 0.5)).toContain("0.5)");
    expect(riskColor(Number.NaN)).toMatch(/^hsla\(/);
  });
});
