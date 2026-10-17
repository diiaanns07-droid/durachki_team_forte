import { describe, expect, it } from "vitest";
import { verdictView, VERDICT_TEXT } from "../src/lib/verdict";
import { stageLabel, STEPS, stepIndex } from "../src/lib/stages";
import { validateVideoFile } from "../src/lib/validate";
import { parseHash, routeHash } from "../src/lib/router";

const v = { type: "X", timecode_sec: 1, description: "", evidence_frames: [] };

describe("verdictView", () => {
  it("формулировки ТЗ", () => {
    expect(verdictView({ status: "done", verdict: "violation", violations: [v] }).title).toBe("Обнаружены признаки возможных нарушений");
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, "complete").title).toBe("Нарушение не обнаружено");
    expect(VERDICT_TEXT.no_violation).toBe("Нарушение не обнаружено");
  });
  it("зелёный — только для полного завершённого анализа", () => {
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, "complete").tone).toBe("success");
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, "unknown").tone).toBe("neutral");
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }).flags).toContain("ПОЛНОТА НЕ ПОДТВЕРЖДЕНА");
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, "complete", true).tone).toBe("neutral");
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, "complete", true).title).toMatch(/Тестовый результат/);
    expect(verdictView({ status: "done", verdict: "no_violation", violations: [] }, true).tone).toBe("warning");
    expect(verdictView({ status: "failed", verdict: "no_violation", violations: [] }).tone).toBe("warning");
    expect(verdictView({ status: "processing", verdict: null, violations: [] }).tone).toBe("neutral");
    expect(verdictView({ status: "done", verdict: null, violations: [] }).tone).toBe("warning");
  });
  it("нарушение — красный", () => {
    expect(verdictView({ status: "done", verdict: "violation", violations: [v, v] }).tone).toBe("danger");
  });
});

describe("стадии", () => {
  it("сопоставляет стадии бэкенда", () => {
    expect(stepIndex("queued", undefined)).toBe(0);
    expect(stepIndex("ingest", 0.1)).toBe(1);
    expect(stepIndex("detect", 0.3)).toBe(2);
    expect(stepIndex("scene", 0.6)).toBe(3);
    expect(stepIndex("vlm", 0.8)).toBe(4);
    expect(stepIndex("fusion", 0.95)).toBe(5);
  });
  it("русские подписи стадий A03", () => {
    expect(stepIndex("Декодирование", 0.05)).toBe(1);
    expect(stepIndex("Поиск людей и объектов", 0.2)).toBe(2);
    expect(stepIndex("Анализ поз", 0.35)).toBe(2);
    expect(stepIndex("Состояние сцены", 0.5)).toBe(3);
    expect(stepIndex("Временная модель", 0.6)).toBe(4);
    expect(stepIndex("Проверка VLM", 0.8)).toBe(4);
    expect(stepIndex("Формирование отчёта", 0.95)).toBe(5);
    expect(stageLabel("Поиск людей и объектов")).toBe("поиск людей и объектов");
  });
  it("неизвестная стадия — по реальному прогрессу", () => {
    expect(stepIndex("xyz", 0.99)).toBe(STEPS.length - 1);
    expect(stageLabel("temporal")).toBe("временная модель");
    expect(stageLabel("custom-stage")).toBe("custom-stage");
  });
});

describe("validateVideoFile", () => {
  it("принимает .mov/.mp4", () => {
    expect(validateVideoFile({ name: "Х-3.MOV", size: 29e6, type: "video/quicktime" })).toBeNull();
    expect(validateVideoFile({ name: "a.mp4", size: 10, type: "" })).toBeNull();
  });
  it("отклоняет прочее", () => {
    expect(validateVideoFile({ name: "a.avi", size: 10, type: "video/x-msvideo" })).toMatch(/не поддерживается/);
    expect(validateVideoFile({ name: "a.mov", size: 0, type: "video/quicktime" })).toMatch(/пустой/);
    expect(validateVideoFile({ name: "a.mov", size: 2e9, type: "video/quicktime" }, 300)).toMatch(/больше/);
    expect(validateVideoFile({ name: "a.mov", size: 10, type: "text/plain" })).toMatch(/не похож/);
  });
});

describe("router", () => {
  it("hash ↔ route", () => {
    expect(parseHash("#/job/abc%20d")).toEqual({ name: "job", id: "abc d" });
    expect(parseHash("#/about")).toEqual({ name: "about" });
    expect(parseHash("")).toEqual({ name: "upload" });
    expect(routeHash({ name: "job", id: "x/y" })).toBe("#/job/x%2Fy");
  });
});
