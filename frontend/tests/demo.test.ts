import { describe, expect, it } from "vitest";
import { buildDemoClean, buildDemoResult, demoSceneSvg, DemoBackend, isDemoMode } from "../src/api/demo";
import { normalizeJob } from "../src/api/normalize";

describe("синтетическое демо", () => {
  it("2 нарушения, timeline, кадры", () => {
    const r = buildDemoResult();
    expect(r.verdict).toBe("violation");
    expect(r.violations).toHaveLength(2);
    expect(r.timeline.length).toBeGreaterThan(50);
    for (const v of r.violations) {
      expect(v.evidence_frames.length).toBeGreaterThan(0);
      expect(v.rules?.[0].url.startsWith("https://adilet.zan.kz")).toBe(true);
    }
    // проходит через тот же нормализатор, что и ответ сервера
    expect(normalizeJob(JSON.parse(JSON.stringify(r))).violations).toHaveLength(2);
  });
  it("нормы в демо помечены как заглушки", () => {
    const r = buildDemoResult();
    expect(r.violations.every((v) => v.rules?.every((x) => x.document.includes("[ДЕМО]")))).toBe(true);
  });
  it("чистый результат", () => {
    expect(buildDemoClean().verdict).toBe("no_violation");
  });
  it("SVG-сцена валидна", () => {
    const s = demoSceneSvg(24.6, { highlight: "safe", label: "сейф" });
    expect(s.startsWith("<svg")).toBe(true);
    expect(s).toContain("</svg>");
  });
  it("isDemoMode", () => {
    expect(isDemoMode("?demo=1")).toBe(true);
    expect(isDemoMode("?demo=0")).toBe(false);
    expect(isDemoMode("")).toBe(false);
  });
  it("DemoBackend: известный демо-id сразу готов", async () => {
    const b = new DemoBackend();
    const { job } = await b.getJob("demo");
    expect(job.status).toBe("done");
    expect((await b.init()).authenticated).toBe(true);
  });
});
