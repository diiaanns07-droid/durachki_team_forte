import { describe, expect, it } from "vitest";
import type { Backend, UploadHandle } from "../src/api/client";
import type { JobResult } from "../src/api/types";
import { BatchStore } from "../src/lib/batch";

function fakeBackend(results: Record<string, Partial<JobResult>>): Backend & { uploads: string[] } {
  let n = 0;
  const polls: Record<string, number> = {};
  const uploads: string[] = [];
  return {
    kind: "http",
    uploads,
    init: async () => ({ token: null, reachable: true, authenticated: true, features: {} }),
    login: async () => ({ token: null, reachable: true, authenticated: true, features: {} }),
    logout: async () => undefined,
    upload(file: File, onProgress: (f: number) => void): UploadHandle {
      uploads.push(file.name);
      onProgress(0.5);
      onProgress(1);
      return { promise: Promise.resolve(`job-${++n}`), abort: () => undefined };
    },
    async getJob(id: string) {
      polls[id] = (polls[id] ?? 0) + 1;
      const base: JobResult = { job_id: id, status: "processing", violations: [], timeline: [], meta: {}, stage: "detect", progress: 0.4 };
      const job = polls[id] >= 2 ? { ...base, ...results[id] } : base;
      return { job: job as JobResult, raw: job };
    },
    ask: async () => ({ answer: "" }),
    videoSource: () => null,
  };
}

const until = async (cond: () => boolean, ms = 2000) => {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error("timeout");
    await new Promise((r) => setTimeout(r, 5));
  }
};

describe("BatchStore", () => {
  it("грузит по очереди, отбраковывает неподходящие, доводит до вердикта", async () => {
    const be = fakeBackend({
      "job-1": { status: "done", verdict: "violation", violations: [{ type: "A", timecode_sec: 3, description: "", evidence_frames: [] }] },
      "job-2": { status: "done", verdict: "no_violation", meta: { partial: true } },
    });
    const store = new BatchStore(() => be, 10);
    const f = (name: string, type = "video/quicktime") => new File([new Uint8Array(16)], name, { type });
    store.add([f("a.mov"), f("b.mp4", "video/mp4"), f("c.avi", "video/x-msvideo")]);
    expect(store.getSnapshot().find((i) => i.name === "c.avi")?.state).toBe("error");
    await until(() => store.getSnapshot().filter((i) => i.state === "done").length === 2);
    const [a, b] = store.getSnapshot();
    expect(be.uploads).toEqual(["a.mov", "b.mp4"]);
    expect(a.verdict).toBe("violation");
    expect(a.violations).toEqual([{ type: "A", timecode_sec: 3 }]);
    expect(b.verdict).toBe("no_violation");
    expect(b.partial).toBe(true);
    expect(store.busy()).toBe(false);
    const sum = store.summary();
    expect(sum).toHaveLength(3);
    expect(JSON.stringify(sum)).not.toContain("image_base64");
    store.clearFinished();
    expect(store.getSnapshot()).toHaveLength(0);
  });
});
