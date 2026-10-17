// Демо-режим (?demo=1): UI работает без бэкенда — для питча и разработки.
// Кадры — синтетические SVG «вид с потолочной камеры» (никаких кадров из датасета).
import type { AskResponse, JobResult, SessionInfo, TimelinePoint, Violation } from "./types";
import type { Backend, UploadHandle } from "./client";

export const DEMO_DURATION = 74;

// ---------- синтетическая сцена ----------
interface SceneOpts {
  highlight?: "safe" | "person2" | "desk";
  label?: string;
  blurHeads?: boolean;
}

function between(t: number, a: number, b: number) {
  return t >= a && t <= b;
}

function lerp(a: number, b: number, k: number) {
  return a + (b - a) * Math.min(1, Math.max(0, k));
}

function personSvg(x: number, y: number, color: string, blur: boolean, rot = 0, bag = false): string {
  const head = blur
    ? `<circle cx="0" cy="0" r="17" fill="#c9b8a8" filter="url(#blur)"/><circle cx="0" cy="0" r="19" fill="none" stroke="#fff" stroke-opacity=".5" stroke-dasharray="3 3"/>`
    : `<circle cx="0" cy="0" r="16" fill="#3b2f2a"/>`;
  return `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${rot})">
    <ellipse cx="0" cy="4" rx="38" ry="22" fill="${color}"/>
    <ellipse cx="-30" cy="18" rx="9" ry="14" fill="${color}" opacity=".9"/>
    <ellipse cx="30" cy="18" rx="9" ry="14" fill="${color}" opacity=".9"/>
    ${head}
    ${bag ? `<rect x="26" y="24" width="30" height="22" rx="4" fill="#2f8f4e" stroke="#1d5e33"/>` : ""}
  </g>`;
}

/** SVG-кадр сцены в момент t (464×848, портрет как у камеры кассы). */
export function demoSceneSvg(t: number, opts: SceneOpts = {}): string {
  const W = 464;
  const H = 848;
  const safeOpen = between(t, 18, 31.5);
  const cashierAway = between(t, 21.6, 31.5);
  const p2 = between(t, 44.9, 53.5);
  // кассир: у стола; уходит вниз кадра и возвращается
  const cx = cashierAway ? lerp(232, 120, (t - 21.6) / 1.5) : 232;
  const cy = cashierAway ? lerp(560, 900, (t - 21.6) / 1.8) : t > 31.5 && t < 33 ? lerp(900, 560, (t - 31.5) / 1.5) : 560;
  const p2x = p2 ? lerp(420, 330, (t - 44.9) / 1.6) : 0;
  const p2y = p2 ? lerp(250, 470, (t - 44.9) / 1.6) : 0;
  const blur = opts.blurHeads ?? true;
  const ts = new Date(Date.UTC(2026, 2, 14, 10, 42, 0) + t * 1000);
  const osd = ts.toISOString().replace("T", " ").slice(0, 19);
  const hl =
    opts.highlight === "safe"
      ? `<rect x="22" y="572" width="130" height="150" rx="6" fill="none" stroke="#ff3b5c" stroke-width="4"/><rect x="22" y="544" width="${Math.max(90, (opts.label?.length ?? 6) * 8.5)}" height="24" rx="4" fill="#ff3b5c"/><text x="30" y="561" font-size="14" font-family="Arial" font-weight="700" fill="#fff">${opts.label ?? "сейф открыт"}</text>`
      : opts.highlight === "person2" && p2
        ? `<rect x="${(p2x - 50).toFixed(0)}" y="${(p2y - 40).toFixed(0)}" width="100" height="100" rx="6" fill="none" stroke="#ff3b5c" stroke-width="4"/><rect x="${(p2x - 50).toFixed(0)}" y="${(p2y - 66).toFixed(0)}" width="${Math.max(90, (opts.label?.length ?? 6) * 8.5)}" height="24" rx="4" fill="#ff3b5c"/><text x="${(p2x - 42).toFixed(0)}" y="${(p2y - 49).toFixed(0)}" font-size="14" font-family="Arial" font-weight="700" fill="#fff">${opts.label ?? "человек"}</text>`
        : opts.highlight === "desk"
          ? `<rect x="150" y="360" width="230" height="160" rx="6" fill="none" stroke="#ffb020" stroke-width="4"/>`
          : "";
  const cash = (x: number, y: number) =>
    `<g transform="translate(${x} ${y})"><rect width="44" height="22" rx="2" fill="#7fb77e" stroke="#4d7d4c"/><rect x="4" y="-5" width="44" height="22" rx="2" fill="#94c993" stroke="#4d7d4c"/></g>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
  <defs>
    <filter id="blur"><feGaussianBlur stdDeviation="5"/></filter>
    <pattern id="tiles" width="58" height="58" patternUnits="userSpaceOnUse"><rect width="58" height="58" fill="#d9d6cf"/><path d="M58 0H0V58" fill="none" stroke="#c4c0b7" stroke-width="2"/></pattern>
    <linearGradient id="vig" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#000" stop-opacity=".25"/><stop offset=".25" stop-color="#000" stop-opacity="0"/><stop offset=".8" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".3"/></linearGradient>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#tiles)"/>
  <!-- стеклянная перегородка и дверь -->
  <rect x="0" y="300" width="${W}" height="10" fill="#9fb7c9" opacity=".8"/>
  <rect x="0" y="0" width="${W}" height="300" fill="#cfd8de" opacity=".55"/>
  <g transform="translate(330 300)">${
    p2 || between(t, 43.8, 54.5)
      ? `<path d="M0 0 L95 -70" stroke="#6b8aa3" stroke-width="10" stroke-linecap="round"/>`
      : `<rect x="0" y="-5" width="110" height="10" fill="#6b8aa3"/>`
  }</g>
  <!-- стол кассира -->
  <rect x="150" y="360" width="230" height="160" rx="8" fill="#8a6f55" stroke="#6d5541" stroke-width="3"/>
  <rect x="180" y="380" width="96" height="64" rx="5" fill="#2b2f36"/><rect x="186" y="386" width="84" height="44" rx="2" fill="#4c6a8a"/>
  <rect x="298" y="384" width="60" height="44" rx="6" fill="#3a3d42"/><rect x="306" y="392" width="44" height="10" rx="2" fill="#69707a"/>
  <rect x="300" y="452" width="40" height="52" rx="5" fill="#e8e8e8" stroke="#999"/>
  ${cash(190, 462)}${t < 60 ? cash(240, 470) : ""}
  <!-- тумба и сейф -->
  <rect x="30" y="580" width="114" height="134" rx="4" fill="#9a9ea4" stroke="#6f7378" stroke-width="3"/>
  ${
    safeOpen
      ? `<rect x="40" y="592" width="94" height="110" fill="#3a3d42"/><rect x="48" y="610" width="78" height="10" fill="#7fb77e"/><path d="M144 590 L196 640 L196 726 L144 712 Z" fill="#b5b9be" stroke="#6f7378" stroke-width="3"/>`
      : `<circle cx="112" cy="648" r="11" fill="#6f7378"/><rect x="40" y="592" width="94" height="110" fill="none" stroke="#7f8388" stroke-width="2"/>`
  }
  <!-- люди (вид сверху; головы размыты) -->
  ${cy < 880 ? personSvg(cx, cy, "#2f4d7a", blur, 180) : ""}
  ${p2 ? personSvg(p2x, p2y, "#7a4a2f", blur, 150, false) : ""}
  ${between(t, 0, 8) ? personSvg(120, 160, "#6a6a6a", blur, 10) : ""}
  ${hl}
  <rect width="${W}" height="${H}" fill="url(#vig)"/>
  <rect x="10" y="10" width="236" height="26" rx="4" fill="#000" opacity=".55"/>
  <text x="18" y="28" font-size="14" font-family="Consolas, monospace" fill="#fff">CAM-01  ${osd}</text>
  <circle cx="440" cy="23" r="7" fill="#ff3b5c"/>
</svg>`;
}

function b64(s: string): string {
  // UTF-8 → base64 (кириллица в подписях)
  if (typeof btoa === "function") {
    const bytes = new TextEncoder().encode(s);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }
  return Buffer.from(s, "utf-8").toString("base64");
}

export function demoFrameBase64(t: number, opts: SceneOpts = {}): string {
  return b64(demoSceneSvg(t, opts));
}

export function demoSceneDataUri(t: number, opts: SceneOpts = {}): string {
  return `data:image/svg+xml;base64,${demoFrameBase64(t, opts)}`;
}

// ---------- кривая риска ----------
function demoTimeline(): TimelinePoint[] {
  const pts: TimelinePoint[] = [];
  let seed = 7;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const bump = (t: number, c: number, w: number, h: number) => h * Math.exp(-((t - c) ** 2) / (2 * w * w));
  for (let t = 0; t <= DEMO_DURATION + 1e-6; t += 0.5) {
    const r =
      0.06 +
      0.05 * rnd() +
      bump(t, 19.5, 1.4, 0.35) +
      bump(t, 25, 3.6, 0.82) +
      bump(t, 47.6, 2.4, 0.66) +
      bump(t, 5, 2, 0.12);
    pts.push([Number(t.toFixed(2)), Math.min(0.98, Number(r.toFixed(3)))]);
  }
  return pts;
}

// Синтетические записи стадий (формат meta.stages A03) — для панели «Полнота анализа» в демо.
const DEMO_STAGES_META = {
  ingest: { status: "ok", seconds: 3.1, required: true },
  detect: { status: "ok", seconds: 11.4, required: true },
  track: { status: "ok", seconds: 1.2 },
  pose: { status: "ok", seconds: 4.6 },
  scene: { status: "ok", seconds: 0.8 },
  rules: { status: "ok", seconds: 0.1, required: true },
  temporal: { status: "ok", seconds: 2.9 },
  vlm_verify: { status: "ok", seconds: 9.7 },
  fusion: { status: "ok", seconds: 0.6, required: true },
};

// Нормы в демо — заглушки: реальные ссылки приходят только из norms/catalog.yaml (A13).
const DEMO_RULE_NOTE =
  "[ДЕМО] Здесь показывается дословная цитата пункта нормы из norms/catalog.yaml (A13). В демо-режиме текст нормы не подставляется, чтобы не выдумывать пункты.";

export function buildDemoResult(jobId = "demo"): JobResult {
  const v1: Violation = {
    type: "Сейф открыт и оставлен без присмотра",
    type_id: "V03",
    timecode_sec: 24.6,
    t_start: 18.0,
    t_end: 31.5,
    confidence: 0.87,
    description:
      "На 00:18.0 дверца сейфа открыта. С 00:21.6 кассир выходит из зоны рабочего места, сейф остаётся открытым, рядом людей нет. На столе видны пачки банкнот. Это может указывать на нарушение порядка хранения ценностей — требуется проверка.",
    evidence_frames: [
      // Как у A12: в evidence_frames — чистые кадры; рамки подсветки — в meta.ui.evidence_overlays.shapes.
      { timestamp_sec: 24.6, image_base64: demoFrameBase64(24.6) },
      { timestamp_sec: 24.2, image_base64: demoFrameBase64(24.2) },
      { timestamp_sec: 29.0, image_base64: demoFrameBase64(29.0) },
    ],
    rules: [
      {
        document: "[ДЕМО] Нормативный документ из каталога норм",
        clause: null,
        url: "https://adilet.zan.kz/rus",
        quote: DEMO_RULE_NOTE,
        verified: false,
      },
    ],
    facts: [
      "Дверца сейфа открыта с 00:18.0 до 00:31.5",
      "С 00:21.6 до 00:31.5 в зоне рабочего места кассира никого нет",
      "На столе видны пачки банкнот",
    ],
    sources: ["rules", "vlm"],
    severity: "high",
  };
  const v2: Violation = {
    type: "Посторонний в кассовом помещении",
    type_id: "V01",
    timecode_sec: 47.8,
    t_start: 44.9,
    t_end: 53.5,
    confidence: 0.71,
    description:
      "На 00:44.9 открывается дверь кассового помещения, в зону кассира входит второй человек без инкассаторской сумки и находится у стола с наличными до 00:53.5. Наличие допуска по видео установить нельзя — требуется проверка.",
    evidence_frames: [
      { timestamp_sec: 47.8, image_base64: demoFrameBase64(47.8) },
      { timestamp_sec: 45.4, image_base64: demoFrameBase64(45.4) },
    ],
    rules: [
      {
        document: "[ДЕМО] Нормативный документ из каталога норм",
        clause: null,
        url: "https://adilet.zan.kz/rus",
        quote: DEMO_RULE_NOTE,
        verified: false,
      },
    ],
    facts: [
      "Дверь кассового помещения открыта с 00:43.8 до 00:54.5",
      "В зоне кассира одновременно два человека",
      "У второго человека не видно инкассаторской сумки",
    ],
    sources: ["rules", "temporal"],
    severity: "high",
  };
  return {
    job_id: jobId,
    status: "done",
    verdict: "violation",
    violations: [v1, v2],
    timeline: demoTimeline(),
    meta: {
      version: "demo",
      duration: DEMO_DURATION,
      file_name: "demo_kassa.mov",
      timings: { ingest: 3.1, detect: 11.4, track: 1.2, pose: 4.6, scene: 0.8, rules: 0.1, temporal: 2.9, vlm: 9.7, fusion: 0.6, total: 34.4 },
      models: [
        { name: "RT-DETRv2", license: "Apache-2.0", purpose: "детекция" },
        { name: "ByteTrack", license: "MIT", purpose: "трекинг" },
        { name: "RTMPose", license: "Apache-2.0", purpose: "поза" },
        { name: "SigLIP2 + temporal head", license: "Apache-2.0", purpose: "временная модель" },
        { name: "Qwen2.5-VL-7B (int4)", license: "Apache-2.0", purpose: "верификация" },
      ],
      warnings: ["Демо-режим: результат сгенерирован в браузере, сервер не используется."],
      device: "demo",
      complete: true,
      coverage: 1,
      required_stages: ["ingest", "detect", "rules", "fusion"],
      stages: DEMO_STAGES_META,
      ui: {
        evidence_overlays: [
          {
            violation_index: 0,
            type_id: "V03",
            timecode_sec: 24.6,
            facts: v1.facts,
            sources: { rules: 0.82, vlm: 0.74 },
            vlm: "confirm",
            shapes: [
              { kind: "box", key: "safe_open", label: "сейф открыт", color: "#e5484d", bbox: [0.06, 0.68, 0.43, 0.86] },
              { kind: "box", key: "cash", label: "наличные", color: "#f5a524", bbox: [0.4, 0.54, 0.56, 0.6] },
            ],
          },
          {
            violation_index: 1,
            type_id: "V01",
            timecode_sec: 47.8,
            facts: v2.facts,
            sources: { rules: 0.66, temporal: 0.58 },
            vlm: "unsure",
            shapes: [{ kind: "box", key: "person", label: "второй человек", color: "#e5484d", bbox: [0.6, 0.5, 0.83, 0.64] }],
          },
        ],
        review_hints: [
          {
            type_id: "V01",
            type: "Посторонний в кассовом помещении",
            timecode_sec: 4.5,
            t_start: 0,
            t_end: 8,
            reasons_ru: ["ниже порога уверенности"],
            facts: ["У стеклянной перегородки в зоне клиента находится человек"],
          },
        ],
      },
    },
    progress: 1,
    stage: "done",
  };
}

export function buildDemoClean(jobId = "demo-clean"): JobResult {
  return {
    job_id: jobId,
    status: "done",
    verdict: "no_violation",
    violations: [],
    timeline: demoTimeline().map(([t, r]) => [t, Math.min(0.28, r * 0.3)] as TimelinePoint),
    meta: {
      version: "demo",
      duration: DEMO_DURATION,
      file_name: "demo_clean.mov",
      timings: { total: 28.2 },
      complete: true,
      coverage: 1,
      required_stages: ["ingest", "detect", "rules", "fusion"],
      stages: DEMO_STAGES_META,
      ui: {
        review_hints: [
          {
            type_id: "V05",
            type: "Мобильный телефон при работе с ценностями",
            timecode_sec: 37.2,
            reasons_ru: ["ниже порога уверенности"],
            facts: ["Предмет в руке кассира, тип предмета не определён уверенно"],
          },
        ],
      },
    },
    progress: 1,
    stage: "done",
  };
}

// ---------- имитация бэкенда ----------
const DEMO_STAGES: [number, string][] = [
  [0.0, "queued"],
  [0.08, "ingest"],
  [0.22, "detect"],
  [0.45, "track"],
  [0.52, "pose"],
  [0.62, "scene"],
  [0.7, "rules"],
  [0.76, "temporal"],
  [0.84, "vlm"],
  [0.95, "fusion"],
];

const DEMO_PROCESS_MS = 7000;

export class DemoBackend implements Backend {
  kind = "demo" as const;
  private started = new Map<string, number>();
  private seq = 0;

  async init(): Promise<SessionInfo> {
    return { token: null, reachable: true, authenticated: true, features: { copilot: true }, version: "demo" };
  }

  async login(): Promise<SessionInfo> {
    return this.init();
  }

  async logout(): Promise<void> {}

  upload(file: File, onProgress: (f: number) => void): UploadHandle {
    let timer: ReturnType<typeof setInterval> | undefined;
    let rejectFn: (e: unknown) => void = () => {};
    const promise = new Promise<string>((resolve, reject) => {
      rejectFn = reject;
      let p = 0;
      timer = setInterval(() => {
        p = Math.min(1, p + 0.08);
        onProgress(p);
        if (p >= 1) {
          clearInterval(timer);
          const id = `demo-${Date.now().toString(36)}-${++this.seq}`;
          this.started.set(id, Date.now());
          void file;
          resolve(id);
        }
      }, 90);
    });
    return {
      promise,
      abort: () => {
        if (timer) clearInterval(timer);
        rejectFn(new DOMException("Загрузка отменена", "AbortError"));
      },
    };
  }

  async getJob(id: string): Promise<{ job: JobResult; raw: unknown }> {
    if (id === "demo-clean") {
      const job = buildDemoClean(id);
      return { job, raw: job };
    }
    const start = this.started.get(id);
    const elapsed = start === undefined ? Number.POSITIVE_INFINITY : Date.now() - start;
    if (elapsed < DEMO_PROCESS_MS) {
      const p = elapsed / DEMO_PROCESS_MS;
      let stage = "queued";
      for (const [k, s] of DEMO_STAGES) if (p >= k) stage = s;
      const job: JobResult = {
        job_id: id,
        status: "processing",
        verdict: null,
        violations: [],
        timeline: [],
        meta: {},
        progress: p,
        stage,
      };
      return { job, raw: job };
    }
    const job = buildDemoResult(id);
    return { job, raw: job };
  }

  async ask(id: string, question: string): Promise<AskResponse> {
    void id;
    await new Promise((r) => setTimeout(r, 650));
    const q = question.toLowerCase();
    if (q.includes("норм") || q.includes("пункт") || q.includes("закон"))
      return {
        answer:
          "Нормы подбираются только из каталога norms/catalog.yaml: для каждого типа указан документ, пункт и дословная цитата со ссылкой на adilet.zan.kz. В демо-режиме вместо текста нормы показана заглушка.",
        sources: ["norms/catalog.yaml"],
      };
    if (q.includes("почему") || q.includes("вывод"))
      return {
        answer:
          "Эпизод 1 (00:24.6): правила зафиксировали открытый сейф при отсутствии кассира в зоне рабочего места более 10 с, локальная VLM подтвердила это по кадрам. Эпизод 2 (00:47.8): второй человек в зоне кассира, без инкассаторской сумки. Это признаки возможных нарушений — решение остаётся за проверяющим.",
        sources: ["эпизод 1", "эпизод 2"],
      };
    if (q.includes("провер") || q.includes("вручную") || q.includes("что делать"))
      return {
        answer:
          "Рекомендую: 1) просмотреть 00:18–00:32 и убедиться, что сейф действительно не был закрыт; 2) сверить по журналу допуска, кто находился у кассы в 00:45–00:54; 3) уточнить у кассира причину ухода с рабочего места. Система не проверяет допуски, суммы и проводки.",
      };
    return {
      answer:
        "Я локальный ассистент проверяющего: отвечаю по результатам этого анализа (эпизоды, таймкоды, факты, нормы из каталога). Спросите, например: «Почему это нарушение?» или «Что проверить вручную?».",
    };
  }

  videoSource(): null {
    return null;
  }

  async copilotStatus() {
    return { available: true, disclaimer: "Синтетическое демо: ответы Copilot заготовлены заранее." };
  }

  async report(id: string): Promise<string> {
    const r = id === "demo-clean" ? buildDemoClean(id) : buildDemoResult(id);
    const lines = [
      "# Заключение (СИНТЕТИЧЕСКОЕ ДЕМО)",
      "",
      `Вердикт: ${r.verdict === "violation" ? "обнаружены признаки возможных нарушений" : "нарушение не обнаружено"}.`,
      "",
      ...r.violations.map((v, i) => `${i + 1}. ${v.type} — ${v.timecode_sec.toFixed(1)} с. ${v.description}`),
      "",
      "Результат — подсказка для проверяющего, не юридическое заключение.",
    ];
    return lines.join("\n");
  }
}

export function isDemoMode(search = typeof window !== "undefined" ? window.location.search : ""): boolean {
  const p = new URLSearchParams(search);
  const v = p.get("demo");
  return v !== null && v !== "0" && v !== "false";
}
