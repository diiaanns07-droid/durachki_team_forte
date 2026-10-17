// Замер UI на ЖИВОМ сервере (для L2). Сценарий как у проверяющего: вход → загрузка → статус → вердикт →
// переход к событию (PTS) → кадр → норма. В отчёт пишутся ТОЛЬКО числа и коды (ни кадров, ни описаний сцен).
//
//   E2E_BASE_URL=http://127.0.0.1:8000/ E2E_KEY=<ключ> E2E_VIDEOS=<папка|файл> \
//   E2E_CHECK_PTS=1 E2E_OUT=e2e-report.json [E2E_LABELS=labels.csv] node scripts/e2e-real.mjs
//
// E2E_VIDEO=<файл> — старый вариант (один ролик). E2E_SHOTS=<папка> — скриншоты (ЛОКАЛЬНО, не коммитить и
// не отправлять в облако). E2E_TIMEOUT_S — лимит ожидания результата на ролик (по умолчанию 300).
// E2E_LABELS — CSV «file,label,second» (label: violation|no_violation|1|0) — РАЗВЕДОЧНАЯ сверка: эти ролики уже
// использовались при настройке, независимым тестом их считать нельзя.
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { launch, loadPlaywright } from "./pw.mjs";

const BASE = process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000/";
const KEY = process.env.E2E_KEY;
const SRC = process.env.E2E_VIDEOS ?? process.env.E2E_VIDEO;
const OUT = process.env.E2E_OUT ?? "e2e-real-report.json";
const SHOTS = process.env.E2E_SHOTS;
const CHECK_PTS = process.env.E2E_CHECK_PTS === "1";
const TIMEOUT_S = Number(process.env.E2E_TIMEOUT_S ?? 300);
const LABELS = process.env.E2E_LABELS;
const TOL_S = Number(process.env.E2E_TIMECODE_TOL_S ?? 3);
if (!KEY || !SRC) {
  console.error("Нужны E2E_KEY и E2E_VIDEOS (папка или файл)");
  process.exit(2);
}

const videos = statSync(SRC).isDirectory()
  ? readdirSync(SRC)
      .filter((f) => /\.(mov|mp4)$/i.test(f))
      .sort()
      .map((f) => path.join(SRC, f))
  : [SRC];

/** Нормализация имени: кириллическая Х → латинская X, регистр, без расширения. */
const normName = (f) => path.basename(f).replace(/\.[^.]+$/, "").replace(/[Хх]/g, "x").toLowerCase().trim();

function loadLabels(file) {
  if (!file || !existsSync(file)) return new Map();
  const rows = readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean);
  const m = new Map();
  for (const r of rows.slice(1)) {
    const [f, lab, sec] = r.split(/[;,]/).map((x) => x.trim());
    if (!f) continue;
    const v = /^(1|violation|нарушение|true)$/i.test(lab ?? "");
    m.set(normName(f), { violation: v, second: sec ? Number(sec) : null });
  }
  return m;
}
const labels = loadLabels(LABELS);

const pw = await loadPlaywright();
const browser = await launch(pw);
const context = await browser.newContext({ viewport: { width: 1366, height: 900 }, acceptDownloads: true });
const report = { base: BASE, started_at: new Date().toISOString(), check_pts: CHECK_PTS, videos: [], summary: {} };

const page = await context.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e).slice(0, 200)));

// --- вход (ключ вводится в форму, никуда не сохраняется) ---
await page.goto(BASE);
await page.getByText("Вход для проверяющего").waitFor({ timeout: 15000 });
await page.getByLabel("Ключ доступа").fill(KEY);
await page.getByRole("button", { name: "Войти" }).click();
await page.getByText("Перетащите видео сюда").waitFor({ timeout: 15000 });
console.log(`✓ вход; роликов: ${videos.length}`);

for (const file of videos) {
  const rec = { file: path.basename(file), size_mb: +(statSync(file).size / 1048576).toFixed(1), ok: false };
  report.videos.push(rec);
  const errBefore = pageErrors.length;
  try {
    await page.goto(`${BASE}#/`);
    await page.getByText("Перетащите видео сюда").waitFor({ timeout: 15000 });
    // Playwright (≥1.56) молча не выставляет файл, если в ПУТИ кириллица (имена «Х-3.mov» в датасете) —
    // такие файлы передаём буфером с исходным именем (до ~50 МБ на файл).
    const base = path.basename(file);
    const fileArg = /^[\x20-\x7E]+$/.test(file)
      ? file
      : { name: base, mimeType: /\.mp4$/i.test(base) ? "video/mp4" : "video/quicktime", buffer: readFileSync(file) };
    await page.locator('input[type="file"]').setInputFiles(fileArg);
    const t0 = Date.now();
    await page.getByRole("button", { name: "Начать проверку" }).click();
    await page.waitForFunction(() => location.hash.startsWith("#/job/"), null, { timeout: 120000 });
    rec.upload_s = +((Date.now() - t0) / 1000).toFixed(2);
    rec.job_id = decodeURIComponent(page.url().split("#/job/")[1] ?? "");
    const banner = page.locator("section.bg-gradient-to-br h1");
    await page.waitForFunction(
      () => /Обнаружены признаки|Нарушение не обнаружено|Анализ не завершён|Анализ неполный|Тестовый результат/.test(document.querySelector("section.bg-gradient-to-br h1")?.textContent ?? "") ||
        document.body.innerText.includes("Анализ не завершён"),
      null,
      { timeout: TIMEOUT_S * 1000 },
    );
    rec.time_to_result_s = +((Date.now() - t0) / 1000).toFixed(2);
    rec.banner = ((await banner.count()) ? await banner.first().innerText() : "Анализ не завершён").trim();
    rec.flags = (await page.locator("section.bg-gradient-to-br span.uppercase").allInnerTexts()).map((s) => s.trim());

    // Ответ сервера — из контекста страницы (cookie-сессия); в отчёт — только коды и числа.
    const job = await page.evaluate(async (id) => (await fetch(`ui/v1/jobs/${encodeURIComponent(id)}`, { credentials: "same-origin" })).json(), rec.job_id);
    const meta = job.meta ?? {};
    const stages = meta.stages && typeof meta.stages === "object" ? meta.stages : {};
    rec.status = job.status;
    rec.verdict = job.verdict ?? null;
    rec.complete = meta.complete ?? null;
    rec.coverage = typeof meta.coverage === "number" ? meta.coverage : null;
    rec.mock = meta.mock === true;
    rec.server_total_s = meta.timings?.total ?? null;
    rec.stage_s = Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, v?.seconds ?? null]));
    rec.stage_status = Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, v?.status ?? null]));
    const vram = Object.values(stages).map((v) => v?.vram_peak_mb).filter((x) => typeof x === "number");
    rec.vram_peak_mb = vram.length ? Math.max(...vram) : null;
    rec.t0_pts_sec = meta.video?.t0_pts_sec ?? meta.t0_pts_sec ?? null;
    const vs = Array.isArray(job.violations) ? job.violations : [];
    rec.events = vs.map((v) => ({
      type_id: v.type_id ?? null,
      t: v.timecode_sec,
      t_start: v.t_start ?? null,
      t_end: v.t_end ?? null,
      conf: v.confidence ?? null,
      n_frames: (v.evidence_frames ?? []).length,
      frame_dt: (v.evidence_frames ?? []).map((f) => +(f.timestamp_sec - v.timecode_sec).toFixed(3)),
      n_rules: (v.rules ?? []).length,
      rules_verified: (v.rules ?? []).filter((r) => r.verified === true).length,
      rules_no_clause: (v.rules ?? []).filter((r) => !r.clause).length,
    }));

    // Переход к событию в UI: клик по таймкоду → часы плеера показывают тот же момент.
    if (vs.length) {
      await page.locator("#episode-1 header button.btn-brand").click();
      await page.waitForTimeout(400);
      rec.ui_clock_after_jump = await page.locator("div.font-mono.text-sm span.font-semibold").first().innerText();
    }
    const vstate = await page.evaluate(() => {
      const v = document.querySelector("video");
      return v ? { readyState: v.readyState, error: v.error?.code ?? null, src: v.currentSrc.startsWith("blob:") ? "local" : "server" } : null;
    });
    rec.video_local = vstate;

    // Серверный видео-маршрут (как после перезагрузки) + сверка PTS по кадрам-доказательствам.
    const p2 = await context.newPage();
    const vresp = p2.waitForResponse((r) => r.url().includes("/video"), { timeout: 20000 }).catch(() => null);
    await p2.goto(`${BASE}#/job/${encodeURIComponent(rec.job_id)}`);
    const vr = await vresp;
    rec.video_route = vr ? { status: vr.status(), content_type: vr.headers()["content-type"] ?? null } : null;
    rec.video_server_ready = await p2
      .locator("video")
      .evaluate((v) => new Promise((res) => (v.readyState >= 1 ? res(v.readyState) : ((v.onloadedmetadata = () => res(v.readyState)), (v.onerror = () => res(-1)), setTimeout(() => res(v.readyState), 10000)))))
      .catch(() => null);
    if (CHECK_PTS && vs.length && rec.video_server_ready >= 1) {
      const src = await p2.locator("video").evaluate((v) => v.currentSrc);
      const frames = vs.flatMap((v, i) => (v.evidence_frames ?? []).map((f) => ({ i, ts: f.timestamp_sec, b64: f.image_base64 })));
      rec.pts_check = await p2.evaluate(
        async ({ src, frames, t0 }) => {
          const DELTAS = [-1, -0.5, -0.2, -0.1, -0.033, 0, 0.033, 0.1, 0.2, 0.5, 1];
          if (t0) DELTAS.push(t0);
          const S = 64;
          const gray = (img) => {
            const c = document.createElement("canvas");
            c.width = S;
            c.height = S;
            const g = c.getContext("2d");
            g.drawImage(img, 0, 0, S, S);
            const d = g.getImageData(0, 0, S, S).data;
            const out = new Float32Array(S * S);
            for (let k = 0; k < S * S; k++) out[k] = 0.299 * d[4 * k] + 0.587 * d[4 * k + 1] + 0.114 * d[4 * k + 2];
            return out;
          };
          const mad = (a, b) => {
            let s = 0;
            for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
            return s / a.length;
          };
          const v = document.createElement("video");
          v.muted = true;
          v.preload = "auto";
          v.src = src;
          await new Promise((r, j) => ((v.onloadeddata = r), (v.onerror = j), setTimeout(j, 15000)));
          const seek = (t) => new Promise((r) => ((v.onseeked = r), (v.currentTime = Math.max(0, Math.min(t, v.duration - 0.02)))));
          const res = [];
          for (const f of frames.slice(0, 30)) {
            const img = new Image();
            img.src = f.b64.startsWith("data:") ? f.b64 : `data:image/jpeg;base64,${f.b64}`;
            await img.decode();
            const ref = gray(img);
            const scores = [];
            for (const d of DELTAS) {
              await seek(f.ts + d);
              scores.push([d, +mad(gray(v), ref).toFixed(2)]);
            }
            scores.sort((a, b) => a[1] - b[1]);
            const at0 = scores.find((x) => x[0] === 0)[1];
            res.push({ ev: f.i, ts: f.ts, best_delta: scores[0][0], best_mad: scores[0][1], mad_at_0: at0 });
          }
          return res;
        },
        { src, frames, t0: typeof rec.t0_pts_sec === "number" && rec.t0_pts_sec > 0 ? rec.t0_pts_sec : 0 },
      );
    }
    if (SHOTS) await p2.screenshot({ path: path.join(SHOTS, `${normName(file)}.png`), fullPage: true });
    await p2.close();

    const lab = labels.get(normName(file));
    if (lab) {
      rec.label = lab;
      rec.verdict_correct = rec.status === "done" ? (rec.verdict === "violation") === lab.violation : null;
      if (lab.violation && lab.second != null && rec.events.length) {
        const dts = rec.events.map((e) => Math.abs(e.t - lab.second));
        rec.timecode_abs_err_s = +Math.min(...dts).toFixed(2);
        rec.timecode_hit = rec.timecode_abs_err_s <= TOL_S;
      }
    }
    rec.js_errors = pageErrors.length - errBefore;
    rec.ok = true;
    console.log(`✓ ${rec.file}: ${rec.banner} · ${rec.events.length} соб. · ${rec.time_to_result_s} с · видео ${rec.video_route?.status ?? "—"}`);
  } catch (e) {
    rec.error = String(e.message ?? e).slice(0, 300);
    rec.js_errors = pageErrors.length - errBefore;
    console.error(`✗ ${rec.file}: ${rec.error}`);
  }
}

// --- сводка ---
const done = report.videos.filter((r) => r.ok);
const lat = done.map((r) => r.time_to_result_s).sort((a, b) => a - b);
const q = (p) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null);
const pts = done.flatMap((r) => r.pts_check ?? []);
const labeled = done.filter((r) => r.label && r.verdict_correct !== null && r.verdict_correct !== undefined);
report.summary = {
  n: report.videos.length,
  ok: done.length,
  latency_s: { p50: q(0.5), p90: q(0.9), max: lat.at(-1) ?? null },
  video_server_playable: done.filter((r) => r.video_server_ready >= 1).length,
  js_errors: report.videos.reduce((s, r) => s + (r.js_errors ?? 0), 0),
  vram_peak_mb_max: Math.max(0, ...done.map((r) => r.vram_peak_mb ?? 0)) || null,
  pts: pts.length
    ? {
        frames: pts.length,
        best_delta_is_0: pts.filter((x) => x.best_delta === 0).length,
        best_delta_abs_le_0_1: pts.filter((x) => Math.abs(x.best_delta) <= 0.1).length,
        best_delta_hist: pts.reduce((h, x) => ((h[x.best_delta] = (h[x.best_delta] ?? 0) + 1), h), {}),
      }
    : null,
  exploratory_labels: labeled.length
    ? {
        note: "разведочно: эти ролики уже использовались при настройке — не независимый тест",
        n: labeled.length,
        fp: labeled.filter((r) => !r.label.violation && r.verdict === "violation").length,
        fn: labeled.filter((r) => r.label.violation && r.verdict !== "violation").length,
        timecode_hit: labeled.filter((r) => r.timecode_hit).length,
        timecode_tol_s: TOL_S,
      }
    : null,
};
writeFileSync(OUT, JSON.stringify(report, null, 2));
console.log("\nСводка:", JSON.stringify(report.summary, null, 1));
console.log(`Отчёт: ${OUT} (только числа и коды; кадров и описаний нет)`);
await browser.close();
try {
  assert.equal(done.length, report.videos.length, "не все ролики прошли сценарий");
} catch (e) {
  console.error("✗", e.message);
  process.exitCode = 1;
}
