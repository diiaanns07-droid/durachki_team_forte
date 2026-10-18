// Мок-e2e реального HTTP-пути (НЕ демо): бэкенд A02 подменён page.route с синтетическими ответами.
// Проверяет: вход по ключу (cookie-сессия) → загрузка → поллинг → результат → перемотка по таймкоду,
// Copilot скрывается при 501, failed не показывается зелёным, в URL нет ключа.
// Запуск: npm run build && npm run e2e:mock
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

const shots = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../screenshots");
mkdirSync(shots, { recursive: true });

const KEY = "ks_test_key_e2e";
const CSRF = "csrf_synthetic_token_123";
let csrfRejected = 0;
const svg = (label) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="232" height="424"><rect width="232" height="424" fill="#cfd8de"/><text x="20" y="200" font-size="20">${label}</text></svg>`,
  ).toString("base64");

function doneJob(id) {
  return {
    job_id: id,
    status: "done",
    verdict: "violation",
    violations: [
      {
        type: "Синтетический тип A",
        timecode_sec: 12.4,
        description: "Синтетическое описание для e2e.",
        evidence_frames: [{ timestamp_sec: 12.4, image_base64: svg("A") }],
        type_id: "V99",
        confidence: 0.66,
        t_start: 10,
        t_end: 15,
        rule: { document: "Синтетический документ", clause: "п. 0", url: "https://adilet.zan.kz/rus", quote: "синтетическая цитата", verified: false },
      },
    ],
    progress: 1,
    stage: "done",
    timeline: [
      [0, 0.1],
      [12.4, 0.8],
      [30, 0.1],
    ],
    meta: { version: "e2e", duration_sec: 30, timings: { total: 3 } },
  };
}

const pw = await loadPlaywright();
const { server, url } = await startPreview(4182);
const browser = await launch(pw);
const failures = [];
const step = async (name, fn) => {
  try {
    await fn();
    console.log("✓", name);
  } catch (e) {
    failures.push(name);
    console.error("✗", name, "\n ", e.message);
  }
};

try {
  // ---------- сценарий 1: полный путь ----------
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
  const page = await ctx.newPage();
  let loggedIn = false;
  let polls = 0;
  let sawKeyInUrl = false;
  page.on("request", (r) => {
    if (r.url().includes(KEY)) sawKeyInUrl = true;
  });
  await page.route("**/ui/v1/**", async (route) => {
    const req = route.request();
    const u = new URL(req.url());
    const p = u.pathname;
    if (p.endsWith("/ui/v1/session")) {
      if (req.method() === "POST") {
        let body = {};
        try {
          body = JSON.parse(req.postData() || "{}");
        } catch {}
        const ok = body.api_key === KEY || body.password === KEY; // как в A02: {api_key} или {password}
        loggedIn = ok;
        return route.fulfill({
          status: ok ? 200 : 401,
          contentType: "application/json",
          headers: ok ? { "set-cookie": "ks_ui=1; HttpOnly; Path=/; SameSite=Strict" } : {},
          body: JSON.stringify(ok ? { authenticated: true, expires_at: 0, csrf_token: CSRF } : { detail: "invalid credentials" }),
        });
      }
      return route.fulfill({
        status: loggedIn ? 200 : 401,
        contentType: "application/json",
        body: JSON.stringify(loggedIn ? { authenticated: true, csrf_token: CSRF } : { authenticated: false, password_login: false }),
      });
    }
    if (!loggedIn) return route.fulfill({ status: 401, body: "{}" });
    // A02: мутирующие запросы /ui/v1/* требуют X-CSRF-Token
    if (req.method() === "POST" && req.headers()["x-csrf-token"] !== CSRF) {
      csrfRejected++;
      return route.fulfill({ status: 403, contentType: "application/json", body: '{"detail":"missing or invalid X-CSRF-Token"}' });
    }
    if (p.endsWith("/ui/v1/jobs")) {
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ jobs: [{ job_id: "old-1", status: "done", stage: "done", filename: "synthetic_old.mov", created_at: 1792000000, finished_at: 1792000030 }] }) });
    }
    if (p.endsWith("/ui/v1/analyze")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ job_id: "e2e-1" }) });
    if (p.endsWith("/ui/v1/jobs/e2e-1")) {
      polls++;
      const body =
        polls === 1
          ? { job_id: "e2e-1", status: "processing", stage: "queued", progress: 0, queue_position: 2, elapsed_s: 0.4 }
          : polls < 4
            ? { job_id: "e2e-1", status: "processing", stage: polls === 2 ? "detect" : "vlm", progress: polls * 0.3, queue_position: 0, elapsed_s: polls * 1.5 }
            : doneJob("e2e-1");
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    }
    if (p.endsWith("/ask")) return route.fulfill({ status: 501, contentType: "application/json", body: JSON.stringify({ detail: "copilot disabled" }) });
    if (p.endsWith("/video")) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ status: 404, body: "{}" });
  });

  await page.goto(`${url}#/`);
  await step("нет демо-ленты в реальном режиме", async () => {
    assert.equal(await page.getByText("Синтетическое демо", { exact: false }).count(), 0);
  });
  await step("форма входа при отсутствии сессии", async () => {
    await page.getByText("Вход для проверяющего").waitFor({ timeout: 5000 });
    await page.screenshot({ path: path.join(shots, "11-login.png") });
  });
  await step("неверный ключ → ошибка", async () => {
    await page.getByLabel("Ключ доступа").fill("wrong");
    await page.getByRole("button", { name: "Войти" }).click();
    await page.getByText("Неверный ключ доступа.").waitFor({ timeout: 5000 });
  });
  await step("верный ключ → зона загрузки; ключ не в localStorage", async () => {
    await page.getByLabel("Ключ доступа").fill(KEY);
    await page.getByRole("button", { name: "Войти" }).click();
    await page.getByText("Перетащите видео сюда").waitFor({ timeout: 5000 });
    const stored = await page.evaluate(() => JSON.stringify(Object.assign({}, localStorage)));
    assert.ok(!stored.includes(KEY), "ключ найден в localStorage");
  });
  await step("загрузка → обработка (реальный progress) → результат", async () => {
    await page.locator('input[type="file"]').setInputFiles({ name: "synthetic.mov", mimeType: "video/quicktime", buffer: Buffer.alloc(2048, 1) });
    await page.getByRole("button", { name: "Начать проверку" }).click();
    await page.getByText("позиция в очереди: 2", { exact: false }).waitFor({ timeout: 8000 });
    await page.getByText("Идёт анализ записи").waitFor({ timeout: 8000 });
    await page.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 15000 });
    assert.ok(polls >= 3, `опросов: ${polls}`);
  });
  await step("клик по таймкоду → перемотка", async () => {
    await page.locator("#episode-1 header button.btn-brand").click();
    await page.waitForTimeout(300);
    const txt = await page.locator("div.font-mono.text-sm").first().innerText();
    assert.ok(txt.includes("00:12.4"), `время плеера: ${txt}`);
  });
  await step("горячие клавиши: ← назад, N — к следующему эпизоду", async () => {
    await page.locator("body").click({ position: { x: 5, y: 300 } });
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("ArrowLeft");
    await page.waitForTimeout(150);
    const back = await page.locator("div.font-mono.text-sm").first().innerText();
    assert.ok(back.startsWith("00:00.0"), `после ←×3: ${back}`);
    await page.keyboard.press("n");
    await page.waitForTimeout(150);
    const fwd = await page.locator("div.font-mono.text-sm").first().innerText();
    assert.ok(fwd.startsWith("00:12.4"), `после N: ${fwd}`);
  });
  await step("кадр открывается в лайтбоксе, норма видна", async () => {
    await page.locator("article button[aria-label^='Открыть кадр']").first().click();
    await page.getByRole("dialog", { name: "Кадр-доказательство" }).waitFor({ timeout: 3000 });
    await page.keyboard.press("Escape");
    await page.getByText("Синтетический документ").waitFor({ timeout: 3000 });
    await page.getByText("цитата не сверена").waitFor({ timeout: 3000 });
  });
  await step("Copilot: 501 → панель скрыта", async () => {
    // U08 заменил общие подсказки («Почему это нарушение?») на вопросы по эпизодам отчёта; проверка та же:
    // после 501 от /ask панель скрывается.
    await page.getByRole("button", { name: /^Объяснить эпизод 1\b/ }).first().click();
    await page.waitForTimeout(800);
    assert.equal(await page.getByText("Copilot проверяющего").count(), 0);
  });
  await step("ключ ни разу не попал в URL запросов", async () => {
    assert.equal(sawKeyInUrl, false);
  });
  await step("POST-запросы шли с X-CSRF-Token (ни одного 403)", async () => {
    assert.equal(csrfRejected, 0, `отклонено CSRF: ${csrfRejected}`);
  });
  await step("серверная история (GET /ui/v1/jobs) видна на главной", async () => {
    await page.goto(`${url}#/`);
    await page.getByText("synthetic_old.mov").waitFor({ timeout: 5000 });
  });
  await ctx.close();

  // ---------- сценарий 1б: пакетная проверка (несколько файлов) ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    let seq = 0;
    const polls = {};
    await pg.route("**/ui/v1/**", (route) => {
      const req = route.request();
      const p = new URL(req.url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      if (req.method() === "POST" && req.headers()["x-csrf-token"] !== CSRF) return route.fulfill({ status: 403, body: "{}" });
      if (p.endsWith("/analyze")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ job_id: `b-${++seq}` }) });
      const m = p.match(/\/jobs\/(b-\d+)$/);
      if (m) {
        polls[m[1]] = (polls[m[1]] ?? 0) + 1;
        const body =
          polls[m[1]] < 2
            ? { job_id: m[1], status: "processing", stage: "detect", progress: 0.5 }
            : m[1] === "b-1"
              ? doneJob("b-1")
              : { job_id: m[1], status: "done", verdict: "no_violation", violations: [], meta: {} };
        return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
      }
      if (p.endsWith("/ui/v1/jobs")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"jobs":[]}' });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/`);
    await step("пакет: 3 файла (1 неподходящий) → 2 задачи → вердикты", async () => {
      await pg.getByText("Перетащите видео сюда").waitFor({ timeout: 5000 });
      await pg.locator('input[type="file"]').setInputFiles([
        { name: "batch_1.mov", mimeType: "video/quicktime", buffer: Buffer.alloc(1024, 1) },
        { name: "batch_2.mp4", mimeType: "video/mp4", buffer: Buffer.alloc(1024, 2) },
        { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("x") },
      ]);
      await pg.getByText("Пакетная проверка").first().waitFor({ timeout: 5000 });
      await pg.getByText("признаки нарушений: 1").waitFor({ timeout: 15000 });
      await pg.getByText("не обнаружено · полнота не подтверждена").waitFor({ timeout: 15000 }); // meta без complete → не зелёный
      await pg.getByText("не поддерживается", { exact: false }).first().waitFor({ timeout: 3000 });
      assert.equal(seq, 2, `загрузок: ${seq}`);
      await pg.waitForTimeout(400);
      await pg.screenshot({ path: path.join(shots, "14-batch.png") });
    });
    await step("пакет: «Открыть отчёт» ведёт на результат", async () => {
      await pg.getByRole("link", { name: "Открыть отчёт" }).first().click();
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 8000 });
    });
    await c.close();
  }

  // ---------- сценарий 1в: copilot status=unavailable, report .md, истечение сессии ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
    const pg = await c.newPage();
    let jobPolls = 0;
    let relogged = false;
    await pg.route("**/ui/v1/**", (route) => {
      const req = route.request();
      const p = new URL(req.url()).pathname;
      if (p.endsWith("/session")) {
        if (req.method() === "POST") {
          relogged = true;
          return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
        }
        return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      }
      if (p.endsWith("/copilot/status")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"enabled":true,"available":false,"reason":"unavailable"}' });
      if (p.endsWith("/jobs/c1/report")) return route.fulfill({ status: 200, contentType: "text/markdown; charset=utf-8", body: "# Синтетическое заключение" });
      if (p.endsWith("/jobs/c1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(doneJob("c1")) });
      if (p.endsWith("/jobs/x1")) {
        jobPolls++;
        if (relogged) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(doneJob("x1")) });
        return jobPolls < 2
          ? route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ job_id: "x1", status: "processing", stage: "detect", progress: 0.2 }) })
          : route.fulfill({ status: 401, contentType: "application/json", body: '{"detail":"UI session required"}' });
      }
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/c1`);
    await step("copilot status available=false → панель не показана", async () => {
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 6000 });
      await pg.waitForTimeout(500);
      assert.equal(await pg.getByText("Copilot проверяющего").count(), 0);
    });
    await step("«Заключение .md» скачивается", async () => {
      const [dl] = await Promise.all([pg.waitForEvent("download", { timeout: 5000 }), pg.getByRole("button", { name: "Заключение .md" }).click()]);
      assert.ok(dl.suggestedFilename().endsWith(".md"), dl.suggestedFilename());
    });
    await pg.goto(`${url}#/job/x1`);
    await step("сессия истекла во время обработки → вход на странице задачи → опрос продолжается", async () => {
      await pg.getByText("Вход для проверяющего").waitFor({ timeout: 8000 });
      await pg.getByLabel("Ключ доступа").fill(KEY);
      await pg.getByRole("button", { name: "Войти" }).click();
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 8000 });
      assert.ok(pg.url().includes("#/job/x1"), "URL задачи сохранён");
    });
    await c.close();
  }

  // ---------- сценарий 1г: /ui/v1 выключен (404) → вход по API-ключу с проверкой ключа ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    let apiKeyHeaderSeen = false;
    await pg.route("**/ui/v1/**", (route) => route.fulfill({ status: 404, contentType: "application/json", body: '{"detail":"UI access disabled"}' }));
    await pg.route("**/api/v1/**", (route) => {
      const req = route.request();
      const ok = req.headers()["x-api-key"] === KEY;
      if (ok) apiKeyHeaderSeen = true;
      if (!ok) return route.fulfill({ status: 401, contentType: "application/json", body: '{"detail":"invalid or missing API key"}' });
      const p = new URL(req.url()).pathname;
      if (p.endsWith("/api/v1/jobs")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"jobs":[]}' });
      if (p.endsWith("/jobs/k1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(doneJob("k1")) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/`);
    await step("UI выключен: неверный ключ отвергается, верный — работает через /api/v1", async () => {
      await pg.getByText("Вход для проверяющего").waitFor({ timeout: 5000 });
      await pg.getByLabel("Ключ доступа").fill("wrong");
      await pg.getByRole("button", { name: "Войти" }).click();
      await pg.getByText("Неверный ключ доступа.").waitFor({ timeout: 5000 });
      await pg.getByLabel("Ключ доступа").fill(KEY);
      await pg.getByRole("button", { name: "Войти" }).click();
      await pg.getByText("Перетащите видео сюда").waitFor({ timeout: 5000 });
      await pg.goto(`${url}#/job/k1`);
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 8000 }).catch(async () => {
        // после перехода по URL память вкладки сохраняется (SPA), но goto перезагружает страницу — ключ не хранится
        await pg.getByText("Не удалось получить результат").waitFor({ timeout: 5000 });
      });
      assert.ok(apiKeyHeaderSeen, "X-API-Key не отправлялся");
      const stored = await pg.evaluate(() => JSON.stringify(Object.assign({}, localStorage)));
      assert.ok(!stored.includes(KEY), "ключ найден в localStorage");
    });
    await c.close();
  }

  // ---------- сценарий 1д: последовательный просмотр эпизодов (синтетическое демо, виртуальный плеер) ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    await pg.goto(`${url}?demo=1#/job/demo`);
    const clock = () => pg.locator("div.font-mono.text-sm span.font-semibold").first().innerText();
    await step("«Смотреть эпизод» проигрывает с запасом до начала", async () => {
      await pg.getByRole("button", { name: "Смотреть эпизод" }).click();
      await pg.waitForTimeout(150);
      const a = await clock();
      assert.ok(a.startsWith("00:16"), `старт ${a} (ожидался t_start−2 = 00:16)`);
      await pg.waitForTimeout(1200);
      const b = await clock();
      assert.notEqual(a, b, "время не идёт — воспроизведение не началось");
    });
    await step("«›» — следующий эпизод: пауза на кадре таймкода, счётчик «2 из 2»", async () => {
      await pg.getByRole("button", { name: "Следующий эпизод" }).click();
      await pg.waitForTimeout(200);
      const a = await clock();
      assert.equal(a, "00:47.8");
      await pg.waitForTimeout(900);
      assert.equal(await clock(), "00:47.8", "после перехода воспроизведение должно стоять на паузе");
      await pg.getByText("Эпизод 2 из 2", { exact: false }).waitFor({ timeout: 2000 });
    });
    await step("отметка «просмотрено» переживает перезагрузку", async () => {
      await pg.getByRole("button", { name: "Отметить просмотренным" }).click();
      await pg.getByText("просмотрено 1 из 2", { exact: false }).waitFor({ timeout: 2000 });
      await pg.reload();
      await pg.getByText("просмотрено 1 из 2", { exact: false }).waitFor({ timeout: 5000 });
    });
    await c.close();
  }

  // ---------- сценарий 1е: обрыв сети (пауза растёт, опрос не обрывается) и повторная отправка файла ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    let n1 = 0;
    let uploads = 0;
    await pg.route("**/ui-config.json", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ pollIntervalMs: 300, pollMaxIntervalMs: 1200 }) }),
    );
    await pg.route("**/ui/v1/**", (route) => {
      const req = route.request();
      const p = new URL(req.url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      if (p.endsWith("/analyze")) {
        uploads++;
        return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ job_id: uploads === 1 ? "r1" : "r2" }) });
      }
      if (p.endsWith("/jobs/n1")) {
        n1++;
        if (n1 <= 3) return route.abort("connectionreset");
        return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(doneJob("n1")) });
      }
      if (p.endsWith("/jobs/r1")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"job_id":"r1","status":"failed","verdict":null,"violations":[],"error":"синтетический сбой"}' });
      if (p.endsWith("/jobs/r2")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(doneJob("r2")) });
      if (p.endsWith("/ui/v1/jobs")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"jobs":[]}' });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/n1`);
    await step("обрыв сети: баннер «Нет связи», затем результат без перезагрузки", async () => {
      await pg.getByText("Нет связи с сервером", { exact: false }).waitFor({ timeout: 6000 });
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 15000 });
      assert.ok(n1 >= 4, `запросов: ${n1}`);
    });
    await pg.goto(`${url}#/`);
    await step("сбой анализа → «Отправить этот файл повторно» → новая задача → результат", async () => {
      await pg.locator('input[type="file"]').setInputFiles({ name: "retry.mov", mimeType: "video/quicktime", buffer: Buffer.alloc(1024, 3) });
      await pg.getByRole("button", { name: "Начать проверку" }).click();
      await pg.getByText("Анализ не завершён").first().waitFor({ timeout: 8000 });
      await pg.getByRole("button", { name: "Отправить этот файл повторно" }).click();
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 10000 });
      assert.equal(uploads, 2);
      assert.ok(pg.url().includes("#/job/r2"), pg.url());
    });
    await c.close();
  }

  // ---------- сценарий 1ж: replay сохранённого отчёта — помечен, без запросов к серверу задач ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    const jobReqs = [];
    await pg.route("**/ui/v1/**", (route) => {
      const p = new URL(route.request().url()).pathname;
      if (p.includes("/jobs")) jobReqs.push(p);
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/replay`);
    await step("replay: JSON-отчёт открывается с пометкой, к серверу задач не обращается", async () => {
      await pg.getByText("Повторный просмотр сохранённого отчёта.", { exact: false }).waitFor({ timeout: 5000 });
      await pg.getByLabel("Отчёт JSON").setInputFiles({ name: "saved.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(doneJob("saved-1"))) });
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 5000 });
      await pg.getByText("ПОВТОРНЫЙ ПРОСМОТР: сохранённый отчёт", { exact: false }).waitFor({ timeout: 2000 });
      assert.equal(await pg.getByText("Copilot проверяющего").count(), 0, "Copilot в replay не нужен");
      assert.deepEqual(jobReqs, [], `запросы к задачам: ${jobReqs.join(",")}`);
    });
    await c.close();
  }

  // ---------- сценарий 1з: формат A02 2055fc4 — failed + обёртка partial (rejected_result, «<jpeg omitted>») ----------
  {
    const c = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const pg = await c.newPage();
    const wrapped = {
      job_id: "w1",
      status: "failed",
      verdict: null,
      violations: [],
      error: "pipeline_failed: incomplete analysis (meta.completeness/coverage): no_violation is not issued",
      partial: {
        job_id: "w1",
        complete: false,
        code: "pipeline_failed",
        partial: { status: "processing", meta: { complete: false } },
        rejected_result: {
          status: "failed",
          verdict: null,
          verdict_state: "no_violation",
          analysis_complete: true,
          violations: [],
          candidates: [{ type: "Синтетический кандидат", type_id: "V05", timecode_sec: 3, description: "", evidence_frames: [{ timestamp_sec: 3, image_base64: "<jpeg omitted>" }] }],
          timeline: [[0, 0.1], [3, 0.4], [6, 0.1]],
          meta: { complete: true, video: { duration: 6 }, stages: { ingest: { status: "ok", required: true, seconds: 0.4 } }, coverage: { ratio: 1, complete: true, by_stage: { ingest: 1 } } },
        },
      },
    };
    await pg.route("**/ui/v1/**", (route) => {
      const p = new URL(route.request().url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      if (p.endsWith("/jobs/w1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(wrapped) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/w1`);
    await step("A02 2055fc4 failed+partial: жёлтый подробный отчёт, плашка «кадр не передан», нет битых картинок и зелёного", async () => {
      await pg.getByText("Анализ не завершён").first().waitFor({ timeout: 6000 });
      await pg.getByText("Полнота анализа").first().waitFor({ timeout: 3000 });
      await pg.getByText("кадр не передан").first().waitFor({ timeout: 3000 });
      const broken = await pg.evaluate(() => [...document.images].filter((i) => i.complete && i.naturalWidth === 0).length);
      assert.equal(broken, 0, `битых картинок: ${broken}`);
      assert.equal(await pg.locator("section.bg-gradient-to-br.from-emerald-600").count(), 0, "зелёный баннер у failed");
    });
    await c.close();
  }

  // ---------- сценарий 1и: модальные окна и мобильный экран ----------
  for (const vp of [{ width: 1280, height: 900, name: "desktop" }, { width: 375, height: 812, name: "375px" }]) {
    const c = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
    const pg = await c.newPage();
    await pg.route("**/ui/v1/**", (route) => {
      const p = new URL(route.request().url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ authenticated: true, csrf_token: CSRF }) });
      if (p.endsWith("/jobs/m1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ...doneJob("m1"), meta: { ...doneJob("m1").meta, complete: true } }) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/m1`);
    await step(`${vp.name}: лайтбокс на весь экран и поверх всего`, async () => {
      await pg.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 6000 });
      await pg.locator("article button[aria-label^='Открыть кадр']").first().click();
      const dlg = pg.getByRole("dialog", { name: "Кадр-доказательство" });
      await dlg.waitFor({ timeout: 3000 });
      const box = await dlg.boundingBox();
      assert.ok(box && box.x === 0 && box.y === 0 && Math.abs(box.width - vp.width) < 2 && Math.abs(box.height - vp.height) < 2, `рамка лайтбокса ${JSON.stringify(box)}`);
      const top = await pg.evaluate(() => {
        const el = document.elementFromPoint(window.innerWidth / 2, 20);
        return !!el?.closest('[role="dialog"]');
      });
      assert.ok(top, "лайтбокс перекрыт другим элементом (шапкой)");
      await pg.keyboard.press("Escape");
    });
    await step(`${vp.name}: окно «Настройки» поверх страницы; нет горизонтальной прокрутки`, async () => {
      await pg.getByRole("button", { name: "Настройки" }).click();
      const dlg = pg.getByRole("dialog");
      await dlg.waitFor({ timeout: 3000 });
      const top = await pg.evaluate(() => {
        const d = document.querySelector('[role="dialog"]');
        const r = d.getBoundingClientRect();
        const el = document.elementFromPoint(r.left + r.width / 2, r.top + 30);
        return !!el?.closest('[role="dialog"]');
      });
      assert.ok(top, "окно настроек перекрыто");
      await pg.keyboard.press("Escape");
      const sw = await pg.evaluate(() => document.documentElement.scrollWidth);
      assert.ok(sw <= vp.width, `scrollWidth ${sw} > ${vp.width}`);
    });
    await c.close();
  }

  // ---------- сценарий 2: failed и неполный анализ не зелёные ----------
  for (const [name, body, expectText] of [
    ["failed", { job_id: "f1", status: "failed", verdict: null, violations: [], error: "синтетическая ошибка" }, "Анализ не завершён"],
    ["failed-partial", { job_id: "f1", status: "failed", verdict: null, violations: [], error: "timeout", partial: { ...doneJob("f1"), verdict: "no_violation", violations: [] , timeline: [[0, 0.1], [5, 0.2]] } }, "Анализ не завершён"],
    ["partial", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: { partial: true } }, "Анализ неполный"],
    ["clean-unknown", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: {} }, "Нарушение не обнаружено"],
    ["clean-complete", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: { complete: true, coverage: 1 } }, "Нарушение не обнаружено"],
    ["clean-coverage-87", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: { complete: true, coverage: 0.87 } }, "Анализ неполный"],
    ["mock", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: { mock: true, complete: true } }, "Тестовый результат"],
  ]) {
    const c = await browser.newContext({ viewport: { width: 1280, height: 860 } });
    const pg = await c.newPage();
    await pg.route("**/ui/v1/**", (route) => {
      const p = new URL(route.request().url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"authenticated":true}' });
      if (p.endsWith("/jobs/f1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/f1`);
    await step(`${name}: «${expectText}»`, async () => {
      await pg.getByText(expectText, { exact: false }).first().waitFor({ timeout: 6000 });
      const green = await pg.locator("section.bg-gradient-to-br.from-emerald-600").count();
      assert.equal(green, name === "clean-complete" ? 1 : 0, `зелёный баннер: ${green} (ожидался только при подтверждённой полноте)`);
      if (!name.startsWith("clean-u") && name !== "clean-complete")
        assert.equal(await pg.getByRole("heading", { name: "Нарушение не обнаружено" }).count(), 0, "формулировка «не обнаружено» при неполном/mock");
      await pg.waitForTimeout(600);
      if (name === "failed") await pg.screenshot({ path: path.join(shots, "12-failed.png") });
      if (name === "partial") await pg.screenshot({ path: path.join(shots, "13-partial.png") });
    });
    await c.close();
  }
} finally {
  await browser.close();
  await server.close();
}

if (failures.length) {
  console.error(`\n${failures.length} проверок не прошло`);
  process.exit(1);
}
console.log("\nВсе мок-e2e проверки прошли");
