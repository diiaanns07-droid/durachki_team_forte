// A15: проверка экранов роли (экран задачи, «до / после», лайтбокс, навигатор эпизодов, повторный просмотр)
// на ширинах 360 / 768 / 1440 px в светлой и тёмной теме. ТОЛЬКО синтетика: демо (?demo=1) и бэкенд A02,
// подменённый page.route синтетическими ответами; реальных видео, кадров и описаний сцен нет.
// Проверяет: нет горизонтальной прокрутки; ничто не уходит за край; текст не обрезан; у кнопок есть имя и
// размер ≥ 24 px; фокус с клавиатуры виден; при prefers-reduced-motion бесконечные анимации стоят.
// Ошибкой считается только найденное внутри компонентов A15 (их CSS-модуль) и прокрутка страницы;
// замечания по чужим элементам печатаются отдельно — для U01.
// Запуск: npm run build && node scripts/a15-ui-check.mjs
//   скриншоты-образцы → screenshots/a15-*.png;  A15_SHOTS_ALL=<папка> — сохранить все сочетания.
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const shotsDir = path.resolve(here, "../screenshots");
const allDir = process.env.A15_SHOTS_ALL ? path.resolve(process.env.A15_SHOTS_ALL) : null;
mkdirSync(shotsDir, { recursive: true });
if (allDir) mkdirSync(allDir, { recursive: true });

const SIZES = [
  { w: 360, h: 780 },
  { w: 768, h: 1024 },
  { w: 1440, h: 900 },
];
const THEMES = ["light", "dark"];
const CSRF = "csrf_synthetic_a15";

const svg = (label, fill = "#cfd8de") =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="640" height="360" fill="${fill}"/>` +
      `<text x="24" y="190" font-size="30" font-family="sans-serif" fill="#334155">СИНТЕТИКА · ${label}</text></svg>`,
  ).toString("base64");

/** Синтетический сохранённый отчёт: эпизод-переход с кадрами «до» и «после» и подсветкой A12. */
function savedReport() {
  return {
    job_id: "a15-saved",
    status: "done",
    verdict: "violation",
    analysis_complete: true,
    violations: [
      {
        type: "Синтетический эпизод-переход",
        type_id: "V02",
        timecode_sec: 21.5,
        t_start: 20,
        t_end: 26,
        confidence: 0.7,
        description: "Синтетическое описание для проверки интерфейса.",
        evidence_frames: [
          { timestamp_sec: 17.5, image_base64: svg("до", "#dbe4ea") },
          { timestamp_sec: 21.5, image_base64: svg("после", "#cfd8de") },
          { timestamp_sec: 24, image_base64: svg("позже", "#c4cfd6") },
        ],
        rules: [{ document: "Синтетический документ", clause: null, url: "https://adilet.zan.kz/rus", quote: "синтетическая цитата", verified: false }],
      },
    ],
    timeline: [
      [0, 0.1],
      [21.5, 0.7],
      [40, 0.1],
    ],
    meta: {
      duration_sec: 40,
      complete: true,
      ui: {
        evidence_overlays: [
          {
            violation_index: 0,
            type_id: "V02",
            timecode_sec: 21.5,
            shapes: [{ kind: "box", key: "zone", label: "синтетическая рамка", color: "#e5484d", bbox: [0.2, 0.25, 0.6, 0.8] }],
          },
        ],
      },
    },
  };
}

/** Подмена A02: сессия есть (или 401), задача отвечает по сценарию. */
async function mockApi(page, { session = 200, job }) {
  const held = [];
  await page.route("**/ui/v1/**", async (route) => {
    const req = route.request();
    const p = new URL(req.url()).pathname;
    const json = (status, body) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (p.endsWith("/ui/v1/session")) {
      return session === 200 ? json(200, { authenticated: true, csrf_token: CSRF }) : json(401, { authenticated: false, password_login: false });
    }
    if (p.endsWith("/ui/v1/jobs")) return json(200, { jobs: [] });
    const m = p.match(/\/ui\/v1\/jobs\/([^/]+)$/);
    if (m) {
      if (job === "hold") return new Promise((resolve) => held.push(() => route.fulfill({ status: 404, body: "{}" }).catch(() => {}).finally(resolve)));
      if (job === "abort") return route.abort("failed");
      if (typeof job === "number") return json(job, { detail: "synthetic" });
      if (job === "failed") return json(200, { job_id: m[1], status: "failed", error: "pipeline_failed: синтетическая ошибка", violations: [], timeline: [], meta: {} });
    }
    if (p.endsWith("/video")) return route.fulfill({ status: 404, body: "" });
    return json(404, {});
  });
  return () => held.splice(0).forEach((f) => f());
}

const SCENES = [
  { id: "loading", hash: "#/job/a15-slow", api: { job: "hold" }, marker: "Получаем статус задачи…" },
  { id: "network", hash: "#/job/a15-net", api: { job: "abort" }, marker: "Нет связи с сервером" },
  { id: "error-404", hash: "#/job/a15-gone", api: { job: 404 }, marker: "Не удалось получить результат" },
  { id: "error-403", hash: "#/job/a15-deny", api: { job: 403 }, marker: "Настройки доступа" },
  { id: "failed", hash: "#/job/a15-fail", api: { job: "failed" }, marker: "Анализ не завершён" },
  { id: "login", hash: "#/job/a15-auth", api: { session: 401, job: 401 }, marker: "Вход для проверяющего" },
  {
    id: "result",
    demo: true,
    hash: "#/job/demo",
    marker: "Эпизод 1 из 2",
    act: async (p) => {
      await p.getByRole("group", { name: "Просмотр эпизодов" }).evaluate((el) => el.scrollIntoView({ block: "center" }));
      await p.waitForTimeout(300);
    },
  },
  {
    id: "before-after",
    demo: true,
    hash: "#/job/demo",
    marker: "До и после начала эпизода",
    act: async (p) => {
      await p.getByText("До и после начала эпизода").first().scrollIntoViewIfNeeded();
      await p.waitForTimeout(400);
    },
  },
  {
    id: "lightbox",
    demo: true,
    hash: "#/job/demo",
    marker: "Эпизод 1 из 2",
    act: async (p) => {
      await p.locator("article button[aria-label^='Открыть кадр']").first().click();
      await p.getByRole("dialog", { name: "Кадр-доказательство" }).waitFor({ timeout: 3000 });
      await p.waitForTimeout(300);
    },
  },
  { id: "replay", hash: "#/replay", api: {}, marker: "Открыть отчёт для повторного просмотра" },
  {
    id: "replay-open",
    hash: "#/replay",
    api: {},
    marker: "Открыть отчёт для повторного просмотра",
    act: async (p) => {
      await p.getByLabel("Отчёт JSON").setInputFiles({ name: "synthetic-saved.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(savedReport())) });
      await p.getByText("ПОВТОРНЫЙ ПРОСМОТР: сохранённый отчёт", { exact: false }).waitFor({ timeout: 5000 });
      await p.getByText("До и после начала эпизода").first().scrollIntoViewIfNeeded();
      await p.waitForTimeout(400);
    },
  },
];

/** Образцы для репозитория (остальные сочетания — только при A15_SHOTS_ALL). */
const SAMPLES = new Set([
  "loading-1440-light",
  "network-768-light",
  "error-404-360-light",
  "failed-1440-dark",
  "login-360-dark",
  "result-1440-light",
  "before-after-768-light",
  "lightbox-360-dark",
  "lightbox-1440-light",
  "replay-768-light",
  "replay-open-1440-dark",
]);

/** Образцы с кольцом фокуса после обхода Tab'ом. */
const FOCUS_SAMPLES = new Set(["lightbox-1440-dark", "error-404-360-light"]);

// ---------- проверки в странице ----------
function audit() {
  const vw = window.innerWidth;
  const mineHash = (() => {
    for (const el of document.querySelectorAll('[class*="_scope_"], [class*="scope"]')) {
      for (const c of el.classList) {
        const m = c.match(/^_scope_([A-Za-z0-9-]+)_\d+$/);
        if (m) return m[1];
      }
    }
    return null;
  })();
  const isMine = (el) => !!mineHash && !!el.closest(`[class*="_${mineHash}_"]`);
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && cs.visibility !== "hidden" && cs.display !== "none" && cs.opacity !== "0";
  };
  const describe = (el) => {
    const cls = [...el.classList].slice(0, 2).join(".");
    const txt = (el.getAttribute("aria-label") || el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 50);
    return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""}${txt ? ` «${txt}»` : ""}`;
  };
  const clippedByAncestor = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const cs = getComputedStyle(p);
      if (/(auto|scroll|hidden|clip)/.test(cs.overflowX)) return true;
    }
    return false;
  };
  const out = { mine: [], others: [], mineFound: !!mineHash };
  const push = (el, msg) => (isMine(el) ? out.mine : out.others).push(`${msg}: ${describe(el)}`);

  const doc = document.documentElement;
  if (doc.scrollWidth > vw + 1) out.mine.push(`горизонтальная прокрутка страницы: ${doc.scrollWidth} > ${vw}`);

  for (const el of document.body.querySelectorAll("*")) {
    if (!visible(el) || el.closest("svg")) continue;
    const r = el.getBoundingClientRect();
    if ((r.right > vw + 1 || r.left < -1) && !clippedByAncestor(el) && getComputedStyle(el).position !== "fixed") push(el, "за краем экрана");
    const hasText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
    if (hasText) {
      const cs = getComputedStyle(el);
      const clipsX = /(hidden|clip)/.test(cs.overflowX) || cs.textOverflow === "ellipsis";
      const clipsY = /(hidden|clip)/.test(cs.overflowY) || (cs.webkitLineClamp && cs.webkitLineClamp !== "none");
      if ((clipsX && el.scrollWidth > el.clientWidth + 1) || (clipsY && el.scrollHeight > el.clientHeight + 2)) push(el, "текст обрезан");
    }
  }
  for (const el of document.querySelectorAll("button, a[href], [role=button], input:not([type=hidden]), label")) {
    if (!isMine(el)) continue;
    const isInput = el.tagName === "INPUT";
    const target = isInput ? el.closest("label") ?? el : el;
    if (!visible(target)) continue;
    if (el.tagName !== "LABEL") {
      const name = (el.getAttribute("aria-label") || el.textContent || el.getAttribute("title") || (isInput && el.labels?.length ? "label" : "")).trim();
      if (!name) push(el, "нет доступного имени");
    }
    if (el.tagName === "BUTTON" || el.tagName === "A") {
      const r = el.getBoundingClientRect();
      if (r.width < 24 || r.height < 24) push(el, `мишень меньше 24 px (${Math.round(r.width)}×${Math.round(r.height)})`);
    }
  }
  return out;
}

/**
 * Фокус: помечаем текущий элемент и снимаем «подпись» стилей (контур и тень у него и двух предков —
 * для скрытого поля файла фокус показывает метка через :focus-within). Видимость фокуса = подпись
 * в фокусе отличается от подписи того же элемента без фокуса.
 */
function focusProbe() {
  const el = document.activeElement;
  if (!el || el === document.body) return null;
  document.querySelectorAll("[data-a15-probe]").forEach((x) => x.removeAttribute("data-a15-probe"));
  el.setAttribute("data-a15-probe", "1");
  let mineHash = null;
  for (const x of document.querySelectorAll('[class*="_scope_"]'))
    for (const c of x.classList) {
      const m = c.match(/^_scope_([A-Za-z0-9-]+)_\d+$/);
      if (m) mineHash = m[1];
    }
  const sig = [el, el.parentElement, el.parentElement?.parentElement]
    .filter(Boolean)
    .map((x) => {
      const cs = getComputedStyle(x);
      return `${cs.outlineStyle} ${cs.outlineWidth} ${cs.outlineColor} | ${cs.boxShadow}`;
    })
    .join(" || ");
  const txt = (el.getAttribute("aria-label") || el.textContent || el.tagName).trim().replace(/\s+/g, " ").slice(0, 40);
  return { mine: !!mineHash && !!el.closest(`[class*="_${mineHash}_"]`), sig, label: `${el.tagName.toLowerCase()} «${txt}»` };
}
function probeSignature() {
  const el = document.querySelector("[data-a15-probe]");
  if (!el) return null;
  el.removeAttribute("data-a15-probe");
  return [el, el.parentElement, el.parentElement?.parentElement]
    .filter(Boolean)
    .map((x) => {
      const cs = getComputedStyle(x);
      return `${cs.outlineStyle} ${cs.outlineWidth} ${cs.outlineColor} | ${cs.boxShadow}`;
    })
    .join(" || ");
}

const pw = await loadPlaywright();
const { server, url } = await startPreview(4186);
const browser = await launch(pw);
let failed = 0;
const othersAll = new Set();
const fail = (msg) => {
  failed++;
  console.error("✗", msg);
};

try {
  for (const sc of SCENES) {
    for (const { w, h } of SIZES) {
      for (const theme of THEMES) {
        const key = `${sc.id}-${w}-${theme}`;
        const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: theme, deviceScaleFactor: w < 500 ? 2 : 1 });
        await ctx.addInitScript((t) => {
          try {
            localStorage.setItem("ks.theme", t);
          } catch {}
        }, theme);
        const page = await ctx.newPage();
        const release = sc.demo ? () => {} : await mockApi(page, sc.api ?? {});
        try {
          await page.goto(`${url}${sc.demo ? "?demo=1" : ""}${sc.hash}`);
          await page.getByText(sc.marker, { exact: false }).first().waitFor({ timeout: 8000 });
          await page.waitForTimeout(450);
          if (sc.act) await sc.act(page);
          // Снимок — до обхода Tab'ом (обход прокручивает страницу и оставляет кольцо фокуса).
          const name = `a15-${key}.png`;
          if (SAMPLES.has(key)) await page.screenshot({ path: path.join(shotsDir, name) });
          if (allDir) await page.screenshot({ path: path.join(allDir, name) });
          const res = await page.evaluate(audit);
          if (!res.mineFound) fail(`${key}: компоненты A15 не найдены на странице`);
          for (const m of res.mine) fail(`${key}: ${m}`);
          for (const m of res.others) othersAll.add(`${sc.id} @${w}: ${m}`);
          // Фокус: проходим Tab'ом; для каждого элемента A15 сравниваем вид в фокусе и без него (Shift+Tab и обратно).
          const seen = new Set();
          for (let i = 0; i < 30; i++) {
            await page.keyboard.press("Tab");
            const f = await page.evaluate(focusProbe);
            if (!f || !f.mine || seen.has(f.label)) continue;
            seen.add(f.label);
            await page.keyboard.press("Shift+Tab");
            const plain = await page.evaluate(probeSignature);
            await page.keyboard.press("Tab");
            if (plain !== null && plain === f.sig) fail(`${key}: фокус не виден на ${f.label}`);
          }
          if (FOCUS_SAMPLES.has(key)) await page.screenshot({ path: path.join(shotsDir, `a15-${key}-focus.png`) });
          console.log("✓", key, seen.size ? `(фокус: ${seen.size} эл.)` : "");
        } catch (e) {
          fail(`${key}: ${e.message.split("\n")[0]}`);
        } finally {
          release();
          await ctx.close();
        }
      }
    }
  }

  // reduced motion: бесконечные анимации на экранах A15 остановлены.
  for (const sc of SCENES.filter((x) => ["loading", "network", "replay"].includes(x.id))) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    const page = await ctx.newPage();
    const release = await mockApi(page, sc.api ?? {});
    try {
      await page.goto(`${url}${sc.hash}`);
      await page.getByText(sc.marker, { exact: false }).first().waitFor({ timeout: 8000 });
      await page.waitForTimeout(300);
      const running = await page.evaluate(() =>
        document
          .getAnimations()
          .filter((a) => a.playState === "running" && a.effect?.getComputedTiming?.().iterations === Infinity)
          .map((a) => `${a.animationName ?? "?"} @ ${a.effect?.target?.className ?? ""}`.slice(0, 80)),
      );
      if (running.length) fail(`reduced motion (${sc.id}): идут анимации: ${running.join("; ")}`);
      else console.log("✓ reduced motion:", sc.id, "— бесконечных анимаций нет");
      const smooth = await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior);
      if (smooth === "smooth") othersAll.add(`reduced motion: html { scroll-behavior: smooth } не отключается (index.css)`);
    } catch (e) {
      fail(`reduced motion (${sc.id}): ${e.message.split("\n")[0]}`);
    } finally {
      release();
      await ctx.close();
    }
  }
} finally {
  await browser.close();
  await server.close();
}

if (othersAll.size) {
  console.log(`\nЗамечания вне компонентов A15 (для U01, не ошибки этой проверки): ${othersAll.size}`);
  for (const m of [...othersAll].slice(0, 60)) console.log("  ·", m);
}
if (failed) {
  console.error(`\n${failed} проверок A15 не прошло`);
  process.exit(1);
}
console.log("\nВсе проверки A15 прошли (360 / 768 / 1440 × светлая / тёмная, фокус, reduced motion)");
