// Скриншоты экранов для питча — ТОЛЬКО синтетическое демо (?demo=1), без реальных кадров.
// Запуск: npm run build && npm run screenshots  → frontend/screenshots/*.png
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

const out = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../screenshots");
mkdirSync(out, { recursive: true });

const pw = await loadPlaywright();
const { server, url } = await startPreview(4180);
const browser = await launch(pw);

async function shot(name, { theme = "light", width = 1440, height = 900, full = true, hash = "#/", before } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: width < 500 ? 2 : 1, colorScheme: theme });
  await ctx.addInitScript((t) => {
    try {
      localStorage.setItem("ks.theme", t);
    } catch {}
  }, theme);
  const page = await ctx.newPage();
  await page.goto(`${url}?demo=1${hash}`);
  await page.waitForTimeout(700);
  if (before) await before(page);
  await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: full });
  await ctx.close();
  console.log("✓", name);
}

try {
  await shot("01-upload-demo");
  await shot("02-processing", {
    full: false,
    before: async (p) => {
      await p.getByRole("button", { name: "Симуляция обработки" }).click();
      await p.waitForTimeout(3200);
    },
  });
  await shot("03-result-light", { hash: "#/job/demo", before: (p) => p.waitForTimeout(500) });
  await shot("04-result-dark", { hash: "#/job/demo", theme: "dark" });
  await shot("05-evidence-lightbox", {
    hash: "#/job/demo",
    full: false,
    before: async (p) => {
      await p.locator("article button[aria-label^='Открыть кадр']").first().click();
      await p.waitForTimeout(300);
    },
  });
  await shot("06-seek-episode-2", {
    hash: "#/job/demo",
    full: false,
    before: async (p) => {
      await p.locator("#episode-2 header button.btn-brand").click();
      await p.waitForTimeout(900);
    },
  });
  await shot("07-result-clean", { hash: "#/job/demo-clean" });
  await shot("08-about", { hash: "#/about" });
  await shot("09-mobile-result", { hash: "#/job/demo", width: 375, height: 812 });
  await shot("10-mobile-upload-dark", { width: 375, height: 812, theme: "dark" });
} finally {
  await browser.close();
  await server.close();
}
