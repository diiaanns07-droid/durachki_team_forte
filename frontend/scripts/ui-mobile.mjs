// Синтетическое демо (?demo=1): ширина 375/768 — нет горизонтальной прокрутки, нет ошибок страницы.
import { launch, loadPlaywright, startPreview } from "./pw.mjs";
const pw = await loadPlaywright();
const { server, url } = await startPreview(4191);
const browser = await launch(pw);
let bad = 0;
try {
  for (const width of [375, 768]) for (const theme of ["light", "dark"]) for (const [name, hash, act] of [
    ["upload", "#/", null], ["result", "#/job/demo", null], ["result-clean", "#/job/demo-clean", null], ["about", "#/about", null],
    ["processing", "#/", async (p) => { await p.getByRole("button", { name: "Симуляция обработки" }).click(); await p.waitForTimeout(2500); }],
  ]) {
    const ctx = await browser.newContext({ viewport: { width, height: 812 }, colorScheme: theme });
    await ctx.addInitScript((t) => { try { localStorage.setItem("ks.theme", t); } catch {} }, theme);
    const p = await ctx.newPage(); const errs = [];
    p.on("pageerror", (e) => errs.push(String(e)));
    await p.goto(`${url}?demo=1${hash}`); await p.waitForTimeout(700);
    if (act) await act(p);
    const sw = await p.evaluate(() => document.documentElement.scrollWidth);
    const ok = sw <= width && errs.length === 0;
    if (!ok) bad++;
    console.log(`${ok ? "✓" : "✗"} ${width}px ${theme} ${name}: scrollWidth=${sw}${errs.length ? " pageerror " + errs[0] : ""}`);
    await ctx.close();
  }
} finally { await browser.close(); await server.close(); }
console.log(bad ? `${bad} не прошло` : "мобильные проверки прошли");
process.exit(bad ? 1 : 0);
