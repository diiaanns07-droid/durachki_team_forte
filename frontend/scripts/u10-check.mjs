// WIP (U10): в 2 из ~10 прогонов падал шаг «ключ сохраняется…» — тост «Настройки сохранены» не дождался за 2 с
// (один раз за ним упал и следующий шаг: окно осталось открытым). Причина не найдена; в шаг добавлена диагностика
// (diag-key-*.png + состояние кнопки). Компоненты это не блокирует: 48/48 PASS в последних 4 прогонах подряд.
// U10: браузерная проверка окна «Настройки» и экрана «О системе» — ТОЛЬКО синтетическое демо (?demo=1).
// Ширины 360 / 768 / 1440, светлая и тёмная тема, клавиатура (Tab/Shift+Tab/Escape), возврат фокуса,
// видимый фокус, действия видны на маленьком экране, нет горизонтальной прокрутки и обрезанных кнопок,
// prefers-reduced-motion. Запуск: npm run build && node scripts/u10-check.mjs  (скриншоты → U10_SHOTS).
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

const out = process.env.U10_SHOTS ?? path.resolve("screenshots-u10");
mkdirSync(out, { recursive: true });
const pw = await loadPlaywright();
const { server, url } = await startPreview(4190);
const browser = await launch(pw);
const results = [];

async function step(name, fn) {
  try {
    await fn();
    results.push(["PASS", name]);
    console.log("PASS", name);
  } catch (e) {
    const msg = String(e?.message ?? e).split("\n").filter((l) => l.trim()).slice(0, 4).join(" | ");
    results.push(["FAIL", name, msg]);
    console.log("FAIL", name, "—", msg);
  }
}

const inDialog = (p) => p.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'));
const noHScroll = async (p, w) => {
  const sw = await p.evaluate(() => document.documentElement.scrollWidth);
  assert.ok(sw <= w, `scrollWidth ${sw} > ${w}`);
};
const clipped = (p, sel) =>
  p.evaluate((s) => {
    const bad = [];
    for (const el of document.querySelectorAll(s)) {
      if (el.scrollWidth > el.clientWidth + 1) bad.push((el.textContent || el.getAttribute("aria-label") || el.tagName).trim().slice(0, 40));
    }
    return bad;
  }, sel);

async function openSettingsByKeyboard(p) {
  const btn = p.getByRole("button", { name: "Настройки", exact: true });
  await btn.focus();
  await p.keyboard.press("Enter");
  const dlg = p.getByRole("dialog", { name: "Настройки" });
  await dlg.waitFor({ timeout: 3000 });
  return dlg;
}

const VIEWPORTS = [
  { name: "360", width: 360, height: 640 },
  { name: "768", width: 768, height: 1024 },
  { name: "1440", width: 1440, height: 900 },
];

try {
  for (const vp of VIEWPORTS) {
    for (const theme of ["light", "dark"]) {
      const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, deviceScaleFactor: vp.width < 500 ? 2 : 1, colorScheme: theme });
      await ctx.addInitScript((t) => {
        try {
          localStorage.setItem("ks.theme", t);
        } catch {}
      }, theme);
      const p = await ctx.newPage();
      await p.goto(`${url}?demo=1#/`);
      await p.waitForTimeout(500);
      const tag = `${vp.name}-${theme}`;

      await step(`${tag}: «Настройки» открываются с клавиатуры, фокус в первом поле`, async () => {
        await openSettingsByKeyboard(p);
        await p.waitForTimeout(450); // анимация появления 0.28 с
        const id = await p.evaluate(() => document.activeElement?.getAttribute("id"));
        const label = await p.evaluate((i) => document.querySelector(`label[for="${i}"]`)?.textContent, id);
        assert.equal(label, "Адрес сервера (API)");
        const top = await p.evaluate(() => document.querySelector('[role="dialog"] form').scrollTop);
        assert.equal(top, 0, "окно открылось не с начала — статус подключения скрыт");
        await p.screenshot({ path: path.join(out, `settings-open-${tag}.png`) });
      });
      await step(`${tag}: Tab и Shift+Tab не выходят из окна; фокус видим`, async () => {
        for (let i = 0; i < 14; i++) {
          await p.keyboard.press("Tab");
          assert.ok(await inDialog(p), `Tab #${i + 1} вывел фокус из окна`);
          const visible = await p.evaluate(() => {
            const cs = getComputedStyle(document.activeElement);
            return cs.outlineStyle !== "none" || (cs.boxShadow && cs.boxShadow !== "none");
          });
          assert.ok(visible, `фокус не виден на Tab #${i + 1}`);
        }
        for (let i = 0; i < 14; i++) {
          await p.keyboard.press("Shift+Tab");
          assert.ok(await inDialog(p), `Shift+Tab #${i + 1} вывел фокус из окна`);
        }
      });
      await step(`${tag}: кнопки «Отмена» и «Сохранить» целиком в экране, тело прокручивается`, async () => {
        for (const name of ["Отмена", "Сохранить"]) {
          const b = await p.getByRole("dialog").getByRole("button", { name, exact: true }).boundingBox();
          assert.ok(b && b.y >= 0 && b.y + b.height <= vp.height + 0.5 && b.x >= 0 && b.x + b.width <= vp.width + 0.5, `${name}: ${JSON.stringify(b)}`);
          assert.ok(b.height >= 40, `${name}: высота ${b.height} < 40`);
        }
        const sc = await p.evaluate(() => {
          const f = document.querySelector('[role="dialog"] form');
          return { sh: f.scrollHeight, ch: f.clientHeight, oy: getComputedStyle(f).overflowY };
        });
        assert.equal(sc.oy, "auto");
        if (vp.height <= 640) assert.ok(sc.sh > sc.ch, "на 360×640 ожидалась прокрутка содержимого");
      });
      await step(`${tag}: нет горизонтальной прокрутки и обрезанных подписей`, async () => {
        await noHScroll(p, vp.width);
        const bad = await clipped(p, '[role="dialog"] button, [role="dialog"] label, [role="dialog"] legend, [role="dialog"] h2');
        assert.deepEqual(bad, []);
      });
      await p.screenshot({ path: path.join(out, `settings-after-tab-${tag}.png`) });
      await step(`${tag}: Escape закрывает окно и возвращает фокус на кнопку «Настройки»`, async () => {
        await p.keyboard.press("Escape");
        await p.getByRole("dialog").waitFor({ state: "detached", timeout: 2000 });
        const name = await p.evaluate(() => document.activeElement?.getAttribute("aria-label"));
        assert.equal(name, "Настройки");
      });

      if (theme === "light") {
        await step(`${tag}: неверный адрес — ошибка у поля, окно не закрывается`, async () => {
          const dlg = await openSettingsByKeyboard(p);
          await p.getByLabel("Адрес сервера (API)").fill("localhost:8000");
          await dlg.getByRole("button", { name: "Сохранить" }).click();
          await dlg.getByText("начиная с http:// или https://").waitFor({ timeout: 1500 });
          await p.waitForTimeout(300);
          assert.equal(await p.getByLabel("Адрес сервера (API)").getAttribute("aria-invalid"), "true");
          assert.ok(await inDialog(p));
          await p.screenshot({ path: path.join(out, `settings-error-${tag}.png`) });
          await p.keyboard.press("Escape");
        });
        await step(`${tag}: ключ сохраняется в памяти вкладки и больше не показывается`, async () => {
          const dlg = await openSettingsByKeyboard(p);
          await dlg.getByLabel("API-ключ", { exact: true }).fill("synthetic-key-777");
          await dlg.getByRole("button", { name: "Сохранить" }).click();
          try {
            await p.getByText("Настройки сохранены").waitFor({ timeout: 2000 });
          } catch (e) {
            const diag = await p.evaluate(() => ({
              dialog: !!document.querySelector('[role="dialog"]'),
              saveDisabled: document.querySelector('[role="dialog"] button[type="submit"]')?.disabled,
              active: document.activeElement?.outerHTML.slice(0, 120),
              toasts: [...document.querySelectorAll('[role="status"]')].map((x) => x.textContent),
            }));
            await p.screenshot({ path: path.join(out, `diag-key-${tag}.png`) });
            throw new Error(`${e.message.split("\n")[0]} | ${JSON.stringify(diag)}`);
          }
          await p.getByRole("dialog").waitFor({ state: "detached", timeout: 2000 });
          await p.getByText("Настройки сохранены").waitFor({ state: "detached", timeout: 6000 });
          const dlg2 = await openSettingsByKeyboard(p);
          await dlg2.getByText("Значение не показывается").waitFor({ timeout: 1500 });
          await p.waitForTimeout(450);
          const html = await p.content();
          assert.ok(!html.includes("synthetic-key-777"), "ключ попал в DOM");
          const stored = await p.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
          assert.ok(!stored.includes("synthetic-key-777"), "ключ попал в web storage");
          await p.screenshot({ path: path.join(out, `settings-key-set-${tag}.png`) });
          await p.keyboard.press("Escape");
        });
        await step(`${tag}: очистка истории — с подтверждением и обратной связью`, async () => {
          const dlg = await openSettingsByKeyboard(p);
          await dlg.getByRole("button", { name: "Очистить историю проверок" }).click();
          await dlg.getByRole("group", { name: "Подтверждение очистки истории" }).waitFor({ timeout: 1500 });
          await dlg.getByRole("button", { name: "Очистить", exact: true }).click();
          await dlg.getByText("История очищена").waitFor({ timeout: 1500 });
          await p.keyboard.press("Escape");
          await p.getByText("История проверок очищена").waitFor({ state: "detached", timeout: 6000 });
        });
      }

      await step(`${tag}: «О системе» без горизонтальной прокрутки и обрезанных заголовков`, async () => {
        await p.goto(`${url}?demo=1#/about`);
        await p.waitForTimeout(400);
        await p.getByRole("heading", { level: 1, name: "О системе Kassa Sentinel" }).waitFor({ timeout: 2000 });
        await noHScroll(p, vp.width);
        const bad = await clipped(p, "main h1, main h2, main h3, main button, main a");
        assert.deepEqual(bad, []);
        await p.screenshot({ path: path.join(out, `about-${tag}.png`), fullPage: true });
      });
      await ctx.close();
    }
  }

  await step("reduced motion: окно без анимации появления", async () => {
    const ctx = await browser.newContext({ viewport: { width: 768, height: 900 }, reducedMotion: "reduce" });
    const p = await ctx.newPage();
    await p.goto(`${url}?demo=1#/`);
    await p.waitForTimeout(300);
    await openSettingsByKeyboard(p);
    const anim = await p.evaluate(() => getComputedStyle(document.querySelector('[role="dialog"]')).animationName);
    assert.equal(anim, "none");
    await ctx.close();
  });
  await step("клик по фону закрывает окно без сохранения", async () => {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const p = await ctx.newPage();
    await p.goto(`${url}?demo=1#/`);
    await p.waitForTimeout(300);
    await openSettingsByKeyboard(p);
    await p.getByLabel("Адрес сервера (API)").fill("http://changed.example");
    await p.mouse.click(10, 450);
    await p.getByRole("dialog").waitFor({ state: "detached", timeout: 2000 });
    const base = await p.evaluate(() => localStorage.getItem("ks.apiBase"));
    assert.equal(base, null);
    await ctx.close();
  });
  await step("без ?demo: сервер недоступен — заметное состояние в окне (синтетика, бэкенда нет)", async () => {
    const ctx = await browser.newContext({ viewport: { width: 768, height: 1024 } });
    const p = await ctx.newPage();
    await p.goto(`${url}#/`);
    await p.waitForTimeout(1500);
    const dlg = await openSettingsByKeyboard(p);
    await p.waitForTimeout(450);
    const txt = await dlg.innerText();
    assert.ok(/Сервер анализа не отвечает|Проверяем связь|Вход не выполнен/.test(txt), txt.slice(0, 200));
    await p.screenshot({ path: path.join(out, "settings-no-server-768-light.png") });
    await ctx.close();
  });
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}

const failed = results.filter((r) => r[0] === "FAIL");
console.log(`\nИТОГ: ${results.length - failed.length} PASS, ${failed.length} FAIL → скриншоты: ${out}`);
process.exit(failed.length ? 1 : 0);
