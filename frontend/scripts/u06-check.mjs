// U06: проверки плеера, кадров-доказательств и лайтбокса в Chromium — ТОЛЬКО синтетика
// (?demo=1 и синтетический отчёт + синтетическое видео в #/replay). Выход: PASS/FAIL по пунктам и скриншоты.
//   npm run build && U06_OUT=/tmp/u06 node scripts/u06-check.mjs
// Фикстуры (portrait.webm 464×848, landscape.webm 640×360, broken.webm, report.json, report-beyond.json) скрипт
// генерирует сам через ffmpeg во временную папку; U06_FIX=<папка> — взять готовые. Реальные видео не нужны и не читаются.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

/** Синтетическое портретное/альбомное видео с подписью «СИНТЕТИКА · U06», кадры из него и отчёт с краевыми случаями. */
function makeFixtures(dir) {
  const font = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"].find(existsSync);
  const label = (text, y, size) => (font ? `,drawtext=fontfile=${font}:text='${text}':fontcolor=white:fontsize=${size}:x=(w-tw)/2:y=${y}` : "");
  const clock = (y, size) => (font ? `,drawtext=fontfile=${font}:text='%{pts\\:hms}':fontcolor=0xfde68a:fontsize=${size}:x=(w-tw)/2:y=${y}` : "");
  const video = (name, w, h, d, box, step) =>
    execFileSync("ffmpeg", [
      ...["-hide_banner", "-loglevel", "error", "-y"],
      ...["-f", "lavfi", "-i", `color=c=0x1d2740:s=${w}x${h}:r=25:d=${d}`, "-f", "lavfi", "-i", `color=c=0xb8436e:s=${box}:r=25:d=${d}`],
      "-filter_complex",
      `[0]drawgrid=w=${step}:h=${step}:t=1:c=0x2f3c5b@0.9[bg];[bg][1]overlay=x='mod(t*30\\,${w - 140})':y=${Math.round(h * 0.6)}` +
        label("СИНТЕТИКА · U06", 50, 24) +
        clock(100, 28),
      ...["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "42", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
      path.join(dir, name),
    ]);
  video("portrait.webm", 464, 848, 20, "140x90", 58);
  video("landscape.webm", 640, 360, 12, "120x70", 40);
  writeFileSync(path.join(dir, "broken.webm"), readFileSync(path.join(dir, "portrait.webm")).subarray(0, 3000));
  const frame = (t) => {
    const out = path.join(dir, `frame_${t}.jpg`);
    execFileSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-ss", String(t), "-i", path.join(dir, "portrait.webm"), "-frames:v", "1", "-q:v", "5", out]);
    return readFileSync(out).toString("base64");
  };
  const note = "[СИНТЕТИКА] Тестовые данные U06 для проверки интерфейса — не результат анализа.";
  const report = {
    job_id: "u06-synthetic",
    status: "done",
    verdict: "violation",
    violations: [
      {
        type: "[СИНТЕТИКА] Сейф оставлен открытым",
        type_id: "V03",
        timecode_sec: 6.0,
        t_start: 5.0,
        t_end: 9.0,
        description: note,
        evidence_frames: [
          { timestamp_sec: 4.0, image_base64: frame(4) },
          { timestamp_sec: 6.0, image_base64: frame(6) },
          { timestamp_sec: 7.0, image_base64: "<jpeg omitted>" }, // сервер сообщил кадр без изображения
          { timestamp_sec: 8.0, image_base64: "/9j/4AAQSkZJRgABAQAAAQABAAD" }, // обрезанный JPEG — браузер не декодирует
        ],
      },
      { type: "[СИНТЕТИКА] Эпизод без кадров-доказательств", type_id: "V01", timecode_sec: 11.0, t_start: 10.5, t_end: 12.0, description: note, evidence_frames: [] },
      {
        type:
          "[СИНТЕТИКА] Очень длинное название эпизода для проверки переноса подписи: посторонний человек находится в кассовом помещении рядом с открытым сейфом и наличными, без инкассаторской сумки, дверь помещения не закрыта, кассир отсутствует на рабочем месте",
        type_id: "V01",
        timecode_sec: 14.0,
        t_start: 13.5,
        t_end: 16.5,
        description: note,
        evidence_frames: [
          { timestamp_sec: 14.0, image_base64: frame(14) },
          { timestamp_sec: 16.0, image_base64: frame(16) },
        ],
      },
    ],
    timeline: Array.from({ length: 41 }, (_, i) => [i / 2, 0.1]),
    meta: { version: "synthetic-u06", duration: 20, file_name: "synthetic_portrait.webm", warnings: [note] },
  };
  writeFileSync(path.join(dir, "report.json"), JSON.stringify(report));
  const beyond = {
    ...report,
    job_id: "u06-synthetic-long",
    meta: { ...report.meta, duration: 40 },
    violations: [
      ...report.violations,
      { type: "[СИНТЕТИКА] Эпизод за пределами видео", type_id: "V01", timecode_sec: 30, t_start: 29, t_end: 31, description: note, evidence_frames: [] },
    ],
  };
  writeFileSync(path.join(dir, "report-beyond.json"), JSON.stringify(beyond));
}

const FIX = process.env.U06_FIX ?? mkdtempSync(path.join(os.tmpdir(), "u06-fix-"));
if (!process.env.U06_FIX) makeFixtures(FIX);
const OUT = process.env.U06_OUT ?? path.join(os.tmpdir(), "u06-shots");
mkdirSync(OUT, { recursive: true });
const pw = await loadPlaywright();
const { server, url } = await startPreview(4191);
const browser = await launch(pw);
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok: !!ok, detail });
  console.log(ok ? "PASS" : "FAIL", "·", name, detail ? `· ${detail}` : "");
};
const SIZES = [
  [1440, 900],
  [768, 1024],
  [360, 780],
];

async function open(width, height, { hash = "", demo = false, theme = "light", reduced = false } = {}) {
  const ctx = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: 1, colorScheme: theme, reducedMotion: reduced ? "reduce" : "no-preference" });
  await ctx.addInitScript((t) => {
    try {
      localStorage.setItem("ks.theme", t);
    } catch {
      /* */
    }
  }, theme);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${url}${demo ? "?demo=1" : ""}${hash}`);
  await page.waitForTimeout(600);
  return { ctx, page, errors };
}

async function replay(page, video, report) {
  if (video) await page.setInputFiles('input[aria-label="Видео для отчёта"]', path.join(FIX, video));
  await page.setInputFiles('input[aria-label="Отчёт JSON"]', path.join(FIX, report));
  await page.waitForSelector("text=Эпизоды с признаками", { timeout: 8000 }).catch(() => undefined);
  await page.waitForTimeout(500);
}

const noHScroll = (page) => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
const stageInfo = (page) =>
  page.evaluate(() => {
    const st = document.querySelector('[class*="_stage_"]');
    if (!st) return null;
    const media = st.querySelector("video, img");
    const r = st.getBoundingClientRect();
    const m = media?.getBoundingClientRect();
    const v = media instanceof HTMLVideoElement ? media : null;
    return {
      cls: st.className,
      stage: { w: r.width, h: r.height, top: r.top, bottom: r.bottom },
      media: m ? { w: m.width, h: m.height } : null,
      fit: media ? getComputedStyle(media).objectFit : null,
      natural: v ? { w: v.videoWidth, h: v.videoHeight } : media instanceof HTMLImageElement ? { w: media.naturalWidth, h: media.naturalHeight } : null,
      play: (() => {
        const b = document.querySelector('button[aria-label="Воспроизвести"], button[aria-label="Пауза"]');
        const br = b?.getBoundingClientRect();
        return br ? { top: br.top, bottom: br.bottom, disabled: (b).disabled } : null;
      })(),
      vh: window.innerHeight,
    };
  });
const playerSection = (page) => page.locator("section.card.no-print").first();
const time = (page) => page.evaluate(() => document.querySelector('[class*="_time_"]')?.textContent ?? "");

try {
  // ------------------------------------------------------------ демо: синтетическая сцена, 3 ширины
  for (const [w, h] of SIZES) {
    const { ctx, page, errors } = await open(w, h, { demo: true, hash: "#/job/demo" });
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.waitForTimeout(300);
    const si = await stageInfo(page);
    check(`demo ${w}: сцена помещается в окно, кнопка ▶ видна`, si && si.stage.h <= si.vh && si.play && si.play.bottom <= si.vh, JSON.stringify({ stage: si?.stage, play: si?.play }));
    check(`demo ${w}: кадр целиком (object-fit: contain)`, si?.fit === "contain");
    check(`demo ${w}: нет горизонтальной прокрутки`, await noHScroll(page));
    check(`demo ${w}: подпись «синтетическая сцена · демо»`, await page.locator("text=синтетическая сцена · демо").isVisible());
    // переход к эпизоду с карточки: плашка эпизода на сцене; на широком экране — в поле, не поверх кадра
    await page.locator("article#episode-1 header button").first().click();
    await page.waitForTimeout(500);
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.waitForTimeout(200);
    const chip = await page.evaluate(() => {
      const st = document.querySelector('[class*="_stage_"]');
      const c = document.querySelector('[class*="_episodeChip_"]');
      const b = document.querySelector('[class*="_badge_"]');
      const m = st?.querySelector("img, video");
      if (!st || !c || !m) return null;
      const r = st.getBoundingClientRect();
      const ar = m instanceof HTMLVideoElement ? m.videoWidth / m.videoHeight : m.naturalWidth / m.naturalHeight;
      const cw = Math.min(r.width, r.height * ar);
      const left = r.left + (r.width - cw) / 2;
      const right = left + cw;
      const over = (e) => {
        const x = e.getBoundingClientRect();
        return x.left < right - 0.5 && x.right > left + 0.5;
      };
      return { chipOverFrame: over(c), badgeOverFrame: b ? over(b) : null, compact: /Compact/.test(c.className), text: c.textContent };
    });
    if (w === 1440) check(`demo ${w}: плашки эпизода и демо в полях, кадр не закрыт`, chip && !chip.chipOverFrame && !chip.badgeOverFrame && !chip.compact, JSON.stringify(chip));
    else check(`demo ${w}: поверх кадра — компактная плашка эпизода (номер и интервал)`, chip && chip.compact, JSON.stringify(chip));
    await page.screenshot({ path: path.join(OUT, `demo-player-${w}.png`) });

    // фокус клавиатуры виден на элементе управления
    await page.locator('button[aria-label="Вперёд 5 с"]').focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    const ring = await page.evaluate(() => {
      const a = document.activeElement;
      const cs = a ? getComputedStyle(a) : null;
      return { label: a?.getAttribute("aria-label"), focusVisible: a?.matches(":focus-visible"), shadow: cs?.boxShadow, outline: cs?.outlineStyle };
    });
    check(`demo ${w}: фокус клавиатуры виден`, ring.focusVisible && ((ring.shadow && ring.shadow !== "none") || ring.outline !== "none"), JSON.stringify(ring));
    if (w === 1440) await playerSection(page).screenshot({ path: path.join(OUT, `demo-focus-${w}.png`) });

    // лайтбокс: фокус, Tab внутри, Esc, возврат фокуса
    const opener = page.locator("article button[aria-label^='Открыть кадр']").first();
    await opener.scrollIntoViewIfNeeded();
    await opener.focus();
    await page.keyboard.press("Enter");
    await page.waitForSelector('[role="dialog"]');
    await page.waitForTimeout(300);
    const dlg = page.locator('[role="dialog"]');
    const focusClose = await page.evaluate(() => document.activeElement?.getAttribute("aria-label"));
    check(`demo ${w}: лайтбокс — фокус на «Закрыть»`, focusClose === "Закрыть просмотр", String(focusClose));
    let inside = true;
    for (let i = 0; i < 14; i++) {
      await page.keyboard.press(i % 5 === 4 ? "Shift+Tab" : "Tab");
      inside &&= await page.evaluate(() => !!document.querySelector('[role="dialog"]')?.contains(document.activeElement));
    }
    check(`demo ${w}: лайтбокс — Tab/Shift+Tab не выходят из диалога`, inside);
    const lb = await page.evaluate(() => {
      const d = document.querySelector('[role="dialog"]');
      const img = d?.querySelector("img[alt^='Кадр-доказательство']");
      const r = img?.getBoundingClientRect();
      const t = d?.querySelector("h2");
      return {
        img: r ? { top: r.top, bottom: r.bottom, left: r.left, right: r.right } : null,
        vw: innerWidth,
        vh: innerHeight,
        titleCut: t ? t.scrollWidth > t.clientWidth + 1 : null,
        bg: d ? getComputedStyle(d).backgroundColor : null,
        blurClaim: d?.textContent?.includes("размыты") ?? null,
        demoLabel: d?.textContent?.includes("демо · синтетический кадр") ?? null,
        buttons: [...(d?.querySelectorAll("button") ?? [])].map((b) => {
          const br = b.getBoundingClientRect();
          return br.right <= innerWidth + 1 && br.bottom <= innerHeight + 1 && br.left >= -1 && br.top >= -1;
        }),
      };
    });
    check(`demo ${w}: лайтбокс — кадр целиком в окне`, lb.img && lb.img.top >= 0 && lb.img.left >= 0 && lb.img.right <= lb.vw && lb.img.bottom <= lb.vh, JSON.stringify(lb.img));
    check(`demo ${w}: лайтбокс — все кнопки в пределах экрана`, lb.buttons.every(Boolean), JSON.stringify(lb.buttons));
    check(`demo ${w}: лайтбокс — подпись не обрезана, нет утверждения о размытии, демо подписано`, lb.titleCut === false && lb.blurClaim === false && lb.demoLabel === true, JSON.stringify(lb));
    await page.screenshot({ path: path.join(OUT, `demo-lightbox-${w}.png`) });
    // навигация клавишами — по времени кадра (демо-сервер присылает 24.6, 24.2, 29.0)
    const meta = async () => (await dlg.locator("p").first().textContent()) ?? "";
    const m1 = await meta();
    await page.keyboard.press("ArrowLeft");
    await page.waitForTimeout(150);
    const m2 = await meta();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(150);
    const m3 = await meta();
    check(
      `demo ${w}: лайтбокс — ←/→ листают по времени`,
      /Кадр 2 из 3 · 00:24\.6/.test(m1) && /Кадр 1 из 3 · 00:24\.2/.test(m2) && /Кадр 3 из 3 · 00:29\.0/.test(m3),
      `${m1} | ${m2} | ${m3}`,
    );
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);
    const after = await page.evaluate(() => ({ open: !!document.querySelector('[role="dialog"]'), label: document.activeElement?.getAttribute("aria-label") }));
    check(`demo ${w}: Esc закрывает, фокус возвращается на кадр`, !after.open && /^Открыть кадр/.test(after.label ?? ""), JSON.stringify(after));
    // закрытие кнопкой
    await opener.click();
    await page.waitForSelector('[role="dialog"]');
    await page.locator('button[aria-label="Закрыть просмотр"]').click();
    check(`demo ${w}: кнопка «Закрыть»`, (await page.locator('[role="dialog"]').count()) === 0);
    check(`demo ${w}: без ошибок JS`, errors.length === 0, errors.join(" | ").slice(0, 300));
    await ctx.close();
  }

  // ------------------------------------------------------------ reduced motion
  {
    const { ctx, page } = await open(1440, 900, { demo: true, hash: "#/job/demo", reduced: true });
    await page.locator("article button[aria-label^='Открыть кадр']").first().click();
    await page.waitForSelector('[role="dialog"]');
    const anim = await page.evaluate(() => getComputedStyle(document.querySelector('[role="dialog"]')).animationName);
    check("reduced motion: у лайтбокса нет анимации", anim === "none", anim);
    await ctx.close();
  }

  // ------------------------------------------------------------ тёмная тема
  {
    const { ctx, page } = await open(1440, 900, { demo: true, hash: "#/job/demo", theme: "dark" });
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(OUT, "demo-player-1440-dark.png") });
    await ctx.close();
  }

  // ------------------------------------------------------------ replay: портретное видео + краевые случаи
  for (const [w, h] of SIZES) {
    const { ctx, page, errors } = await open(w, h, { hash: "#/replay" });
    await replay(page, "portrait.webm", "report.json");
    await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 1, null, { timeout: 8000 });
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.waitForTimeout(300);
    const si = await stageInfo(page);
    check(`replay ${w}: портретное видео 464×848 распознано`, si?.natural?.w === 464 && si?.natural?.h === 848, JSON.stringify(si?.natural));
    check(`replay ${w}: портрет — сцена по высоте окна, без обрезки`, si && !/Landscape/.test(si.cls) && si.fit === "contain" && si.stage.h <= si.vh, JSON.stringify(si?.stage));
    check(`replay ${w}: кнопка ▶ на экране`, si?.play && si.play.bottom <= si.vh && !si.play.disabled);
    check(`replay ${w}: нет горизонтальной прокрутки`, await noHScroll(page));
    // переход к эпизоду с карточки
    await page.locator("article#episode-1 header button", { hasText: "00:06.0" }).click();
    await page.waitForTimeout(700);
    const cur = await page.evaluate(() => document.querySelector("video")?.currentTime ?? -1);
    check(`replay ${w}: «▶ 00:06.0» переводит видео к 6.0 с`, Math.abs(cur - 6) < 0.15, String(cur));
    check(`replay ${w}: на сцене выбранный эпизод`, await page.locator('[class*="_episodeChip_"]').isVisible());
    await page.screenshot({ path: path.join(OUT, `replay-portrait-${w}.png`) });
    // битый кадр: плашка, без подмены; кадр без изображения — плашка карточки
    const broken = page.locator("article#episode-1 [role='img'][aria-label*='изображение не загрузилось']");
    check(`replay ${w}: битый кадр — явная плашка «Кадр не загрузился»`, (await broken.count()) === 1);
    check(`replay ${w}: кадр без изображения — «кадр не передан»`, await page.locator("article#episode-1 >> text=кадр не передан").isVisible());
    check(`replay ${w}: эпизод без кадров — причина`, await page.locator("article#episode-2 >> text=Сервер не передал кадры-доказательства").isVisible());
    // только для склейки длинного скриншота: липкая шапка приложения не должна попадать в середину карточки
    const unstick = await page.addStyleTag({ content: ".sticky{position:static!important}" });
    await page.locator("article#episode-1").screenshot({ path: path.join(OUT, `replay-evidence-${w}.png`) });
    await unstick.evaluate((n) => n.remove());
    // лайтбокс на битом кадре: плашка и в большом просмотре
    await page.locator("article#episode-1 [role='img'][aria-label*='изображение не загрузилось']").first().click();
    await page.waitForSelector('[role="dialog"]');
    await page.waitForTimeout(300);
    check(`replay ${w}: лайтбокс на битом кадре — плашка, не изображение`, await page.locator("[role='dialog'] >> text=Кадр не загрузился").isVisible());
    await page.screenshot({ path: path.join(OUT, `replay-lightbox-broken-${w}.png`) });
    await page.keyboard.press("Escape");
    // длинная подпись
    await page.locator("article#episode-3 button[aria-label^='Открыть кадр']").first().click();
    await page.waitForSelector('[role="dialog"]');
    await page.waitForTimeout(300);
    const cap = await page.evaluate(() => {
      const t = document.querySelector('[role="dialog"] h2');
      const box = t?.parentElement;
      return { lines: t ? Math.round(t.getBoundingClientRect().height / parseFloat(getComputedStyle(t).lineHeight)) : 0, cut: t ? t.scrollWidth > t.clientWidth + 1 : true, scrollable: box ? box.scrollHeight > box.clientHeight : false };
    });
    check(`replay ${w}: длинная подпись переносится, не обрезается`, cap.lines >= 2 && !cap.cut, JSON.stringify(cap));
    await page.screenshot({ path: path.join(OUT, `replay-lightbox-long-caption-${w}.png`) });
    // «Перейти к … в видео» из лайтбокса
    await page.locator("[role='dialog'] button", { hasText: "Перейти к" }).click();
    await page.waitForTimeout(900);
    const cur2 = await page.evaluate(() => document.querySelector("video")?.currentTime ?? -1);
    check(`replay ${w}: «Перейти к 00:14.0 в видео» из лайтбокса`, Math.abs(cur2 - 14) < 0.15, String(cur2));
    check(`replay ${w}: без ошибок JS`, errors.length === 0, errors.join(" | ").slice(0, 300));
    await ctx.close();
  }

  // ------------------------------------------------------------ «Смотреть эпизод»: проигрывание интервала и остановка
  {
    const { ctx, page } = await open(1440, 900, { hash: "#/replay" });
    await replay(page, "portrait.webm", "report.json");
    await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 1);
    await page.locator("button", { hasText: "Смотреть эпизод" }).click(); // эпизод 1: 5.0–9.0, запас 2 / 1 с → 3.0–10.0
    await page.waitForTimeout(400);
    const playing = await page.evaluate(() => !document.querySelector("video")?.paused);
    await page.waitForFunction(() => document.querySelector("video")?.paused, null, { timeout: 15000 }).catch(() => undefined);
    const end = await page.evaluate(() => document.querySelector("video")?.currentTime ?? -1);
    check("playRange: играет и останавливается в конце интервала (≈10.0 с)", playing && end >= 9.8 && end <= 10.6, `playing=${playing} end=${end}`);
    await ctx.close();
  }

  // ------------------------------------------------------------ альбомное видео: без пустых полос
  {
    const { ctx, page } = await open(1440, 900, { hash: "#/replay" });
    await replay(page, "landscape.webm", "report.json");
    await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 1);
    await page.waitForTimeout(300);
    const si = await stageInfo(page);
    const ratio = si ? si.stage.w / si.stage.h : 0;
    check("альбомное видео 640×360: сцена по пропорциям кадра (без полос)", si && /Landscape/.test(si.cls) && Math.abs(ratio - 16 / 9) < 0.05, `ratio=${ratio.toFixed(3)}`);
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.screenshot({ path: path.join(OUT, "replay-landscape-1440.png") });
    await ctx.close();
  }

  // ------------------------------------------------------------ видео нет: переход не изображает успех
  for (const [w, h] of [
    [1440, 900],
    [360, 780],
  ]) {
    const { ctx, page } = await open(w, h, { hash: "#/replay" });
    await replay(page, null, "report.json");
    await page.locator("article#episode-1 header button", { hasText: "00:06.0" }).click();
    await page.waitForTimeout(500);
    const msg = await page.locator('[role="alert"]').first().textContent().catch(() => "");
    const tc = await time(page);
    check(`без видео ${w}: причина на сцене`, await page.getByText("Исходное видео недоступно для этой задачи", { exact: true }).isVisible());
    check(
      `без видео ${w}: «перемотка не выполнена» с причиной; момент отмечен на шкале и подписан`,
      /Перемотка не выполнена: исходное видео недоступно для этой задачи\. На шкале риска отмечен момент 00:06\.0/.test(msg ?? "") &&
        tc.startsWith("00:06.0") &&
        (await page.getByText("видео недоступно — время по шкале").isVisible()),
      `${msg} | ${tc}`,
    );
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.screenshot({ path: path.join(OUT, `no-video-${w}.png`) });
    await ctx.close();
  }

  // ------------------------------------------------------------ видео не декодируется
  {
    const { ctx, page } = await open(1440, 900, { hash: "#/replay" });
    await replay(page, "broken.webm", "report.json");
    await page.waitForSelector("text=Браузер не смог воспроизвести видео", { timeout: 8000 }).catch(() => undefined);
    const shown = await page.getByText("Браузер не смог воспроизвести видео", { exact: true }).isVisible();
    const reason = await page.locator("text=Причина:").first().textContent().catch(() => "");
    await page.locator("article#episode-1 header button", { hasText: "00:06.0" }).click();
    await page.waitForTimeout(400);
    const msg = await page.locator('[role="alert"]').first().textContent().catch(() => "");
    check("битое видео: причина показана", shown && /Причина: /.test(reason ?? ""), reason ?? "");
    check("битое видео: «перемотка не выполнена» с причиной", /Перемотка не выполнена: видео не воспроизводится в этом браузере/.test(msg ?? ""), msg ?? "");
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.screenshot({ path: path.join(OUT, "broken-video-1440.png") });
    await ctx.close();
  }

  // ------------------------------------------------------------ момент за пределами записи
  {
    const { ctx, page } = await open(1440, 900, { hash: "#/replay" });
    await replay(page, "portrait.webm", "report-beyond.json");
    await page.waitForFunction(() => (document.querySelector("video")?.readyState ?? 0) >= 1);
    await page.locator("article#episode-4 header button", { hasText: "00:30.0" }).click();
    await page.waitForTimeout(600);
    const msg = await page.locator('[role="status"]').filter({ hasText: "за пределами записи" }).first().textContent().catch(() => "");
    check("момент вне записи: сказано прямо, показан конец", /Момент 00:30\.0 за пределами записи \(длительность 00:20\.0\)/.test(msg ?? ""), msg ?? "");
    await playerSection(page).evaluate((el) => el.scrollIntoView({ block: "start", behavior: "instant" }));
    await page.screenshot({ path: path.join(OUT, "beyond-duration-1440.png") });
    await ctx.close();
  }
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} PASS · скриншоты: ${OUT} · фикстуры: ${FIX}`);
if (failed.length) process.exitCode = 1;
