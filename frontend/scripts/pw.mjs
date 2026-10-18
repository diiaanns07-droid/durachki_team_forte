// Общие хелперы для скриптов скриншотов/мок-e2e (Playwright — опционально, не в зависимостях).
import { createRequire } from "node:module";
import { existsSync } from "node:fs";

export async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    const require = createRequire(import.meta.url);
    const candidates = [process.env.PLAYWRIGHT_MODULE, "/opt/node22/lib/node_modules/playwright"].filter(Boolean);
    for (const c of candidates) {
      try {
        return require(c);
      } catch {
        /* next */
      }
    }
    throw new Error("Playwright не найден: npm i -D playwright или PLAYWRIGHT_MODULE=/path/to/playwright");
  }
}

export async function launch(pw) {
  const exe = process.env.CHROMIUM_PATH ?? "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
  return pw.chromium.launch(existsSync(exe) ? { executablePath: exe } : {});
}

export async function startPreview(port = 4173) {
  const { preview } = await import("vite");
  const server = await preview({ preview: { port, strictPort: false, host: "127.0.0.1" }, logLevel: "warn" });
  const url = server.resolvedUrls?.local?.[0] ?? `http://127.0.0.1:${port}/`;
  return { server, url };
}
