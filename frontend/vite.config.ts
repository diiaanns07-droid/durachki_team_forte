/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Бэкенд (A02) в dev-режиме: VITE_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
const target = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  // A02 раздаёт dist с корня и отдаёт index.html на любой путь (SPA-fallback): абсолютный base
  // не ломает ассеты при заходе на /что-угодно. Для подкаталога: VITE_BASE=/kassa/ npm run build.
  // Роутинг — через hash (#/job/<id>), поэтому HTML всегда один.
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    // Шрифты/иконки — в бандле, без CDN (офлайн-контур банка).
    assetsInlineLimit: 4096,
  },
  server: {
    port: 5173,
    // Только маршруты API: префикс "/ui" перехватил бы и статический /ui-config.json.
    proxy: {
      "/api/": { target, changeOrigin: true },
      "/ui/v1/": { target, changeOrigin: true },
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
