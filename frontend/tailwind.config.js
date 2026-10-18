/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  // U01: строка `!!dark` в src/lib/theme.ts читается сканером как кандидат «!dark» (important-модификатор), и
  // Tailwind дублировал бы все правила `.dark …` из src/index.css как мёртвые `.\!dark …{…!important}`
  // (≈3 КБ). Класса «!dark» на элементах не бывает, поэтому кандидат исключён.
  blocklist: ["!dark"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter Variable"', "Inter", "system-ui", "Segoe UI", "Roboto", "Arial", "sans-serif"],
        display: ['"Manrope Variable"', "Manrope", '"Inter Variable"', "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "Menlo", "monospace"],
      },
      colors: {
        ks: {
          bg: "rgb(var(--ks-bg) / <alpha-value>)",
          surface: "rgb(var(--ks-surface) / <alpha-value>)",
          "surface-2": "rgb(var(--ks-surface-2) / <alpha-value>)",
          "surface-3": "rgb(var(--ks-surface-3) / <alpha-value>)",
          text: "rgb(var(--ks-text) / <alpha-value>)",
          "text-2": "rgb(var(--ks-text-2) / <alpha-value>)",
          "text-3": "rgb(var(--ks-text-3) / <alpha-value>)",
          accent: "rgb(var(--ks-accent) / <alpha-value>)",
          "accent-contrast": "rgb(var(--ks-accent-contrast) / <alpha-value>)",
          focus: "rgb(var(--ks-focus) / <alpha-value>)",
          event: "rgb(var(--ks-event) / <alpha-value>)",
          "event-text": "rgb(var(--ks-event-text) / <alpha-value>)",
          "event-contrast": "rgb(var(--ks-event-contrast) / <alpha-value>)",
          success: "rgb(var(--ks-success) / <alpha-value>)",
          "success-text": "rgb(var(--ks-success-text) / <alpha-value>)",
          warning: "rgb(var(--ks-warning) / <alpha-value>)",
          "warning-text": "rgb(var(--ks-warning-text) / <alpha-value>)",
          info: "rgb(var(--ks-info) / <alpha-value>)",
          "info-text": "rgb(var(--ks-info-text) / <alpha-value>)",
        },
        ink: {
          50: "#f5f7fb",
          100: "#e9edf5",
          200: "#cfd7e6",
          300: "#a8b5cf",
          400: "#7a8bb0",
          500: "#56688f",
          600: "#3f4f74",
          700: "#2f3c5b",
          800: "#1d2740",
          900: "#121a2e",
          950: "#0a0f1d",
        },
        brand: {
          50: "#fbf3f6",
          100: "#f6e3ea",
          200: "#eec8d6",
          300: "#e09db6",
          400: "#cd6a8e",
          500: "#b8436e",
          600: "#9e2c56",
          700: "#842147",
          800: "#6e1d3d",
          900: "#5c1b36",
          950: "#350a1b",
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,.04), 0 4px 16px -4px rgba(16,24,40,.08)",
        lift: "0 10px 40px -12px rgba(16,24,40,.25)",
        "ks-card": "var(--ks-shadow-card)",
        "ks-lift": "var(--ks-shadow-lift)",
        "ks-action": "var(--ks-shadow-action)",
      },
      // U01: семантические токены рабочей области (значения — CSS-переменные в src/index.css, .ks-ws).
      // Работают только внутри контейнера .ks-ws; прежние палитры ink/brand не меняются.
      // Пример: text-ks-event, bg-ks-surface, border-ks, rounded-ks-lg, max-w-ks-content.
      borderColor: {
        ks: "var(--ks-border)",
        "ks-strong": "var(--ks-border-strong)",
        "ks-event": "var(--ks-event-border)",
        "ks-success": "var(--ks-success-border)",
        "ks-warning": "var(--ks-warning-border)",
        "ks-info": "var(--ks-info-border)",
      },
      borderRadius: {
        "ks-sm": "var(--ks-radius-sm)",
        "ks-md": "var(--ks-radius-md)",
        "ks-lg": "var(--ks-radius-lg)",
      },
      maxWidth: {
        "ks-content": "var(--ks-content-max)",
      },
      keyframes: {
        "scan": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(.8)", opacity: ".7" },
          "100%": { transform: "scale(1.6)", opacity: "0" },
        },
      },
      animation: {
        scan: "scan 2.4s ease-in-out infinite alternate",
        // backwards (не both): после анимации transform не остаётся — иначе предок становится опорным блоком
        // для position: fixed (лайтбокс, модалки) и они «съезжают» внутрь карточки.
        "fade-in": "fade-in .35s ease-out backwards",
        "pulse-ring": "pulse-ring 1.6s ease-out infinite",
      },
    },
  },
  plugins: [],
};
