export type ThemePref = "system" | "light" | "dark";

export function getThemePref(): ThemePref {
  try {
    const v = localStorage.getItem("ks.theme");
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(pref: ThemePref) {
  try {
    if (pref === "system") localStorage.removeItem("ks.theme");
    else localStorage.setItem("ks.theme", pref);
  } catch {
    /* noop */
  }
  const dark =
    pref === "dark" || (pref === "system" && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", !!dark);
}

export function isDarkNow(): boolean {
  return typeof document !== "undefined" && document.documentElement.classList.contains("dark");
}
