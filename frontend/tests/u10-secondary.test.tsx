// U10: SettingsDialog / AboutScreen — статический рендер без браузера (поведение фокуса — scripts/u10-check.mjs).
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { DemoBackend } from "../src/api/demo";
import { settings } from "../src/api/client";
import { AppContext, type AppContextValue } from "../src/context";
import AboutScreen from "../src/components/AboutScreen";
import SettingsDialog, { SettingsPanel, validateApiBase } from "../src/components/SettingsDialog";

function ctx(over: Partial<AppContextValue> = {}): AppContextValue {
  return {
    backend: new DemoBackend(),
    demo: false,
    session: { token: "t", reachable: true, authenticated: true, features: {} },
    serverOk: true,
    authenticated: true,
    login: async () => undefined,
    logout: async () => undefined,
    copilotEnabled: true,
    disableCopilot: () => undefined,
    refreshSession: async () => undefined,
    toast: () => undefined,
    openSettings: () => undefined,
    lastResult: null,
    setLastResult: () => undefined,
    ...over,
  };
}
const wrap = (node: ReactNode, over: Partial<AppContextValue> = {}) =>
  renderToString(<AppContext.Provider value={ctx(over)}>{node}</AppContext.Provider>);
const panel = (over: Partial<AppContextValue> = {}) => wrap(<SettingsPanel onClose={() => undefined} onSaved={() => undefined} />, over);

afterEach(() => {
  settings.apiKey = "";
});

describe("validateApiBase", () => {
  it.each([
    ["", null],
    ["   ", null],
    ["/kassa", null],
    ["http://127.0.0.1:8000", null],
    ["https://sentinel.local/", null],
  ])("допустимо: %j", (v, want) => expect(validateApiBase(v)).toBe(want));
  it.each([["127.0.0.1:8000"], ["localhost"], ["ftp://host"], ["http://a b"], ["/ka ssa"], ["http://"]])("ошибка: %j", (v) =>
    expect(validateApiBase(v)).toEqual(expect.any(String)),
  );
});

describe("SettingsPanel", () => {
  it("диалог с заголовком, описанием и группами существующих полей", () => {
    const html = panel();
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toMatch(/aria-labelledby="[^"]+"/);
    for (const t of ["Подключение к серверу", "API-ключ — запасной вход", "Данные этого браузера", "Адрес сервера (API)", "Сохранить", "Отмена"])
      expect(html).toContain(t);
    expect(html).toContain("Выйти из сессии");
  });
  it("заданный ключ не выводится в разметку и не подставляется в поле", () => {
    settings.apiKey = "very-secret-key-123";
    const html = panel();
    expect(html).not.toContain("very-secret-key-123");
    expect(html).toContain("Значение не показывается");
    expect(html).toContain(">Заменить<");
    expect(html).toContain(">Удалить<");
  });
  it("без ключа — поле типа password, без кнопки «показать»", () => {
    const html = panel();
    expect(html).toMatch(/<input[^>]+type="password"/);
    expect(html).not.toMatch(/Показать ключ/);
  });
  it("демо подписано; выход скрыт", () => {
    const html = panel({ demo: true });
    expect(html).toContain("Синтетическое демо");
    expect(html).not.toContain("Выйти из сессии");
  });
  it("сервер недоступен и проверка связи — заметные состояния, не «всё хорошо»", () => {
    expect(panel({ serverOk: false })).toContain("Сервер анализа не отвечает");
    expect(panel({ serverOk: null })).toContain("Проверяем связь с сервером");
    const noLogin = panel({ session: { token: null, reachable: true, authenticated: false, features: {} } });
    expect(noLogin).toContain("Вход не выполнен");
    expect(noLogin).not.toContain("Сервер доступен");
  });
  it("Сохранить без изменений недоступна", () => {
    expect(panel()).toMatch(/<button type="submit"[^>]*disabled=""/);
  });
  it("закрытый диалог ничего не рендерит", () => {
    expect(wrap(<SettingsDialog open={false} onClose={() => undefined} onSaved={() => undefined} />)).toBe("");
  });
});

describe("AboutScreen", () => {
  it("назначение, процесс, ограничения; типы и лицензии на месте", () => {
    const html = wrap(<AboutScreen />);
    for (const t of ["О системе", "Как проходит анализ", "Ограничения", "API для автоматической проверки", "Типы нарушений", "Модели и лицензии"])
      expect(html).toContain(t);
    expect(html).toContain("V01");
    expect(html).toContain("Apache-2.0");
    expect(html).toContain("не юридическое заключение");
    expect(html).toContain("не считается «без нарушений»");
    expect(html).toContain("если включена");
  });
  it("без неподтверждённых обещаний точности, скорости и готовности", () => {
    const html = wrap(<AboutScreen />).replace(/<[^>]+>/g, " ");
    expect(html).not.toMatch(/\d+\s?%/);
    expect(html).not.toMatch(/≤\s?\d|не более \d+ (мин|с)/i);
    expect(html).not.toMatch(/гарантир|точност[ьи] \d|готов[ао] к/i);
    expect(html).not.toContain("головы на кадрах размыты");
  });
  it("модели без данных сервера подписаны как состав по архитектуре, в демо — как демо", () => {
    expect(wrap(<AboutScreen />)).toContain("не подтверждение, что модель загружена");
    const demoHtml = wrap(<AboutScreen />, {
      demo: true,
      lastResult: { job_id: "d", status: "done", violations: [], timeline: [], meta: { models: ["mock-a"] } } as never,
    });
    expect(demoHtml).toContain("Синтетическое демо: список взят из демонстрационного отчёта");
  });
});
