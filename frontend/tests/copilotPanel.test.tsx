// CopilotPanel (U08): SSR-рендер состояний и чистые функции. Только синтетические данные (демо-результат).
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DemoBackend, buildDemoClean, buildDemoResult } from "../src/api/demo";
import type { JobResult } from "../src/api/types";
import { AppContext, type AppContextValue } from "../src/context";
import CopilotPanel, { copilotInternals } from "../src/components/CopilotPanel";

const { historyBefore, quickQuestions, MAX_HISTORY_TURNS, MAX_TURN_CHARS } = copilotInternals;

function ctx(over: Partial<AppContextValue> = {}): AppContextValue {
  return {
    backend: new DemoBackend(),
    demo: false,
    session: { token: null, reachable: true, authenticated: true, features: {} },
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

const render = (jobId: string, over?: Partial<AppContextValue>) =>
  renderToString(
    <AppContext.Provider value={ctx(over)}>
      <CopilotPanel jobId={jobId} />
    </AppContext.Provider>,
  );

describe("CopilotPanel: состояния", () => {
  it("выключен/недоступен — компактная карточка без имитации ответа", () => {
    const html = render("demo", { copilotEnabled: false });
    expect(html).toContain("Copilot недоступен");
    expect(html).toContain("доступны без него");
    expect(html).not.toContain("Copilot проверяющего");
    expect(html).not.toContain("<textarea");
  });

  it("доступен: заголовок, поле вопроса, честная граница данных", () => {
    const html = render("demo");
    expect(html).toContain("Copilot проверяющего");
    expect(html).toContain("<textarea");
    expect(html).toContain("Видео и изображения кадров не просматривает");
    expect(html).toContain("Enter — отправить");
  });

  it("демо явно подписано", () => {
    expect(render("demo", { demo: true })).toContain("Демо: ответы заготовлены");
    expect(render("demo")).not.toContain("Демо: ответы заготовлены");
  });

  it("быстрые вопросы — только по результату выбранного анализа", () => {
    const job = buildDemoResult("job-a");
    const same = render("job-a", { lastResult: job });
    expect(same).toContain("Объяснить эпизод 1");
    expect(same).toContain("Перечислить кадры-доказательства");
    const other = render("job-b", { lastResult: job });
    expect(other).not.toContain("Объяснить эпизод 1");
    expect(other).not.toContain("Спросить по отчёту");
  });
});

describe("CopilotPanel: быстрые вопросы", () => {
  it("эпизоды → объяснение с таймкодом; без норм нет вопроса об основании", () => {
    const job: JobResult = {
      ...buildDemoResult("x"),
      violations: buildDemoResult("x").violations.map((v) => ({ ...v, rules: [] })),
    };
    const quick = quickQuestions(job);
    expect(quick[0].label).toMatch(/^Эпизод 1 · \d\d:\d\d\.\d$/);
    expect(quick[0].ariaLabel).toMatch(/^Объяснить эпизод 1, /);
    expect(quick[0].question).toContain("Объясни эпизод 1");
    expect(quick.map((q) => q.label)).not.toContain("Указанное основание");
  });

  it("нет нарушений → только вопрос «почему нарушений не найдено»", () => {
    const labels = quickQuestions(buildDemoClean("c")).map((q) => q.label);
    expect(labels).toEqual(["Почему без нарушений"]);
  });

  it("результат не завершён или неизвестен → вопросов нет", () => {
    expect(quickQuestions(null)).toEqual([]);
    expect(quickQuestions({ ...buildDemoResult("p"), status: "processing" })).toEqual([]);
  });
});

describe("CopilotPanel: история для сервера (протокол не меняется)", () => {
  type M = Parameters<typeof historyBefore>[0][number];
  const long: M[] = [];
  for (let i = 1; i <= 50; i++) {
    long.push(i % 2 ? { id: i, role: "user", text: `q${i}` } : { id: i, role: "assistant", text: `a${i}`, sources: [] });
  }

  it("длинная история обрезается до последних ходов (лимит сервера — 40)", () => {
    const h = historyBefore(long);
    expect(h).toHaveLength(MAX_HISTORY_TURNS);
    expect(MAX_HISTORY_TURNS).toBeLessThanOrEqual(40);
    expect(h[h.length - 1]).toEqual({ role: "assistant", text: "a50" });
  });

  it("уведомления об ошибках в историю не попадают; повтор берёт ходы до своего вопроса", () => {
    const msgs: M[] = [
      { id: 1, role: "user", text: "первый" },
      { id: 2, role: "assistant", text: "ответ", sources: [] },
      { id: 3, role: "user", text: "второй" },
      { id: 4, role: "notice", tone: "error", text: "503", questionId: 3, question: "второй" },
    ];
    expect(historyBefore(msgs)).toEqual([
      { role: "user", text: "первый" },
      { role: "assistant", text: "ответ" },
      { role: "user", text: "второй" },
    ]);
    expect(historyBefore(msgs, 3)).toEqual([
      { role: "user", text: "первый" },
      { role: "assistant", text: "ответ" },
    ]);
  });

  it("слишком длинный ход режется до лимита сервера", () => {
    const h = historyBefore([{ id: 1, role: "assistant", text: "я".repeat(MAX_TURN_CHARS + 500), sources: [] }]);
    expect(h[0].text).toHaveLength(MAX_TURN_CHARS);
  });
});
