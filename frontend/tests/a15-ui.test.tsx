// A15, переработка интерфейса: экран задачи, «до / после», подсветка, навигатор эпизодов, повторный просмотр.
// SSR-рендер на синтетике: смысл данных не меняется, отсутствующее не подменяется, ничего не выдумывается.
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { DemoBackend } from "../src/api/demo";
import type { Violation } from "../src/api/types";
import { AppContext, type AppContextValue } from "../src/context";
import { BeforeAfter, EvidenceImage } from "../src/components/Evidence";
import EpisodeNavigator from "../src/components/EpisodeNavigator";
import JobScreen from "../src/components/JobScreen";
import ReplayScreen from "../src/components/ReplayScreen";

const JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD";

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
const wrap = (node: ReactNode) => renderToString(<AppContext.Provider value={ctx()}>{node}</AppContext.Provider>);
/** Текст без тегов и комментариев React (<!-- -->), с нормализованными пробелами. */
const text = (html: string) =>
  html
    .replace(/<!--.*?-->/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&quot;/g, '"')
    .replace(/\s+/g, " ")
    .trim();

const ep = (i: number, over: Partial<Violation> = {}): Violation => ({
  type: `Синтетический тип ${i + 1}`,
  type_id: "V01",
  timecode_sec: 10 + i * 10,
  description: "",
  evidence_frames: [],
  ...over,
});

describe("EpisodeNavigator", () => {
  const noop = () => undefined;
  it("счётчики, кнопки ‹ ›, «Смотреть эпизод», отметка — тексты, на которые опираются e2e", () => {
    const html = renderToString(
      <EpisodeNavigator violations={[ep(0), ep(1)]} current={0} reviewed={new Set([1])} onGo={noop} onPlay={noop} onToggleReviewed={noop} />,
    );
    const t = text(html);
    expect(t).toContain("Эпизод 1 из 2 · просмотрено 1 из 2");
    expect(t).toContain("Смотреть эпизод");
    expect(t).toContain("Отметить просмотренным");
    expect(html).toMatch(/aria-label="Предыдущий эпизод"[^>]*disabled|disabled[^>]*aria-label="Предыдущий эпизод"/);
    expect(html).toContain('aria-label="Следующий эпизод"');
    expect(html).toContain('aria-pressed="false"');
  });
  it("название типа не обрезается (без truncate), длинное — переносится", () => {
    const long = "Очень длинное синтетическое название типа эпизода, которое не помещается в одну строку на телефоне";
    const html = renderToString(
      <EpisodeNavigator violations={[ep(0, { type: long })]} current={0} reviewed={new Set()} onGo={noop} onPlay={noop} onToggleReviewed={noop} />,
    );
    expect(text(html)).toContain(long);
    expect(html).not.toContain("truncate");
  });
  it("шкала просмотра: сегмент на эпизод (только наглядность, aria-hidden); очень много эпизодов — сплошная полоса", () => {
    const few = renderToString(
      <EpisodeNavigator violations={[ep(0), ep(1), ep(2)]} current={1} reviewed={new Set([0])} onGo={noop} onPlay={noop} onToggleReviewed={noop} />,
    );
    expect(few.match(/data-current="true"/g)).toHaveLength(1);
    expect(few.match(/data-reviewed="true"/g)).toHaveLength(1);
    const many = Array.from({ length: 60 }, (_, i) => ep(i));
    const lots = renderToString(<EpisodeNavigator violations={many} current={0} reviewed={new Set()} onGo={noop} onPlay={noop} onToggleReviewed={noop} />);
    expect(lots).not.toContain("data-current");
    expect(lots).toContain("width:0%");
  });
  it("нет эпизодов — ничего не рисуется", () => {
    expect(renderToString(<EpisodeNavigator violations={[]} current={0} reviewed={new Set()} onGo={noop} onPlay={noop} onToggleReviewed={noop} />)).toBe("");
  });
});

describe("BeforeAfter", () => {
  const base = { type: "Переход", type_id: "V02", timecode_sec: 21.5, t_start: 20, t_end: 26, description: "" };
  it("кадры сервера «до» и «после»: время на плитке, промежуток из меток времени", () => {
    const v = { ...base, evidence_frames: [{ timestamp_sec: 17.5, image_base64: JPEG }, { timestamp_sec: 21.5, image_base64: JPEG }] };
    const html = renderToString(<BeforeAfter v={v} peakT={21.5} boxes={[]} highlight onSeek={() => undefined} onOpen={() => undefined} />);
    expect(html).toContain("Открыть кадр «до» (00:17.5)");
    expect(html).toContain("Открыть кадр «после» (00:21.5)");
    expect(text(html)).toContain("+4,0 с");
    expect(html).not.toContain("без размытия"); // пометка — только для кадра, взятого из видео в браузере
  });
  it("нет кадра «до» и захвата из видео — честная плашка и переход к моменту, промежуток не рисуется", () => {
    const v = { ...base, evidence_frames: [{ timestamp_sec: 21.5, image_base64: JPEG }] };
    const html = renderToString(<BeforeAfter v={v} peakT={21.5} boxes={[]} highlight onSeek={() => undefined} onOpen={() => undefined} />);
    const t = text(html);
    expect(t).toContain("кадра «до» нет");
    expect(t).toContain("к 00:18.0");
    expect(t).not.toMatch(/\+\d+,\d с/);
  });
  it("кадр без изображения (missing) не становится ни «до», ни «после»", () => {
    const v = { ...base, evidence_frames: [{ timestamp_sec: 17, image_base64: "", missing: true }, { timestamp_sec: 22, image_base64: "", missing: true }] };
    const html = renderToString(<BeforeAfter v={v} peakT={21.5} boxes={[]} highlight onSeek={() => undefined} onOpen={() => undefined} />);
    expect(text(html)).toContain("кадра «после» нет");
    expect(html).not.toContain("<img");
  });
});

describe("EvidenceImage", () => {
  it("рамка с тёмной подложкой и подписью — цвет и подпись из данных сервера", () => {
    const html = renderToString(
      <EvidenceImage src="data:image/jpeg;base64,AAAA" alt="кадр" boxes={[{ label: "сейф открыт", color: "#e5484d", bbox: [0.1, 0.2, 0.5, 0.6] }]} />,
    );
    expect(html.match(/<rect/g)).toHaveLength(2);
    expect(html).toContain('stroke="#e5484d"');
    expect(text(html)).toContain("сейф открыт");
  });
  it("без рамок — только изображение", () => {
    const html = renderToString(<EvidenceImage src="data:image/jpeg;base64,AAAA" alt="кадр" boxes={[]} />);
    expect(html).not.toContain("<svg");
  });
});

describe("JobScreen до первого ответа сервера", () => {
  it("шапка как у экрана обработки, номер задачи; без выдуманных процентов и сроков", () => {
    const html = wrap(<JobScreen jobId="syn-1" />);
    const t = text(html);
    expect(t).toContain("Получаем статус задачи…");
    expect(t).toContain("задача syn-1");
    expect(html).toContain('role="status"');
    expect(t).not.toMatch(/\d+ ?%/);
    expect(t).not.toMatch(/осталось|ETA/i);
  });
});

describe("ReplayScreen", () => {
  it("всегда помечен как повторный просмотр; JSON обязателен, видео по желанию", () => {
    const html = wrap(<ReplayScreen />);
    const t = text(html);
    expect(t).toContain("Повторный просмотр сохранённого отчёта.");
    expect(t).toContain("Это не новый анализ");
    expect(html).toContain('aria-label="Отчёт JSON"');
    expect(html).toContain('aria-label="Видео для отчёта"');
    expect(t).toContain("обязательно");
    expect(t).toContain("по желанию");
    expect(t).not.toContain("Обнаружены признаки возможных нарушений");
  });
});
