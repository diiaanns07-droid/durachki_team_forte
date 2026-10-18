// U06: плеер и кадры-доказательства — то, что проверяется без DOM (renderToString). Синтетические данные.
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { EvidenceFrame, Violation } from "../src/api/types";
import { BeforeAfter, EvidenceSequence } from "../src/components/Evidence";
import VideoPlayer from "../src/components/VideoPlayer";

const JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD";
const noop = () => undefined;
const count = (html: string, re: RegExp) => (html.match(re) ?? []).length;

describe("VideoPlayer: недоступное видео не изображает успех", () => {
  it("нет источника и не демо — причина на сцене, воспроизведение выключено, <video> нет", () => {
    const html = renderToString(<VideoPlayer source={null} duration={74} timeline={[]} violations={[]} />);
    expect(html).toContain("Исходное видео недоступно для этой задачи");
    expect(html).toContain("Перемотка и кадры из видео недоступны");
    expect(html).not.toContain("<video");
    expect(html).toMatch(/<button[^>]*aria-label="Воспроизвести"[^>]*disabled=""/);
    // время — отмеченный момент на шкале, а не позиция воспроизведения: так и подписано
    expect(html).toContain("видео недоступно — время по шкале");
  });
  it("демо — синтетическая сцена подписана", () => {
    const html = renderToString(<VideoPlayer source={null} synthetic duration={74} timeline={[]} violations={[]} />);
    expect(html).toContain("синтетическая сцена · демо");
    expect(html).toContain('alt="Синтетическая сцена (демо)"');
  });
  it("есть источник — <video> и индикатор загрузки", () => {
    const html = renderToString(<VideoPlayer source={{ url: "blob:synthetic" }} duration={0} timeline={[]} violations={[]} />);
    expect(html).toContain("<video");
    expect(html).toContain('aria-label="Видео загружается"');
    expect(html).toContain("Длительность записи неизвестна");
  });
});

describe("EvidenceSequence: последовательность с фактическими таймкодами", () => {
  const frames: EvidenceFrame[] = [
    { timestamp_sec: 5, image_base64: "", missing: true },
    { timestamp_sec: 6.03, image_base64: JPEG },
    { timestamp_sec: 7.5, image_base64: JPEG },
  ];
  it("кадр без изображения — плашка и переход, не <img>; номера и таймкоды по порядку", () => {
    const html = renderToString(<EvidenceSequence frames={frames} peakT={6} boxes={[]} highlight onOpen={noop} onSeek={noop} />);
    expect(html).toContain("кадр не передан");
    expect(html).toContain("сервер сообщил момент без изображения");
    expect(html).toContain("перейти к 00:05.0 в видео");
    expect(count(html, /<img /g)).toBe(2);
    expect(html).not.toMatch(/<img[^>]+src="data:image\/jpeg;base64,"/);
    const order = ["00:05.0", "00:06.0", "00:07.5"].map((tc) => html.indexOf(tc));
    expect(order.every((i, k) => i >= 0 && (k === 0 || i > order[k - 1]))).toBe(true);
  });
  it("порядок показа — по времени кадра, даже если сервер прислал сначала кадр пика", () => {
    const peakFirst: EvidenceFrame[] = [
      { timestamp_sec: 24.6, image_base64: JPEG },
      { timestamp_sec: 24.2, image_base64: JPEG },
      { timestamp_sec: 29.0, image_base64: JPEG },
    ];
    const html = renderToString(<EvidenceSequence frames={peakFirst} peakT={24.6} boxes={[]} highlight onOpen={noop} onSeek={noop} />);
    const at = (tc: string) => html.indexOf(`Открыть кадр ${tc}`);
    expect(at("1 · 00:24.2")).toBeGreaterThan(-1);
    expect(at("2 · 00:24.6 · момент эпизода")).toBeGreaterThan(at("1 · 00:24.2"));
    expect(at("3 · 00:29.0")).toBeGreaterThan(at("2 · 00:24.6 · момент эпизода"));
  });
  it("«момент эпизода» — только у кадра в пределах highlightMaxDtS от t_peak", () => {
    const html = renderToString(<EvidenceSequence frames={frames} peakT={6} boxes={[]} highlight onOpen={noop} onSeek={noop} />);
    expect(count(html, /момент эпизода<\/div>/g)).toBe(1);
    const far = renderToString(<EvidenceSequence frames={frames} peakT={6.5} boxes={[]} highlight onOpen={noop} onSeek={noop} />);
    expect(far).not.toContain("момент эпизода</div>");
  });
});

describe("BeforeAfter: точная причина вместо подстановки", () => {
  const v: Violation = {
    type: "B",
    type_id: "V02",
    timecode_sec: 30,
    t_start: 30,
    t_end: 40,
    description: "",
    evidence_frames: [{ timestamp_sec: 31, image_base64: JPEG }],
  };
  it("нет кадра «до» и нет видео — причина названа", () => {
    const html = renderToString(<BeforeAfter v={v} peakT={30} boxes={[]} highlight onSeek={noop} onOpen={noop} />);
    expect(html).toContain("кадра «до» нет");
    expect(html).toContain("сервер не передал кадр до начала эпизода.");
    expect(html).toContain("кадр сервера");
  });
});
