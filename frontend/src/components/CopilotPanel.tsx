import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, Bot, BotOff, Eraser, Maximize2, Minimize2, RotateCcw, Send, X } from "lucide-react";
import { ApiError, CopilotUnavailable } from "../api/client";
import type { JobResult } from "../api/types";
import { useApp } from "../context";
import { formatTimecode } from "../lib/format";
import { cx, Spinner } from "./ui";
import s from "./CopilotPanel.module.css";

/*
 * Copilot проверяющего (U08). Протокол не меняется: POST …/jobs/{id}/ask {question, history} через backend.ask.
 * Лимиты сервера A14 (copilot/router.py): вопрос ≤ 2000 символов, история ≤ 40 ходов по ≤ 8000 символов —
 * поэтому отправляем только последние MAX_HISTORY_TURNS ходов (сервис сам берёт ещё меньше).
 * Когда помощник выключен или недоступен — компактная карточка; ответов не имитируем, облако не подключаем.
 */
const MAX_QUESTION = 1000;
const MAX_HISTORY_TURNS = 16;
const MAX_TURN_CHARS = 8000;

type Turn = { role: "user" | "assistant"; text: string };
type Msg =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "assistant"; text: string; sources: string[] }
  | { id: number; role: "notice"; tone: "error" | "empty" | "stopped"; text: string; questionId: number; question: string };
type NoticeMsg = Extract<Msg, { role: "notice" }>;
type Availability = "checking" | "ready" | "unknown";
interface Quick {
  label: string; // коротко — помещается в узкую колонку
  ariaLabel: string; // полный смысл для скринридера и подсказки
  question: string;
}

/** История для сервера: только вопросы и ответы до указанного вопроса, без служебных уведомлений. */
function historyBefore(msgs: Msg[], beforeId?: number): Turn[] {
  const turns: Turn[] = [];
  for (const m of msgs) {
    if (beforeId !== undefined && m.id >= beforeId) break;
    if (m.role === "user" || m.role === "assistant") turns.push({ role: m.role, text: m.text.slice(0, MAX_TURN_CHARS) });
  }
  return turns.slice(-MAX_HISTORY_TURNS);
}

/** Быстрые вопросы — только о том, что реально есть в результате: эпизоды, кадры, указанное основание. */
function quickQuestions(result: JobResult | null): Quick[] {
  if (!result || result.status !== "done") return [];
  const viols = result.violations ?? [];
  const out: Quick[] = viols.slice(0, 2).map((v, i) => {
    const tc = formatTimecode(v.timecode_sec);
    return {
      label: `Эпизод ${i + 1} · ${tc}`,
      ariaLabel: `Объяснить эпизод ${i + 1}, ${tc}: ${v.type}`,
      question: `Объясни эпизод ${i + 1} («${v.type}», ${tc}): что видно на записи и почему это может быть нарушением?`,
    };
  });
  if (viols.some((v) => (v.evidence_frames?.length ?? 0) > 0))
    out.push({
      label: "Кадры-доказательства",
      ariaLabel: "Перечислить кадры-доказательства по эпизодам",
      question: "Перечисли кадры-доказательства по каждому эпизоду с таймкодами.",
    });
  if (viols.some((v) => (v.rules?.length ?? 0) > 0))
    out.push({
      label: "Указанное основание",
      ariaLabel: "Показать нормативное основание, указанное в отчёте",
      question: "Какое нормативное основание указано в отчёте для эпизодов? Назови документ и пункт, если они указаны.",
    });
  if (viols.length === 0 && result.verdict === "no_violation")
    out.push({
      label: "Почему без нарушений",
      ariaLabel: "Почему нарушений не найдено",
      question: "Почему в отчёте нет эпизодов с признаками нарушений и что система не проверяет?",
    });
  return out;
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Внутренние функции — для тестов (frontend/tests/copilotPanel.test.tsx). */
export const copilotInternals = { historyBefore, quickQuestions, MAX_HISTORY_TURNS, MAX_TURN_CHARS, MAX_QUESTION };

export default function CopilotPanel({ jobId }: { jobId: string }) {
  const { backend, copilotEnabled, disableCopilot, toast, demo, lastResult } = useApp();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [availability, setAvailability] = useState<Availability>("checking");
  const [disclaimer, setDisclaimer] = useState<string | undefined>();
  const [wide, setWide] = useState(false);
  const seq = useRef(0); // id сообщений (монотонно)
  const reqToken = useRef(0); // ответы на устаревшие запросы (смена анализа, «не ждать») отбрасываются
  const pendingQ = useRef<{ questionId: number; question: string } | null>(null);
  const msgsRef = useRef(msgs);
  msgsRef.current = msgs;
  const wideBtnRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();

  const result = lastResult && lastResult.job_id === jobId ? lastResult : null;
  const quick = useMemo(() => quickQuestions(result), [result]);

  // Сменился выбранный анализ: история, черновик и ожидание относились к другому отчёту.
  useEffect(() => {
    reqToken.current += 1;
    pendingQ.current = null;
    setMsgs([]);
    setText("");
    setBusy(false);
    setWide(false);
  }, [jobId]);

  // A14: GET …/copilot/status — модель недоступна → компактное состояние (не ждём 501 от /ask).
  useEffect(() => {
    if (!copilotEnabled) return;
    if (!backend.copilotStatus) {
      setAvailability("unknown");
      return;
    }
    let alive = true;
    setAvailability("checking");
    backend
      .copilotStatus()
      .then((st) => {
        if (!alive) return;
        if (!st) setAvailability("unknown");
        else if (!st.available) disableCopilot();
        else {
          setDisclaimer(st.disclaimer);
          setAvailability("ready");
        }
      })
      .catch(() => alive && setAvailability("unknown"));
    return () => {
      alive = false;
    };
  }, [backend, copilotEnabled, disableCopilot]);

  // Реальное время ожидания ответа (не прогноз).
  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    setElapsed(0);
    const t = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(t);
  }, [busy]);

  const ask = useCallback(
    async (raw: string, retry?: NoticeMsg) => {
      const question = raw.trim().slice(0, MAX_QUESTION);
      if (!question || busy) return;
      const token = ++reqToken.current;
      let questionId: number;
      let history: Turn[];
      if (retry) {
        questionId = retry.questionId;
        history = historyBefore(msgsRef.current, questionId);
        setMsgs((m) => m.filter((x) => x.id !== retry.id));
      } else {
        questionId = ++seq.current;
        history = historyBefore(msgsRef.current);
        setMsgs((m) => [...m, { id: questionId, role: "user", text: question }]);
        setText("");
      }
      pendingQ.current = { questionId, question };
      setBusy(true);
      try {
        const r = await backend.ask(jobId, question, history);
        if (token !== reqToken.current) return;
        const answer = typeof r.answer === "string" ? r.answer : "";
        setMsgs((m) => [
          ...m,
          answer.trim()
            ? { id: ++seq.current, role: "assistant", text: answer, sources: r.sources ?? [] }
            : {
                id: ++seq.current,
                role: "notice",
                tone: "empty",
                text: "Copilot вернул пустой ответ. Повторите вопрос или сформулируйте его иначе.",
                questionId,
                question,
              },
        ]);
      } catch (e) {
        if (token !== reqToken.current) return;
        if (e instanceof CopilotUnavailable) {
          disableCopilot();
          toast("Copilot отключён на этом сервере — отчёт доступен без него");
          return;
        }
        const message = e instanceof ApiError ? e.message : "Не удалось получить ответ. Проверьте соединение с сервером.";
        setMsgs((m) => [...m, { id: ++seq.current, role: "notice", tone: "error", text: message, questionId, question }]);
      } finally {
        if (token === reqToken.current) {
          pendingQ.current = null;
          setBusy(false);
        }
      }
    },
    [backend, busy, disableCopilot, jobId, toast],
  );

  /** «Не ждать»: ответ на этот запрос, если придёт, не показываем; вопрос можно повторить. */
  const stopWaiting = useCallback(() => {
    const p = pendingQ.current;
    reqToken.current += 1;
    pendingQ.current = null;
    setBusy(false);
    if (p)
      setMsgs((m) => [
        ...m,
        {
          id: ++seq.current,
          role: "notice",
          tone: "stopped",
          text: "Ожидание прервано. Если ответ придёт позже, он не будет показан.",
          questionId: p.questionId,
          question: p.question,
        },
      ]);
  }, []);

  const clear = useCallback(() => {
    reqToken.current += 1;
    pendingQ.current = null;
    setMsgs([]);
    setBusy(false);
  }, []);

  const closeWide = useCallback(() => setWide(false), []);

  if (!copilotEnabled) return <Unavailable />;

  const statusLine =
    availability === "checking"
      ? "локальная модель · проверяем доступность…"
      : availability === "ready"
        ? "локальная модель · доступна"
        : "локальная модель · статус не сообщается";

  const chat = (variant: "side" | "wide") => (
    <>
      {msgs.length === 0 ? (
        <Intro quick={quick} busy={busy} onPick={(q) => void ask(q)} wide={variant === "wide"} />
      ) : (
        <ChatLog msgs={msgs} busy={busy} elapsed={elapsed} wide={variant === "wide"} onRetry={(n) => void ask(n.question, n)} onStop={stopWaiting} />
      )}
      <Composer
        value={text}
        onChange={setText}
        onSubmit={() => void ask(text)}
        busy={busy}
        rows={variant === "wide" ? 2 : 1}
        autoFocus={variant === "wide"}
      />
    </>
  );

  const header = (variant: "side" | "wide", headingId?: string) => (
    <div className={s.head}>
      <span className={s.mark} aria-hidden>
        <Bot className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <h2 id={headingId} className="font-display text-sm font-bold leading-tight text-slate-900 dark:text-white">
            Copilot проверяющего
          </h2>
          {demo && <span className={s.demoChip}>Демо: ответы заготовлены</span>}
        </div>
        <p className="mt-0.5 text-[11px] leading-snug text-slate-500 dark:text-slate-400">{statusLine}</p>
      </div>
      <div className={s.actions}>
        {msgs.length > 0 && (
          <button type="button" className={s.iconBtn} onClick={clear} disabled={busy} aria-label="Начать новый диалог" title="Новый диалог">
            <Eraser className="h-4 w-4" />
          </button>
        )}
        {variant === "side" ? (
          <button
            ref={wideBtnRef}
            type="button"
            className={s.iconBtn}
            onClick={() => setWide(true)}
            aria-label="Открыть диалог в широком режиме"
            aria-haspopup="dialog"
            title="Широкий режим"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        ) : (
          <>
            <button type="button" className={s.iconBtn} onClick={closeWide} aria-label="Свернуть в боковую панель" title="Свернуть">
              <Minimize2 className="h-4 w-4" />
            </button>
            <button type="button" className={s.iconBtn} onClick={closeWide} aria-label="Закрыть широкий режим" title="Закрыть (Esc)">
              <X className="h-4 w-4" />
            </button>
          </>
        )}
      </div>
    </div>
  );

  const banner = disclaimer ? (
    <p className={s.banner} role="note">
      {disclaimer}
    </p>
  ) : null;

  return (
    <section className={cx("card no-print", s.panel)} aria-labelledby={titleId}>
      {header("side", titleId)}
      {banner}
      {wide ? (
        <div className={s.wideNote}>
          <span>Диалог открыт в широком режиме.</span>
          <button type="button" className={s.linkBtn} onClick={closeWide}>
            Вернуть сюда
          </button>
        </div>
      ) : (
        chat("side")
      )}
      {wide &&
        createPortal(
          <ReadingSheet onClose={closeWide} returnFocus={wideBtnRef} label="Copilot проверяющего — широкий режим">
            {header("wide")}
            {banner}
            {chat("wide")}
          </ReadingSheet>,
          document.body,
        )}
    </section>
  );
}

function Unavailable() {
  return (
    <section className={cx("card no-print", s.off)} aria-label="Copilot недоступен">
      <span className={s.offMark} aria-hidden>
        <BotOff className="h-4 w-4" />
      </span>
      <div className="min-w-0">
        <h2 className="font-display text-sm font-bold text-slate-800 dark:text-slate-100">Copilot недоступен</h2>
        <p className="mt-0.5 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          Локальный помощник на этом сервере выключен или не отвечает. Отчёт, видео и кадры доступны без него.
        </p>
      </div>
    </section>
  );
}

function Intro({ quick, busy, onPick, wide }: { quick: Quick[]; busy: boolean; onPick: (q: string) => void; wide: boolean }) {
  return (
    <div className={cx(s.intro, wide && s.introWide)}>
      <p className="text-[13px] leading-relaxed text-slate-600 dark:text-slate-300">
        Отвечает по этому отчёту: эпизоды, факты, таймкоды кадров, указанные нормы. Видео и изображения кадров не просматривает.
      </p>
      {quick.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1.5">Спросить по отчёту</div>
          <div className={s.quick}>
            {quick.map((q) => (
              <button
                key={q.label}
                type="button"
                className={s.quickBtn}
                onClick={() => onPick(q.question)}
                disabled={busy}
                aria-label={q.ariaLabel}
                title={q.ariaLabel}
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ChatLog({
  msgs,
  busy,
  elapsed,
  wide,
  onRetry,
  onStop,
}: {
  msgs: Msg[];
  busy: boolean;
  elapsed: number;
  wide: boolean;
  onRetry: (n: NoticeMsg) => void;
  onStop: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true); // читатель у последнего сообщения
  const userScroll = useRef(false); // прокрутку начал человек (колесо, касание, клавиши), а не автопрокрутка
  const last = msgs[msgs.length - 1];

  // Прокрутка к новому сообщению, только если читатель внизу (не дёргаем при чтении истории) или спросил сам.
  // Длинный ответ показываем с начала, а не с конца.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !last) return;
    if (last.role === "user") stick.current = true;
    if (!stick.current) return;
    const behavior: ScrollBehavior = prefersReducedMotion() ? "auto" : "smooth";
    const node = el.querySelector<HTMLElement>(`[data-msg="${last.id}"]`);
    if (node && last.role !== "user" && !busy && node.offsetHeight > el.clientHeight - 32) {
      el.scrollTo({ top: Math.max(0, node.offsetTop - 12), behavior });
      return;
    }
    el.scrollTo({ top: el.scrollHeight, behavior });
  }, [msgs.length, busy, last]);

  return (
    <div
      ref={ref}
      className={cx(s.log, wide && s.logWide, "scrollbar-thin")}
      role="log"
      aria-live="polite"
      aria-relevant="additions"
      aria-busy={busy}
      aria-label="История вопросов и ответов"
      tabIndex={0}
      onWheel={() => (userScroll.current = true)}
      onTouchStart={() => (userScroll.current = true)}
      onKeyDown={() => (userScroll.current = true)}
      onPointerDown={() => (userScroll.current = true)}
      onScroll={(e) => {
        if (!userScroll.current) return; // события автопрокрутки не меняют решение
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      {msgs.map((m) => (
        <MessageView key={m.id} m={m} busy={busy} onRetry={onRetry} />
      ))}
      {busy && (
        <div className={s.pending}>
          <span className={s.dots} aria-hidden>
            <i />
            <i />
            <i />
          </span>
          <span>Copilot готовит ответ</span>
          {elapsed >= 2 && (
            <span className="tnum text-slate-400" aria-hidden>
              · {elapsed} с
            </span>
          )}
          <button type="button" className={s.linkBtn} onClick={onStop}>
            Не ждать
          </button>
        </div>
      )}
    </div>
  );
}

function MessageView({ m, busy, onRetry }: { m: Msg; busy: boolean; onRetry: (n: NoticeMsg) => void }) {
  if (m.role === "user")
    return (
      <div className={cx(s.userRow, s.msgIn)} data-msg={m.id}>
        <p className={s.userBubble}>
          <span className="sr-only">Вопрос: </span>
          {m.text}
        </p>
      </div>
    );
  if (m.role === "assistant")
    return (
      <article className={cx(s.answer, s.msgIn)} data-msg={m.id}>
        <div className={s.answerHead}>
          <Bot className="h-3.5 w-3.5" aria-hidden /> Copilot
        </div>
        <div className={s.answerText}>{m.text}</div>
        {m.sources.length > 0 && (
          <div className={s.sources}>
            <span className="text-[11px] text-slate-500 dark:text-slate-400">Источники ответа:</span>
            <ul className={s.sourceList}>
              {m.sources.map((src, i) => (
                <li key={i} className={s.source} title={src}>
                  {src}
                </li>
              ))}
            </ul>
          </div>
        )}
      </article>
    );
  return (
    <div className={cx(s.notice, m.tone === "error" ? s.noticeError : s.noticeWarn, s.msgIn)} data-msg={m.id}>
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <p className="min-w-0 flex-1">
        <span className="sr-only">{m.tone === "error" ? "Ошибка: " : "Внимание: "}</span>
        {m.text}
      </p>
      <button type="button" className={s.retryBtn} onClick={() => onRetry(m)} disabled={busy}>
        <RotateCcw className="h-3.5 w-3.5" aria-hidden /> Повторить
      </button>
    </div>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  busy,
  rows,
  autoFocus,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
  rows: number;
  autoFocus: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const id = useId();
  const len = value.length;

  // Автовысота поля: растёт до ~6 строк, дальше — прокрутка внутри поля.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  return (
    <form
      className={s.composer}
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor={id} className="sr-only">
        Вопрос Copilot по этому отчёту
      </label>
      <div className={s.inputRow}>
        <textarea
          id={id}
          ref={ref}
          className={cx("input", s.textarea)}
          rows={rows}
          value={value}
          maxLength={MAX_QUESTION}
          placeholder="Вопрос по этому отчёту…"
          aria-describedby={`${id}-hint`}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onSubmit();
            }
          }}
        />
        <button type="submit" className={cx("btn-primary", s.send)} disabled={busy || !value.trim()} aria-label="Отправить вопрос">
          {busy ? <Spinner className="h-4 w-4" /> : <Send className="h-4 w-4" />}
        </button>
      </div>
      <div id={`${id}-hint`} className={s.hint}>
        <span>{busy ? "Дождитесь ответа — следующий вопрос можно уже набирать" : "Enter — отправить · Shift+Enter — новая строка"}</span>
        {len > MAX_QUESTION * 0.8 && (
          <span className={cx("tnum", len >= MAX_QUESTION && s.counterMax)}>
            {len}/{MAX_QUESTION}
          </span>
        )}
      </div>
    </form>
  );
}

/** Широкий режим чтения: боковой лист поверх страницы (портал — предки анимированы через transform). */
function ReadingSheet({
  onClose,
  returnFocus,
  label,
  children,
}: {
  onClose: () => void;
  returnFocus: RefObject<HTMLButtonElement>;
  label: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const body = document.body;
    const overflow = body.style.overflow;
    body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
      } else if (e.key === "Tab" && ref.current) {
        const items = Array.from(
          ref.current.querySelectorAll<HTMLElement>('button:not([disabled]),textarea:not([disabled]),[href],[tabindex]:not([tabindex="-1"])'),
        ).filter((el) => el.offsetParent !== null);
        if (items.length === 0) return;
        const first = items[0];
        const lastEl = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          lastEl.focus();
        } else if (!e.shiftKey && document.activeElement === lastEl) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    const target = returnFocus.current;
    return () => {
      document.removeEventListener("keydown", onKey);
      body.style.overflow = overflow;
      target?.focus();
    };
  }, [returnFocus]);

  return (
    <div className={cx("no-print", s.sheetRoot)}>
      <div className={s.backdrop} onMouseDown={() => onCloseRef.current()} aria-hidden />
      <div ref={ref} role="dialog" aria-modal="true" aria-label={label} className={cx("card", s.sheet)}>
        {children}
      </div>
    </div>
  );
}
