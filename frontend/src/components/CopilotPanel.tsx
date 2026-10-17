import { useEffect, useRef, useState } from "react";
import { Bot, Send } from "lucide-react";
import { ApiError, CopilotUnavailable } from "../api/client";
import { useApp } from "../context";
import { cx, Spinner } from "./ui";

interface Msg {
  role: "user" | "assistant";
  text: string;
  sources?: string[];
}

const SUGGESTIONS = ["Почему это нарушение?", "Какая норма применяется?", "Что проверить вручную?"];

/** Чат с локальной LLM (POST …/jobs/{id}/ask). Необязательный: при 501 панель скрывается. */
export default function CopilotPanel({ jobId }: { jobId: string }) {
  const { backend, copilotEnabled, disableCopilot, toast } = useApp();
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [disclaimer, setDisclaimer] = useState<string | undefined>();

  // A14: GET …/copilot/status — если модель недоступна, панель не показываем (не ждём 501).
  useEffect(() => {
    if (!copilotEnabled || !backend.copilotStatus) return;
    let alive = true;
    void backend.copilotStatus().then((st) => {
      if (!alive || !st) return;
      if (!st.available) disableCopilot();
      else setDisclaimer(st.disclaimer);
    });
    return () => {
      alive = false;
    };
  }, [backend, copilotEnabled, disableCopilot]);

  if (!copilotEnabled) return null;

  const send = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setError(null);
    const history = msgs.map((m) => ({ role: m.role, text: m.text }));
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setText("");
    setBusy(true);
    try {
      const r = await backend.ask(jobId, question, history);
      setMsgs((m) => [...m, { role: "assistant", text: r.answer, sources: r.sources }]);
    } catch (e) {
      if (e instanceof CopilotUnavailable) {
        disableCopilot();
        toast("Copilot отключён на этом сервере — отчёт доступен без него");
        return;
      }
      setError(e instanceof ApiError ? e.message : "Не удалось получить ответ.");
    } finally {
      setBusy(false);
      setTimeout(() => listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" }), 30);
    }
  };

  return (
    <section className="card no-print flex flex-col overflow-hidden">
      <div className="flex items-center gap-3 border-b border-slate-100 px-4 py-3 dark:border-white/5">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-ink-700 text-white">
          <Bot className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-bold">Copilot проверяющего</div>
          <div className="text-[11px] text-slate-500">локальная LLM · отвечает по этому отчёту</div>
          {disclaimer && <div className="mt-0.5 text-[11px] text-amber-700 dark:text-amber-300">{disclaimer}</div>}
        </div>
      </div>
      <div ref={listRef} className="scrollbar-thin max-h-80 min-h-[120px] space-y-3 overflow-y-auto p-4">
        {msgs.length === 0 && <p className="text-sm text-slate-500">Спросите о найденных эпизодах, нормах или о том, что стоит проверить вручную.</p>}
        {msgs.map((m, i) => (
          <div key={i} className={cx("flex", m.role === "user" ? "justify-end" : "justify-start")}>
            <div
              className={cx(
                "max-w-[88%] whitespace-pre-wrap rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed",
                m.role === "user" ? "rounded-br-md bg-ink-900 text-white dark:bg-white dark:text-ink-900" : "rounded-bl-md bg-slate-100 text-slate-800 dark:bg-white/[.06] dark:text-slate-100",
              )}
            >
              {m.text}
              {m.sources && m.sources.length > 0 && <div className="mt-1.5 text-[11px] opacity-60">источники: {m.sources.join(", ")}</div>}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Spinner className="h-3.5 w-3.5" /> Copilot думает…
          </div>
        )}
        {error && <div className="text-xs text-rose-600 dark:text-rose-400">{error}</div>}
      </div>
      {msgs.length === 0 && (
        <div className="flex flex-wrap gap-1.5 px-4 pb-3">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="chip border-slate-200 text-slate-600 hover:border-brand-300 hover:text-brand-700 dark:border-white/10 dark:text-slate-300" onClick={() => void send(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
      <form
        className="flex gap-2 border-t border-slate-100 p-3 dark:border-white/5"
        onSubmit={(e) => {
          e.preventDefault();
          void send(text);
        }}
      >
        <input className="input py-2" value={text} onChange={(e) => setText(e.target.value)} placeholder="Ваш вопрос…" maxLength={1000} aria-label="Вопрос Copilot" />
        <button className="btn-primary px-3" type="submit" disabled={busy || !text.trim()} aria-label="Отправить">
          <Send className="h-4 w-4" />
        </button>
      </form>
    </section>
  );
}
