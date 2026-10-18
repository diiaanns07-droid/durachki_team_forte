import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, CheckCircle2, Database, Info, KeyRound, Loader2, LogOut, Server, ShieldAlert, Trash2, X } from "lucide-react";
import { clearHistory, settings } from "../api/client";
import { useApp } from "../context";
import { cx } from "./ui";
import s from "./SecondaryScreens.module.css";

// U10: окно настроек — свой доступный диалог (Tab остаётся внутри, Escape закрывает, фокус возвращается туда,
// откуда окно открыли). Поля — только существующие: адрес API, API-ключ, история, выход из сессии.

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Пусто — тот же адрес, что у интерфейса; "/путь" — тот же сервер; иначе нужен http(s)://. null — ошибок нет. */
export function validateApiBase(value: string): string | null {
  const v = value.trim();
  if (!v) return null;
  if (/\s/.test(v)) return "Адрес не должен содержать пробелов";
  if (v.startsWith("/")) return null;
  if (!/^https?:\/\//i.test(v)) return "Укажите адрес полностью, начиная с http:// или https://";
  try {
    new URL(v);
  } catch {
    return "Адрес не распознан — проверьте написание";
  }
  return null;
}

type KeyMode = "keep" | "replace" | "remove";

export default function SettingsDialog({ open, onClose, onSaved }: { open: boolean; onClose: () => void; onSaved: () => void }) {
  if (!open || typeof document === "undefined") return null;
  // Монтируется заново при каждом открытии: поля всегда показывают текущие сохранённые значения.
  return createPortal(<SettingsPanel onClose={onClose} onSaved={onSaved} />, document.body);
}

/** Содержимое окна (экспорт — для тестов; в приложении используется через SettingsDialog). */
export function SettingsPanel({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const { session, demo, serverOk, toast, logout } = useApp();
  const ids = {
    title: useId(),
    desc: useId(),
    form: useId(),
    base: useId(),
    baseHint: useId(),
    baseErr: useId(),
    key: useId(),
    keyHint: useId(),
    keyErr: useId(),
  };
  const initialBase = useMemo(() => settings.apiBase, []);
  // Сам ключ в окно не возвращаем: только факт, что он задан в памяти этой вкладки.
  const hadKey = useMemo(() => settings.apiKey !== "", []);
  const [apiBase, setApiBase] = useState(initialBase);
  const [keyMode, setKeyMode] = useState<KeyMode>(hadKey ? "keep" : "replace");
  const [newKey, setNewKey] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [historyCleared, setHistoryCleared] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLInputElement>(null);
  const keyRef = useRef<HTMLInputElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  const baseError = validateApiBase(apiBase);
  const keyError = hadKey && keyMode === "replace" && !newKey.trim() ? "Введите новый ключ или нажмите «Не менять»" : null;
  const baseChanged = apiBase.trim().replace(/\/+$/, "") !== initialBase;
  const keyChanged = keyMode === "remove" || (keyMode === "replace" && newKey.trim() !== "");
  const dirty = baseChanged || keyChanged;
  const showBaseError = !!baseError && (submitted || baseChanged);
  const showKeyError = !!keyError && submitted;

  // Фокус: запоминаем, откуда открыли; ставим на первое поле; при закрытии возвращаем. Прокрутку страницы блокируем.
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    // preventScroll: первое поле получает фокус, но окно остаётся пролистанным к началу (статус виден)
    const raf = requestAnimationFrame(() => (baseRef.current ?? panelRef.current)?.focus({ preventScroll: true }));
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeRef.current();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      if (opener && opener.isConnected) opener.focus();
    };
  }, []);

  const trapTab = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab" || !panelRef.current) return;
    const nodes = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
    if (!nodes.length) {
      e.preventDefault();
      return;
    }
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === panelRef.current)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const submit = () => {
    setSubmitted(true);
    if (baseError) {
      baseRef.current?.focus();
      return;
    }
    if (keyError) {
      keyRef.current?.focus();
      return;
    }
    if (baseChanged) settings.apiBase = apiBase;
    if (keyMode === "remove") settings.apiKey = "";
    else if (keyMode === "replace" && newKey.trim()) settings.apiKey = newKey;
    onSaved();
  };

  const doLogout = async () => {
    setLoggingOut(true);
    try {
      await logout();
      toast("Вы вышли из сессии");
      onClose();
    } catch (err) {
      setLoggingOut(false);
      toast(`Не удалось выйти: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  };

  const connection = connectionStatus({ demo, serverOk, authenticated: !!session?.authenticated, uiSession: !!session?.token, hadKey });

  return (
    <div className={cx("no-print", s.backdrop)} onMouseDown={onClose}>
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={ids.title}
        aria-describedby={ids.desc}
        tabIndex={-1}
        className={s.panel}
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={trapTab}
      >
        <div className={s.head}>
          <div className={cx(s.headInner, "flex items-start justify-between gap-4")}>
            <div className="min-w-0">
              <h2 id={ids.title} className="text-xl font-extrabold text-slate-900 dark:text-white">
                Настройки
              </h2>
              <p id={ids.desc} className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Подключение к серверу анализа и данные этого браузера.
              </p>
            </div>
            <button
              type="button"
              className="-mr-1 -mt-1 inline-flex h-10 w-10 flex-none items-center justify-center rounded-xl text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-white/10 dark:hover:text-white"
              onClick={onClose}
              aria-label="Закрыть настройки без сохранения"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        <form
          id={ids.form}
          className={s.body}
          noValidate
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <fieldset className={s.group}>
            <legend className={s.legend}>
              <span className={s.legendIcon} aria-hidden>
                <Server className="h-4 w-4" />
              </span>
              Подключение к серверу
            </legend>
            <p className={s.groupLead}>Интерфейс обращается к серверу анализа в локальной сети.</p>
            <div className={cx(s.status, connection.tone)} role="status">
              {connection.icon}
              <span>{connection.text}</span>
            </div>
            {!demo && session?.authenticated && (
              <div className="mt-3">
                <button type="button" className={cx("btn-ghost", s.smallBtn)} onClick={doLogout} disabled={loggingOut}>
                  {loggingOut ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <LogOut className="h-4 w-4" aria-hidden />}
                  {loggingOut ? "Выходим…" : "Выйти из сессии"}
                </button>
              </div>
            )}
            <div className={cx(s.field, "mt-4")}>
              <label htmlFor={ids.base} className={s.fieldLabel}>
                Адрес сервера (API)
              </label>
              <input
                id={ids.base}
                ref={baseRef}
                className={cx("input font-mono", showBaseError && s.inputInvalid)}
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                placeholder="тот же адрес, что у интерфейса"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
                inputMode="url"
                aria-invalid={showBaseError || undefined}
                aria-describedby={cx(ids.baseHint, showBaseError && ids.baseErr)}
              />
              <span id={ids.baseHint} className={s.hint}>
                Оставьте пустым, если интерфейс и сервер работают по одному адресу. Значение сохраняется в этом браузере.
              </span>
              {showBaseError && (
                <span id={ids.baseErr} className={s.error}>
                  <AlertTriangle className="mt-px h-3.5 w-3.5 flex-none" aria-hidden />
                  {baseError}
                </span>
              )}
            </div>
          </fieldset>

          <fieldset className={s.group}>
            <legend className={s.legend}>
              <span className={s.legendIcon} aria-hidden>
                <KeyRound className="h-4 w-4" />
              </span>
              API-ключ — запасной вход
            </legend>
            <p className={s.groupLead}>
              Нужен, только если вход в интерфейс по ключу доступа недоступен. Передаётся в заголовке{" "}
              <code className="whitespace-nowrap font-mono text-[12px]">X-API-Key</code>.
            </p>
            {keyMode !== "replace" ? (
              <div className={s.keyRow}>
                <div className={cx(s.status, keyMode === "remove" ? s.toneWarn : s.toneNeutral)} role="status">
                  {keyMode === "remove" ? <Trash2 className="h-4 w-4" aria-hidden /> : <KeyRound className="h-4 w-4" aria-hidden />}
                  <span>{keyMode === "remove" ? "Ключ будет удалён из этой вкладки после сохранения." : "Ключ задан в этой вкладке. Значение не показывается."}</span>
                </div>
                {keyMode === "keep" ? (
                  <span className="flex gap-2">
                    <button
                      type="button"
                      className={cx("btn-ghost", s.smallBtn)}
                      onClick={() => {
                        setKeyMode("replace");
                        requestAnimationFrame(() => keyRef.current?.focus());
                      }}
                    >
                      Заменить
                    </button>
                    <button type="button" className={cx("btn-ghost", s.smallBtn)} onClick={() => setKeyMode("remove")}>
                      Удалить
                    </button>
                  </span>
                ) : (
                  <button type="button" className={cx("btn-ghost", s.smallBtn)} onClick={() => setKeyMode("keep")}>
                    Не удалять
                  </button>
                )}
              </div>
            ) : (
              <div className={s.field}>
                <label htmlFor={ids.key} className={s.fieldLabel}>
                  {hadKey ? "Новый API-ключ" : "API-ключ"}
                </label>
                <div className="flex gap-2">
                  <input
                    id={ids.key}
                    ref={keyRef}
                    className={cx("input font-mono", showKeyError && s.inputInvalid)}
                    type="password"
                    autoComplete="off"
                    spellCheck={false}
                    autoCapitalize="off"
                    autoCorrect="off"
                    value={newKey}
                    onChange={(e) => setNewKey(e.target.value)}
                    placeholder={hadKey ? "введите новый ключ" : "не задан"}
                    aria-invalid={showKeyError || undefined}
                    aria-describedby={cx(ids.keyHint, showKeyError && ids.keyErr)}
                  />
                  {hadKey && (
                    <button
                      type="button"
                      className={cx("btn-ghost flex-none", s.smallBtn)}
                      onClick={() => {
                        setKeyMode("keep");
                        setNewKey("");
                      }}
                    >
                      Не менять
                    </button>
                  )}
                </div>
                <span id={ids.keyHint} className={s.hint}>
                  Хранится только в памяти этой вкладки: не записывается на диск и не показывается после ввода. После перезагрузки
                  страницы ключ нужно ввести снова.
                </span>
                {showKeyError && (
                  <span id={ids.keyErr} className={s.error}>
                    <AlertTriangle className="mt-px h-3.5 w-3.5 flex-none" aria-hidden />
                    {keyError}
                  </span>
                )}
              </div>
            )}
          </fieldset>

          <fieldset className={s.group}>
            <legend className={s.legend}>
              <span className={s.legendIcon} aria-hidden>
                <Database className="h-4 w-4" />
              </span>
              Данные этого браузера
            </legend>
            <p className={s.groupLead}>Список недавних проверок хранится только здесь. Очистка не удаляет задачи и отчёты на сервере.</p>
            {!confirmClear ? (
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className={cx("btn-ghost", s.smallBtn)}
                  onClick={() => {
                    setConfirmClear(true);
                    setHistoryCleared(false);
                  }}
                >
                  <Trash2 className="h-4 w-4" aria-hidden /> Очистить историю проверок
                </button>
                {historyCleared && (
                  <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-700 dark:text-emerald-300" role="status">
                    <CheckCircle2 className="h-4 w-4" aria-hidden /> История очищена
                  </span>
                )}
              </div>
            ) : (
              <div className={s.confirm} role="group" aria-label="Подтверждение очистки истории">
                <p className="font-semibold">Удалить список недавних проверок из этого браузера?</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    className={cx("btn bg-rose-600 text-white hover:bg-rose-700", s.smallBtn)}
                    onClick={() => {
                      clearHistory();
                      setConfirmClear(false);
                      setHistoryCleared(true);
                      toast("История проверок очищена", "success");
                    }}
                  >
                    Очистить
                  </button>
                  <button type="button" className={cx("btn-ghost", s.smallBtn)} onClick={() => setConfirmClear(false)}>
                    Отмена
                  </button>
                </div>
              </div>
            )}
          </fieldset>
        </form>

        <div className={s.foot}>
          <span className={cx(s.saveNote, dirty && s.saveNoteDirty)} aria-live="polite">
            {dirty ? "Есть несохранённые изменения" : "Изменений нет"}
          </span>
          <div className={s.footActions}>
            <button type="button" className="btn-ghost" onClick={onClose}>
              Отмена
            </button>
            <button type="submit" form={ids.form} className="btn-brand" disabled={!dirty} title={dirty ? undefined : "Изменений нет"}>
              Сохранить
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function connectionStatus(p: { demo: boolean; serverOk: boolean | null; authenticated: boolean; uiSession: boolean; hadKey: boolean }): {
  tone: string;
  icon: ReactNode;
  text: string;
} {
  if (p.demo)
    return {
      tone: s.toneWarn,
      icon: <Info className="h-4 w-4" aria-hidden />,
      text: "Синтетическое демо: запросы к серверу не выполняются, настройки подключения сейчас не используются.",
    };
  if (p.serverOk === false)
    return {
      tone: s.toneError,
      icon: <ShieldAlert className="h-4 w-4" aria-hidden />,
      text: "Сервер анализа не отвечает. Проверьте адрес ниже и доступность сервера в сети.",
    };
  if (p.serverOk === null)
    return { tone: s.toneNeutral, icon: <Loader2 className="h-4 w-4 animate-spin" aria-hidden />, text: "Проверяем связь с сервером…" };
  if (p.uiSession)
    return { tone: s.toneOk, icon: <CheckCircle2 className="h-4 w-4" aria-hidden />, text: "Сервер доступен, вход выполнен: сессия интерфейса активна." };
  if (p.authenticated && p.hadKey)
    return { tone: s.toneOk, icon: <CheckCircle2 className="h-4 w-4" aria-hidden />, text: "Сервер доступен, доступ по API-ключу из этой вкладки." };
  if (p.authenticated)
    return { tone: s.toneOk, icon: <CheckCircle2 className="h-4 w-4" aria-hidden />, text: "Сервер доступен, доступ разрешён." };
  return {
    tone: s.toneWarn,
    icon: <AlertTriangle className="h-4 w-4" aria-hidden />,
    // serverOk = «ответил любым статусом» — это ещё не подтверждение сервера анализа, поэтому без «сервер доступен»
    text: "Вход не выполнен. Войдите с ключом доступа на странице проверки или задайте API-ключ ниже.",
  };
}
