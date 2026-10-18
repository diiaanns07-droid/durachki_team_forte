import { useId, useState } from "react";
import { BadgeCheck, ChevronDown, ExternalLink, Scale, ShieldAlert } from "lucide-react";
import type { RuleRef } from "../api/types";
import { cx } from "./ui";
import s from "./ViolationCard.module.css";

/** Длинный текст (название документа, цитата) сворачивается до нескольких строк. */
const LONG_TITLE = 140;

/** Ссылку показываем только для http(s): прочие схемы (javascript:, data:) не рендерим как ссылку. */
export function safeHttpUrl(url: string | undefined): URL | null {
  if (!url) return null;
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u : null;
  } catch {
    return null;
  }
}

/**
 * Нормативное основание (данные только из norms/catalog.yaml через сервер; ничего не дописываем).
 * Первый уровень: документ, пункт («без пункта», если не установлен), статус сверки, понятная ссылка.
 * Длинное название и цитата раскрываются по кнопке; при печати раскрыты.
 */
export default function NormCard({ rule }: { rule: RuleRef }) {
  const [quoteOpen, setQuoteOpen] = useState(false);
  const [titleOpen, setTitleOpen] = useState(false);
  const quoteId = useId();
  const titleId = useId();
  const verified = rule.verified === true;
  const clause = rule.clause && rule.clause.trim() ? rule.clause.trim() : null;
  const quote = rule.quote && rule.quote.trim() ? rule.quote.trim() : null;
  const document = rule.document && rule.document.trim() ? rule.document.trim() : null;
  const longTitle = !!document && document.length > LONG_TITLE;
  const link = safeHttpUrl(rule.url);

  return (
    <div className={s.norm} data-verified={verified ? "true" : "false"}>
      <div className="flex items-start gap-2.5">
        <Scale className="mt-0.5 h-4 w-4 shrink-0 text-ink-500 dark:text-slate-400" aria-hidden />
        <div className="min-w-0 flex-1">
          {document ? (
            <p
              id={titleId}
              className={cx(s.breakWords, "text-sm font-semibold leading-snug text-slate-900 dark:text-white", longTitle && !titleOpen && s.clamp2)}
            >
              {document}
            </p>
          ) : (
            <p className="text-sm font-semibold text-slate-500">Документ не указан сервером</p>
          )}
          {longTitle && (
            <button
              type="button"
              className={cx(s.linkButton, "no-print mt-0.5")}
              aria-expanded={titleOpen}
              aria-controls={titleId}
              onClick={() => setTitleOpen((o) => !o)}
            >
              {titleOpen ? "Свернуть название" : "Полное название"}
            </button>
          )}

          <dl className={cx(s.normMeta, "mt-2 text-[13px]")}>
            <div>
              <dt>Пункт</dt>
              <dd>
                {clause ? (
                  <span className="font-mono font-semibold text-slate-800 dark:text-slate-100">{clause}</span>
                ) : (
                  <span className="text-slate-600 dark:text-slate-300" title="Конкретный пункт в каталоге норм не установлен — ничего не придумываем">
                    без пункта
                  </span>
                )}
              </dd>
            </div>
            <div>
              <dt>Сверка</dt>
              <dd>
                {verified ? (
                  <span className="inline-flex items-center gap-1 font-medium text-emerald-700 dark:text-emerald-300" title="Цитата сверена с текстом первоисточника">
                    <BadgeCheck className="h-3.5 w-3.5" aria-hidden /> цитата сверена
                  </span>
                ) : (
                  <span className={s.unverified} title="Цитату нужно сверить с первоисточником">
                    <ShieldAlert className="h-3.5 w-3.5" aria-hidden /> цитата не сверена
                  </span>
                )}
                {!verified && rule.status_ru && <span className="mt-1 block text-xs text-amber-800 dark:text-amber-200">{rule.status_ru}</span>}
              </dd>
            </div>
          </dl>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
            {link ? (
              <a href={link.href} target="_blank" rel="noopener noreferrer" className={s.docLink}>
                <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden />
                <span>
                  Открыть документ на <span className="whitespace-nowrap">{link.hostname}</span>
                </span>
                <span className="sr-only"> (откроется в новой вкладке)</span>
              </a>
            ) : (
              <span className="text-xs text-slate-500 dark:text-slate-400">Ссылка на документ не передана</span>
            )}
            {quote && (
              <button
                type="button"
                className={cx(s.linkButton, "no-print inline-flex items-center gap-1")}
                aria-expanded={quoteOpen}
                aria-controls={quoteId}
                onClick={() => setQuoteOpen((o) => !o)}
              >
                <ChevronDown className={cx("h-3.5 w-3.5", s.chevron, quoteOpen && s.chevronOpen)} aria-hidden />
                {quoteOpen ? "Скрыть цитату" : "Показать цитату"}
              </button>
            )}
          </div>

          {quote && (
            <blockquote id={quoteId} className={cx(s.quote, s.breakWords, quoteOpen ? "block animate-fade-in" : "hidden print-expand")}>
              «{quote}»
            </blockquote>
          )}
        </div>
      </div>
    </div>
  );
}
