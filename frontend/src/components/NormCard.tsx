import { useState } from "react";
import { BadgeCheck, BookOpen, ChevronDown, ExternalLink, ShieldAlert } from "lucide-react";
import type { RuleRef } from "../api/types";
import { cx } from "./ui";

/** Карточка нормы РК (данные только из norms/catalog.yaml через сервер). */
export default function NormCard({ rule }: { rule: RuleRef }) {
  const [open, setOpen] = useState(false);
  const hasQuote = !!rule.quote && rule.quote.trim().length > 0;
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4 dark:border-white/10 dark:bg-white/[.03]">
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-900 text-white dark:bg-white dark:text-ink-900">
          <BookOpen className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold leading-snug text-slate-900 dark:text-white">{rule.document}</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            {rule.clause && rule.clause.trim() ? (
              <span className="font-mono font-semibold text-slate-700 dark:text-slate-200">{rule.clause}</span>
            ) : (
              <span className="chip border-slate-200 text-slate-500 dark:border-white/10" title="Конкретный пункт в каталоге норм не установлен — ничего не придумываем">
                без пункта
              </span>
            )}
            {rule.verified === true ? (
              <span className="chip border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-300" title="Цитата сверена с текстом первоисточника">
                <BadgeCheck className="h-3.5 w-3.5" /> цитата сверена
              </span>
            ) : (
              <span className="chip border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-400/20 dark:bg-amber-400/10 dark:text-amber-300" title="Цитату нужно сверить с первоисточником">
                <ShieldAlert className="h-3.5 w-3.5" /> цитата не сверена
              </span>
            )}
          </div>
          {!rule.verified && rule.status_ru && <div className="mt-1 text-[11px] text-amber-700 dark:text-amber-300">{rule.status_ru}</div>}
        </div>
      </div>
      {hasQuote && (
        <>
          <button className="no-print mt-3 inline-flex items-center gap-1 text-xs font-semibold text-brand-700 hover:underline dark:text-brand-300" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            <ChevronDown className={cx("h-3.5 w-3.5 transition", open && "rotate-180")} />
            {open ? "Скрыть цитату" : "Показать цитату"}
          </button>
          <blockquote className={cx("mt-2 border-l-2 border-brand-500 pl-3 text-sm italic leading-relaxed text-slate-700 dark:text-slate-300", open ? "block" : "hidden print-expand")}>
            «{rule.quote}»
          </blockquote>
        </>
      )}
      {rule.url && (
        <a
          href={rule.url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-ink-700 hover:underline dark:text-slate-200"
        >
          <ExternalLink className="h-3.5 w-3.5" />
          {(() => {
            try {
              return new URL(rule.url).hostname;
            } catch {
              return "Открыть документ";
            }
          })()}
        </a>
      )}
    </div>
  );
}
