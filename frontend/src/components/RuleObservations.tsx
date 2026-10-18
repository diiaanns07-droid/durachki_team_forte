// Объяснение по наблюдениям правил (ml.seqrules): статусы правил (что проверено, что неприменимо) и состав
// выбранного эпизода — сколько кадров наблюдалось, сколько неизвестно. Плюс переключатели слоёв оверлея.
import { Layers } from "lucide-react";
import type { ObsEpisode, Observations } from "../api/observations";
import { RULE_RU, trackColor } from "../api/observations";
import type { Violation } from "../api/types";
import { formatTimecode } from "../lib/format";
import type { OverlayLayers } from "./ObservationOverlay";
import { cx } from "./ui";

const STATUS_RU: Record<string, string> = {
  ok: "проверено",
  disabled: "выключено",
  unavailable: "неприменимо",
  error: "ошибка правила",
  no_persons: "людей не найдено",
};
const ROLE_RU: Record<string, string> = { cashier: "кассир", collector: "инкассатор", visitor: "посетитель", unknown: "роль не определена" };

export function OverlayToggles({ layers, onChange }: { layers: OverlayLayers; onChange: (l: OverlayLayers) => void }) {
  const items: [keyof OverlayLayers, string][] = [
    ["skeletons", "Скелеты"],
    ["objects", "Предметы"],
    ["zones", "Зоны"],
    ["all", "Все эпизоды"],
  ];
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs" role="group" aria-label="Слои наблюдений на кадре">
      <span className="flex items-center gap-1 font-semibold text-slate-500">
        <Layers className="h-3.5 w-3.5" /> Наблюдения:
      </span>
      {items.map(([k, label]) => (
        <button
          type="button"
          key={k}
          aria-pressed={layers[k]}
          className={cx(
            "rounded-full px-2.5 py-1 font-semibold ring-1 transition",
            layers[k] ? "bg-ink-900 text-white ring-ink-900 dark:bg-white dark:text-ink-900" : "text-slate-600 ring-slate-300 hover:bg-slate-50 dark:text-slate-300 dark:ring-white/20",
          )}
          onClick={() => onChange({ ...layers, [k]: !layers[k] })}
        >
          {label}
        </button>
      ))}
      <span className="text-slate-500">только то, на чём основаны события</span>
    </div>
  );
}

function countsLine(ep: ObsEpisode): string {
  const c = ep.counts ?? {};
  const parts = [
    c.observed ? `наблюдалось ${c.observed}` : "",
    c.absent ? `не видно рядом ${c.absent}` : "",
    c.unknown ? `неизвестно ${c.unknown}` : "",
  ].filter(Boolean);
  return parts.length ? `кадров: ${parts.join(", ")}` : "";
}

export default function RuleObservations({
  obs,
  selected,
  violation,
  onSeek,
}: {
  obs: Observations;
  selected: ObsEpisode[];
  violation?: Violation;
  onSeek: (t: number) => void;
}) {
  return (
    <section className="card p-4" aria-label="Наблюдения правил">
      <div className="label mb-1">Наблюдения правил</div>
      <p className="mb-3 text-xs text-slate-500">
        Что видели модели (люди, суставы, предметы, зоны) и как это сложилось во времени. Пороги — технические
        параметры распознавания, не требования норм.
      </p>
      {violation && (
        <div className="mb-3 space-y-2">
          {selected.length === 0 ? (
            <p className="text-xs text-slate-500">Для выбранного эпизода наблюдений правил нет — его нашла другая стадия анализа.</p>
          ) : (
            selected.map((ep) => {
              const checks = ep.safe_checks ?? [];
              const open = checks.filter((c) => c.state === "open").length;
              return (
                <div key={ep.id} className="rounded-lg bg-slate-50 p-2.5 text-xs dark:bg-white/5" data-episode={ep.id}>
                  <div className="flex items-center gap-2 font-semibold">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: trackColor(ep.track_id) }} />
                    <span className="min-w-0 flex-1 truncate">{RULE_RU[ep.rule] ?? ep.rule}</span>
                    <button className="font-mono text-slate-500 hover:underline tnum" onClick={() => onSeek(ep.t_anchor)}>
                      {`${formatTimecode(ep.t_start)}–${formatTimecode(ep.t_end)}`}
                    </button>
                  </div>
                  {ep.track_id !== null && (
                    <div className="mt-1 text-slate-600 dark:text-slate-300">
                      {`Человек #${ep.track_id} · ${ROLE_RU[ep.role ?? "unknown"] ?? ep.role}`}
                    </div>
                  )}
                  {countsLine(ep) && <div className="mt-0.5 text-slate-500">{countsLine(ep)}</div>}
                  {checks.length > 0 && (
                    <div className="mt-0.5 text-slate-500">
                      {`проверок сейфа: ${checks.length}, «открыт» — ${open}`}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
      <ul className="space-y-1 text-xs">
        {Object.entries(obs.rules).map(([name, r]) => (
          <li key={name} className="flex gap-2" data-rule={name}>
            <span
              className={cx(
                "mt-1 h-2 w-2 shrink-0 rounded-full",
                r.status === "ok" ? "bg-emerald-500" : r.status === "disabled" ? "bg-slate-300" : "bg-amber-500",
              )}
            />
            <span className="min-w-0 flex-1">
              <span className="font-medium">{`${RULE_RU[name] ?? name}: `}</span>
              <span className="text-slate-500">{`${STATUS_RU[r.status] ?? r.status}${r.reason ? ` — ${r.reason}` : ""}`}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
