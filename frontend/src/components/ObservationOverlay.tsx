// Оверлей наблюдений правил поверх видимой области кадра: зоны, люди (bbox + номер трека + скелет), сустав,
// с которым связан предмет, и сам предмет. Только эпизоды, на которых основаны события (или выбранный эпизод).
// Координаты наблюдений нормированы к кадру; width/height — размер видимой области кадра в px.
import type { ObsEpisode, Observations } from "../api/observations";
import { COCO_EDGES, STATE_RU, frameAt, trackColor } from "../api/observations";

export interface OverlayLayers {
  skeletons: boolean;
  objects: boolean;
  zones: boolean;
  /** true — все эпизоды-основания событий; false — только выбранный эпизод. */
  all: boolean;
}

export const DEFAULT_LAYERS: OverlayLayers = { skeletons: true, objects: true, zones: true, all: false };

const ROLE_RU: Record<string, string> = { cashier: "кассир", collector: "инкассатор", visitor: "посетитель", unknown: "роль ?" };
const ZONE_RU: Record<string, string> = {
  cashier_area: "рабочая зона",
  desk: "стол",
  safe: "сейф",
  door: "дверь",
  client_area: "зона клиента",
  cash_tray: "лоток",
};
const OBJ_RU: Record<string, string> = { cell_phone: "телефон", cash: "деньги" };
const KP_MIN = 0.3;

interface Props {
  obs: Observations;
  episodes: ObsEpisode[];
  selectedIds: Set<string>;
  t: number;
  width: number;
  height: number;
  layers: OverlayLayers;
}

export default function ObservationOverlay({ obs, episodes, selectedIds, t, width, height, layers }: Props) {
  if (!(width > 0 && height > 0)) return null;
  const X = (x: number) => x * width;
  const Y = (y: number) => y * height;
  const shown = episodes.map((ep) => ({ ep, f: frameAt(ep, t) })).filter((x) => x.f !== null);
  const checks = layers.objects
    ? episodes.flatMap((ep) =>
        (ep.safe_checks ?? []).filter((c) => c.bbox && c.state === "open" && Math.abs(c.t - t) <= 1.0).map((c) => ({ ep, c })),
      )
    : [];
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Наблюдения правил на кадре"
      style={{ display: "block", overflow: "visible" }}
    >
      {layers.zones &&
        obs.zones.map((z) => {
          const work = z.name === obs.workZone;
          const pts = z.polygon.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
          const [lx, ly] = z.polygon.reduce(([ax, ay], [x, y]) => (y < ay ? [x, y] : [ax, ay]), [1, 1]);
          return (
            <g key={`z-${z.name}`} data-zone={z.name}>
              <polygon
                points={pts}
                fill={work ? "rgba(56,189,248,0.08)" : "none"}
                stroke={work ? "#38bdf8" : "rgba(226,232,240,0.55)"}
                strokeWidth={work ? 2 : 1.25}
                strokeDasharray="6 4"
              />
              <text x={X(lx) + 4} y={Math.max(12, Y(ly) + 13)} fontSize={11} fill={work ? "#7dd3fc" : "#cbd5e1"} paintOrder="stroke" stroke="rgba(2,6,23,0.8)" strokeWidth={3}>
                {ZONE_RU[z.name] ?? z.name}
              </text>
            </g>
          );
        })}
      {shown.map(({ ep, f }) => {
        const fr = f!;
        const sel = selectedIds.has(ep.id);
        const col = trackColor(ep.track_id);
        const sw = sel ? 2.5 : 1.5;
        return (
          <g key={ep.id} data-episode={ep.id} opacity={sel ? 1 : 0.6}>
            {layers.skeletons && fr.person && (
              <>
                <rect
                  x={X(fr.person[0])}
                  y={Y(fr.person[1])}
                  width={X(fr.person[2] - fr.person[0])}
                  height={Y(fr.person[3] - fr.person[1])}
                  fill="none"
                  stroke={col}
                  strokeWidth={sw}
                  rx={4}
                />
                <text x={X(fr.person[0]) + 3} y={Y(fr.person[1]) - 4} fontSize={12} fontWeight={700} fill={col} paintOrder="stroke" stroke="rgba(2,6,23,0.85)" strokeWidth={3}>
                  {[ep.track_id !== null ? `#${ep.track_id}` : "", ep.role ? ROLE_RU[ep.role] ?? ep.role : ""].filter(Boolean).join(" ")}
                </text>
              </>
            )}
            {layers.skeletons &&
              fr.kp &&
              COCO_EDGES.map(([a, b]) => {
                const pa = fr.kp![a];
                const pb = fr.kp![b];
                if (!pa || !pb || pa[2] < KP_MIN || pb[2] < KP_MIN) return null;
                return <line key={`e${a}-${b}`} x1={X(pa[0])} y1={Y(pa[1])} x2={X(pb[0])} y2={Y(pb[1])} stroke={col} strokeWidth={sw} strokeLinecap="round" />;
              })}
            {layers.skeletons &&
              fr.kp?.map((p, i) =>
                p[2] >= KP_MIN ? <circle key={`k${i}`} cx={X(p[0])} cy={Y(p[1])} r={sel ? 2.6 : 2} fill={col} /> : null,
              )}
            {layers.skeletons && fr.joint_xy && (
              <circle cx={X(fr.joint_xy[0])} cy={Y(fr.joint_xy[1])} r={7} fill="none" stroke="#fbbf24" strokeWidth={2} data-joint={fr.joint} />
            )}
            {layers.objects && fr.object && (
              <>
                <rect
                  x={X(fr.object[0])}
                  y={Y(fr.object[1])}
                  width={Math.max(2, X(fr.object[2] - fr.object[0]))}
                  height={Math.max(2, Y(fr.object[3] - fr.object[1]))}
                  fill="rgba(251,191,36,0.15)"
                  stroke="#fbbf24"
                  strokeWidth={2}
                />
                <text x={X(fr.object[2]) + 3} y={Y(fr.object[1]) + 10} fontSize={11} fill="#fde68a" paintOrder="stroke" stroke="rgba(2,6,23,0.85)" strokeWidth={3}>
                  {`${OBJ_RU[fr.object_label ?? ""] ?? fr.object_label ?? "предмет"}${fr.object_score !== undefined ? ` ${fr.object_score.toFixed(2)}` : ""}`}
                </text>
              </>
            )}
            {fr.state !== "observed" && fr.person && (
              <text x={X(fr.person[0]) + 3} y={Y(fr.person[3]) - 5} fontSize={11} fill="#fca5a5" paintOrder="stroke" stroke="rgba(2,6,23,0.85)" strokeWidth={3} data-state={fr.state}>
                {STATE_RU[fr.state]}
              </text>
            )}
          </g>
        );
      })}
      {checks.map(({ ep, c }) => (
        <g key={`s-${ep.id}-${c.t}`} data-check="safe">
          <rect x={X(c.bbox![0])} y={Y(c.bbox![1])} width={X(c.bbox![2] - c.bbox![0])} height={Y(c.bbox![3] - c.bbox![1])} fill="none" stroke="#f87171" strokeWidth={2} strokeDasharray="4 3" />
          <text x={X(c.bbox![0]) + 3} y={Y(c.bbox![1]) - 4} fontSize={11} fill="#fecaca" paintOrder="stroke" stroke="rgba(2,6,23,0.85)" strokeWidth={3}>
            {`${ep.rule === "storage" ? "место хранения: открыто" : "сейф: открыт"} (проверка ${c.t.toFixed(1)} с)`}
          </text>
        </g>
      ))}
    </svg>
  );
}
