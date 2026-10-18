"""Фиксация результатов профилей и проверка целостности (L2): только агрегаты в stdout, подробности — JSON локально.

Проверяется: соответствие видео ↔ метки (sha256 в meta против data/manifest.csv), режим и попадания кэша,
статусы стадий (ошибки, таймауты, пропуски), неполный анализ с вердиктом «нет нарушения», частота кадров.

    python tools/l2/freeze_results.py old_vlm=runs/l2/exp4 pose_rules=runs/l2/pose_rules/preds \\
        --config old_vlm=runs/l2/exp4.yaml,eval/configs/l2/vlm_states_v3.yaml \\
        --config pose_rules=backend/pipeline/profiles/pose_rules.yaml,ml/pose/presets/pose_rules.yaml,ml/seqrules/config.yaml,deploy/profiles/pose_rules.env \\
        --out runs/l2/analysis/freeze.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles", nargs="+", help="name=preds_dir")
    ap.add_argument("--config", action="append", default=[], help="name=file1,file2 — конфиги профиля")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", default="runs/l2/analysis/freeze.json")
    a = ap.parse_args()
    man = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    by_vid = {r["video_id"]: r for r in man}
    cfgs = {c.split("=", 1)[0]: [x for x in c.split("=", 1)[1].split(",") if x] for c in a.config}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    report: dict = {"git_head_at_freeze": head, "labels_sha256": sha(Path(a.labels)), "manifest_sha256": sha(Path(a.manifest)),
                    "profiles": {}}
    for spec in a.profiles:
        name, d = spec.split("=", 1)
        files = sorted(p for p in Path(d).glob("*.json") if p.name not in ("summary.json",))
        files = [p for p in files if not p.name.endswith("metrics.json")]
        agg = {k: Counter() for k in ("status", "verdict", "complete", "pipeline_version", "sample_fps", "cache_mode")}
        stage_status: dict[str, Counter] = {}
        cache_hits = Counter()
        mism, missing_sha, no_vio_incomplete, timeouts = [], 0, 0, 0
        warn_total = 0
        per_file = {}
        for p in files:
            dta = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(dta, dict) or "meta" not in dta:
                continue
            m = dta["meta"]
            vid = p.stem
            per_file[p.name] = sha(p)
            agg["status"][dta.get("status")] += 1
            agg["verdict"][dta.get("verdict")] += 1
            agg["complete"][m.get("complete")] += 1
            agg["pipeline_version"][m.get("pipeline_version")] += 1
            agg["sample_fps"][m.get("sample_fps")] += 1
            agg["cache_mode"][(m.get("cache") or {}).get("mode")] += 1
            for h in (m.get("cache") or {}).get("hits") or []:
                cache_hits[h] += 1
            for st, rec in (m.get("stages") or {}).items():
                stage_status.setdefault(st, Counter())[rec.get("status")] += 1
                if rec.get("status") == "timeout":
                    timeouts += 1
            warn_total += len(m.get("warnings") or [])
            vsha = (m.get("video") or {}).get("sha256")
            mrow = by_vid.get(vid)
            if not vsha:
                missing_sha += 1
            elif mrow is not None and mrow.get("sha256") and mrow["sha256"] != vsha:
                mism.append(vid)
            if dta.get("verdict") == "no_violation" and (dta.get("status") != "done" or m.get("complete") is not True):
                no_vio_incomplete += 1
        conf = {c: (sha(REPO / c)[:16] if (REPO / c).exists() else "missing") for c in cfgs.get(name, [])}
        rep = {"preds_dir": d, "n_files": len(per_file), "files_sha256": per_file,
               "status": dict(agg["status"]), "verdict": dict(agg["verdict"]), "complete": {str(k): v for k, v in agg["complete"].items()},
               "pipeline_version": dict(agg["pipeline_version"]), "sample_fps": {str(k): v for k, v in agg["sample_fps"].items()},
               "cache_mode": dict(agg["cache_mode"]), "cache_hits": dict(cache_hits),
               "stage_status": {k: dict(v) for k, v in stage_status.items()}, "warnings_total": warn_total,
               "video_label_sha_mismatch": mism, "missing_video_sha": missing_sha,
               "no_violation_but_incomplete": no_vio_incomplete, "stage_timeouts": timeouts, "configs_sha256_16": conf}
        report["profiles"][name] = rep
        print(f"[{name}] files={rep['n_files']} status={rep['status']} verdict={rep['verdict']} complete={rep['complete']}")
        print(f"   version={rep['pipeline_version']} fps={rep['sample_fps']} cache={rep['cache_mode']} hits={rep['cache_hits']}")
        print(f"   stages={rep['stage_status']}")
        print(f"   sha_mismatch={len(mism)} missing_sha={missing_sha} no_violation_incomplete={no_vio_incomplete} "
              f"timeouts={timeouts} warnings={warn_total} configs={conf}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("git HEAD:", head[:12], "| labels:", report["labels_sha256"][:12], "| manifest:", report["manifest_sha256"][:12], "| →", out)


if __name__ == "__main__":
    main()
