"""Прогон папки видео через пайплайн → preds/*.json → метрики, scorecard, proxy-балл, публичная сводка.

Режимы:
* ``--mode inproc`` (по умолчанию): ``backend.pipeline.run(path, job_id, progress, deadline_s)`` в этом процессе;
* ``--mode cli``: ``python -m backend.pipeline batch <videos> --out <dir>`` (шаблон — config.yaml runner.cli_cmd);
* ``--api URL --key KEY``: через публичный API (POST /api/v1/analyze, GET /api/v1/jobs/{id}); латентность —
  от приёма файла до done, включая очередь;
* ``--preds DIR``: ничего не запускать, только метрики.

``--raw`` — сохранить ``meta.scores_raw`` (для ml.fusion.tune / eval.loo) в ``<out>/scores_raw.jsonl``.
``--subset train|calib|holdout`` — только этот split (data/splits.csv); holdout — только с ``--final``.

Все выходы — в ``runs/`` (gitignored). preds содержат кадры и описания → только локально; наружу — summary_public.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .common import REPO_ROOT, list_videos, load_config, normalize_video_id, read_json, video_id_for_path, write_json


# ----------------------------------------------------------------------------- сериализация
def result_to_api(result: Any) -> dict:
    """Сериализатор A01 (политика вердикта: no_violation только при подтверждённой полноте и т.д.).

    Fallback на asdict — ТОЛЬКО если модуля A01 нет (автономная ветка). Исключение самого сериализатора
    не маскируется: вызывающий записывает видео как failed с причиной (production-политику не обходим).
    """
    try:
        from contracts.serialize import result_to_api as _r  # type: ignore
    except ImportError:  # A01 не смерджен: asdict(AnalysisResult) ⊇ схема ТЗ
        from .common import to_jsonable

        return to_jsonable(result)
    return _r(result)


def _full_meta(result: Any) -> dict:
    meta = getattr(result, "meta", None)
    if meta is None and isinstance(result, dict):
        meta = result.get("meta")
    from .common import to_jsonable

    return to_jsonable(meta or {})


# ----------------------------------------------------------------------------- раннеры
class InprocRunner:
    def __init__(self, cfg: dict, raw: bool = False) -> None:
        self.cfg, self.raw = cfg, raw
        if raw:
            os.environ.update({k: str(v) for k, v in ((cfg.get("runner") or {}).get("raw_env") or {}).items()})
        from backend import pipeline  # type: ignore

        self.pipeline = pipeline
        if hasattr(pipeline, "warmup"):
            try:
                pipeline.warmup()
            except Exception as e:  # noqa: BLE001
                print(f"warmup: {e}", file=sys.stderr)

    def close(self) -> None:
        """Остановить воркер пайплайна (A03 release_gpu): иначе процесс eval ждёт его на выходе."""
        rel = getattr(self.pipeline, "release_gpu", None)
        if callable(rel):
            try:
                rel()
            except Exception as e:  # noqa: BLE001 — метрики уже посчитаны; сообщаем, не маскируем
                print(f"release_gpu: {e!r}", file=sys.stderr)

    def run_one(self, path: Path, video_id: str) -> tuple[dict, dict]:
        from .protocol import VramSampler, torch_peak_mb

        job_id = f"eval-{video_id}-{uuid.uuid4().hex[:6]}"
        info: dict[str, Any] = {"video_id": video_id, "file": str(path), "mode": "inproc"}
        torch_peak_mb(reset=True)
        sampler = VramSampler()
        t0 = time.perf_counter()
        try:
            with sampler:
                res = self.pipeline.run(str(path), job_id, lambda stage, p: None,
                                        deadline_s=float((self.cfg.get("runner") or {}).get("deadline_s", 240)))
            api = result_to_api(res)
            meta = _full_meta(res)
            if meta and isinstance(api, dict):
                api.setdefault("meta", {})
                for k, v in meta.items():
                    api["meta"].setdefault(k, v)
        except Exception as e:  # noqa: BLE001 — явная ошибка, не «нарушений нет»
            api = {"job_id": job_id, "status": "failed", "verdict": None, "violations": [], "error": repr(e)[:500]}
            info["error"] = repr(e)[:500]
        info["latency_s"] = round(time.perf_counter() - t0, 3)
        info.update(sampler.as_info())                     # устройство: все процессы (nvidia-smi); None — не измерено
        info["vram_eval_process_peak_mb"] = torch_peak_mb()  # процесс eval (при --inline у A03 — и пайплайн)
        return api, info


class ApiRunner:
    def __init__(self, url: str, key: str, cfg: dict, raw: bool = False) -> None:
        import requests  # лениво

        self.s = requests.Session()
        self.url, self.key, self.cfg, self.raw = url.rstrip("/"), key, cfg, raw
        r = cfg.get("runner") or {}
        self.poll_s, self.timeout_s = float(r.get("api_poll_s", 2.0)), float(r.get("api_timeout_s", 600))

    def close(self) -> None:
        self.s.close()

    def run_one(self, path: Path, video_id: str) -> tuple[dict, dict]:
        info: dict[str, Any] = {"video_id": video_id, "file": str(path), "mode": "api"}
        params = {"raw": "1"} if self.raw else None
        t_up = time.perf_counter()
        try:
            with open(path, "rb") as f:
                r = self.s.post(f"{self.url}/api/v1/analyze", headers={"X-API-Key": self.key}, params=params,
                                files={"file": (path.name, f, "video/quicktime")}, timeout=300)
            r.raise_for_status()
            job_id = r.json()["job_id"]
        except Exception as e:  # noqa: BLE001
            info.update(error=f"upload: {e!r}"[:500], latency_s=None)
            return {}, info
        t_acc = time.perf_counter()
        info["upload_s"] = round(t_acc - t_up, 3)
        api: dict = {}
        from .protocol import VramSampler

        sampler = VramSampler()       # имеет смысл, если сервис на этой же машине (L2)
        sampler.__enter__()
        while time.perf_counter() - t_acc < self.timeout_s:
            try:
                g = self.s.get(f"{self.url}/api/v1/jobs/{job_id}", headers={"X-API-Key": self.key}, params=params,
                               timeout=60)
                api = g.json() if g.ok else api
            except Exception as e:  # noqa: BLE001
                info["poll_error"] = repr(e)[:300]
            if api.get("status") in ("done", "failed"):
                break
            time.sleep(self.poll_s)
        else:
            info["timeout"] = True
            info["error"] = f"timeout {self.timeout_s:.0f} s (status={api.get('status')})"
        info["latency_s"] = round(time.perf_counter() - t_acc, 3)  # от приёма, включая очередь
        sampler.__exit__()
        info.update(sampler.as_info())   # устройство целиком; процесс/воркер — из meta ответа (если сервис отдаёт)
        return api, info


def run_cli_batch(videos: list[tuple[str, Path]], out_dir: Path, cfg: dict, raw: bool = False) -> dict[str, tuple[dict, dict]]:
    """Один вызов CLI A03 на временную папку со ссылками на выбранные видео."""
    rcfg = cfg.get("runner") or {}
    tmp_in = Path(tempfile.mkdtemp(prefix="eval_in_"))
    tmp_out = Path(tempfile.mkdtemp(prefix="eval_out_"))
    try:
        for vid, p in videos:
            dst = tmp_in / f"{vid}{p.suffix.lower()}"
            try:
                os.symlink(p.resolve(), dst)
            except OSError:
                shutil.copy2(p, dst)
        cmd = [str(c).format(python=sys.executable, videos=str(tmp_in), out=str(tmp_out)) for c in rcfg.get("cli_cmd", [])]
        if raw and rcfg.get("cli_raw_flag"):
            cmd.append(rcfg["cli_raw_flag"])
        env = dict(os.environ)  # SENTINEL_* и конфиг пайплайна уже выставлены в os.environ (apply_env)
        if raw:
            env.update({k: str(v) for k, v in (rcfg.get("raw_env") or {}).items()})
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                              timeout=float(rcfg.get("cli_timeout_s", 7200)), check=False)
        total = time.perf_counter() - t0
        found = {normalize_video_id(p.stem) or p.stem: p for p in tmp_out.rglob("*.json")}
        out: dict[str, tuple[dict, dict]] = {}
        for vid, p in videos:
            info: dict[str, Any] = {"video_id": vid, "file": str(p), "mode": "cli", "returncode": proc.returncode}
            if vid in found:
                api = read_json(found[vid])
                info["latency_s_estimated"] = round(total / max(1, len(videos)), 3)
            else:
                api = {}
                info["error"] = f"CLI не создал JSON (rc={proc.returncode}): {proc.stderr[-300:]}"
            out[vid] = (api, info)
        return out
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


# ----------------------------------------------------------------------------- выбор видео
def select_videos(videos_dir: str | Path, labels_csv: Optional[str | Path], cfg: dict, subset_ids: Optional[set] = None,
                  only: Optional[set] = None, limit: Optional[int] = None, all_members: bool = False
                  ) -> list[tuple[str, Path]]:
    files = list_videos(videos_dir, (cfg.get("runner") or {}).get("video_exts") or [".mov", ".mp4"])
    by_id: dict[str, Path] = {}
    for p in files:
        by_id.setdefault(video_id_for_path(p), p)
    skip: set[str] = set()
    if labels_csv and Path(labels_csv).exists() and not all_members:
        from .labels import read_labels_csv

        for r in read_labels_csv(labels_csv):  # дубликаты по содержимому гоняем один раз
            if r.group and r.group != r.video_id and r.group in by_id:
                skip.add(r.video_id)
    out = []
    for vid, p in by_id.items():
        if vid in skip or (only and vid not in only) or (subset_ids is not None and vid not in subset_ids):
            continue
        out.append((vid, p))
    return out[:limit] if limit else out


def apply_env(cfg: dict, pipeline_config: Optional[str] = None, env_pairs: Optional[list[str]] = None) -> dict:
    """--pipeline-config и --env K=V → os.environ ДО импорта пайплайна. Возвращает применённые пары."""
    applied: dict[str, str] = {}
    if pipeline_config:
        name = str((cfg.get("runner") or {}).get("pipeline_config_env") or "SENTINEL_PIPELINE_CONFIG")
        applied[name] = str(Path(pipeline_config).resolve())
    for pair in env_pairs or []:
        k, sep, v = pair.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--env ждёт K=V, получено {pair!r}")
        applied[k.strip()] = v
    os.environ.update(applied)
    return applied


def _extract_raw(api: dict) -> Any:
    meta = (api or {}).get("meta") or {}
    return meta.get("scores_raw")


def run_folder(videos: list[tuple[str, Path]], out_dir: Path, cfg: dict, mode: str = "inproc", api_url: str = "",
               api_key: str = "", raw: bool = False, skip_existing: bool = False,
               log: Callable[[str], None] = print) -> dict[str, dict]:
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    runinfo_path = preds_dir / "_runinfo.json"
    runinfo: dict[str, dict] = read_json(runinfo_path) if runinfo_path.exists() else {}
    todo = [(v, p) for v, p in videos if not (skip_existing and (preds_dir / f"{v}.json").exists())]
    results: dict[str, tuple[dict, dict]] = {}
    if mode == "cli":
        results = run_cli_batch(todo, out_dir, cfg, raw)
    else:
        runner = ApiRunner(api_url, api_key, cfg, raw) if mode == "api" else InprocRunner(cfg, raw)
        try:
            for i, (vid, p) in enumerate(todo, 1):
                api, info = runner.run_one(p, vid)
                results[vid] = (api, info)
                log(f"[{i}/{len(todo)}] {vid}: {api.get('status')} {api.get('verdict')} "
                    f"{info.get('latency_s')} с {('ОШИБКА ' + info['error']) if info.get('error') else ''}")
                write_json(preds_dir / f"{vid}.json", api)
                runinfo[vid] = info
                write_json(runinfo_path, runinfo)
        finally:
            runner.close()
    for vid, (api, info) in results.items():
        write_json(preds_dir / f"{vid}.json", api)
        runinfo[vid] = info
        if raw:
            sr = _extract_raw(api)
            if sr is not None:
                write_json(out_dir / "raw" / f"{vid}.json", sr)
    write_json(runinfo_path, runinfo)
    return runinfo


def write_scores_raw(out_dir: Path, labels_csv: Optional[str | Path]) -> Optional[Path]:
    """scores_raw.jsonl: {video_id, group, label, gt_times, scores_raw} — вход для ml.fusion.tune / eval.loo."""
    preds_dir = out_dir / "preds"
    gt: dict[str, Any] = {}
    if labels_csv and Path(labels_csv).exists():
        from .labels import read_labels_csv

        gt = {r.video_id: r for r in read_labels_csv(labels_csv)}
    lines = []
    for p in sorted(preds_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        api = read_json(p)
        raw_p = out_dir / "raw" / p.name
        sr = read_json(raw_p) if raw_p.exists() else _extract_raw(api)
        vid = normalize_video_id(p.stem) or p.stem
        r = gt.get(vid)
        vios = [{"t": v.get("timecode_sec"), "score": v.get("confidence"), "type_id": v.get("type_id"),
                 "source": "violations"} for v in (api.get("violations") or []) if isinstance(v, dict)]
        lines.append({"video_id": vid, "group": (r.group if r else vid), "label": (r.label if r else None),
                      "gt_times": (r.times if r else []), "verdict": api.get("verdict"),
                      "scores_raw": sr, "violations": vios})
    if not lines:
        return None
    path = out_dir / "scores_raw.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    """Флаги пайплайна (--env, --pipeline-config, raw) действуют только на время прогона: окружение процесса
    восстанавливается при выходе (вызов из тестов/ablate --inproc-arms не оставляет SENTINEL_* после себя)."""
    saved = dict(os.environ)
    try:
        return _main(argv)
    finally:
        for k in [k for k in os.environ if k not in saved]:
            del os.environ[k]
        os.environ.update(saved)


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Прогон папки через пайплайн + метрики")
    ap.add_argument("--videos", default="data/raw", help="папка видео (L1: data/raw)")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--out", default=None, help="по умолчанию runs/eval/<дата-время>")
    ap.add_argument("--mode", choices=["inproc", "cli", "api"], default="inproc")
    ap.add_argument("--api", default=None, help="URL сервиса (включает режим api)")
    ap.add_argument("--key", default=os.environ.get("SENTINEL_API_KEY", ""))
    ap.add_argument("--preds", default=None, help="готовые preds — только метрики")
    ap.add_argument("--raw", action="store_true", help="сохранить meta.scores_raw")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--subset", default=None, help="train | calib | holdout | train,calib")
    ap.add_argument("--final", action="store_true", help="разрешить holdout (обращение логируется)")
    ap.add_argument("--only", default=None, help="X-1,X-3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-members", action="store_true", help="гонять и md5-дубликаты (проверка детерминизма)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--robustness", default=None)
    ap.add_argument("--jury-sim", default=None)
    ap.add_argument("--pipeline-config", default=None, help="конфиг пайплайна → env runner.pipeline_config_env")
    ap.add_argument("--env", action="append", default=[], help="K=V флаг пайплайна (повторяемый)")
    ap.add_argument("--label", default=None, help="гипотеза/название прогона для журнала")
    ap.add_argument("--role", default="single", help="baseline | candidate | single")
    ap.add_argument("--log", default="reports/runs.jsonl", help="журнал итераций (агрегаты)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--kind", default=None, help="exploratory | holdout_frozen (по умолчанию из config)")
    ap.add_argument("--ablation-id", default=None)
    args = ap.parse_args(argv)
    if args.kind == "holdout_frozen":  # только замороженный hold-out (eval.extra freeze), открытый с --final
        man_p = Path(args.splits).with_name(Path(args.splits).stem + "_manifest.json")
        frozen = man_p.exists() and bool(read_json(man_p).get("frozen"))
        if not (frozen and args.subset == "holdout" and args.final):
            print("ВНИМАНИЕ: --kind holdout_frozen допустим только для --subset holdout --final по split'у из "
                  "eval.extra freeze; помечаю как exploratory.")
            args.kind = "exploratory"
    cfg = load_config(args.config, {"dataset": {"evaluation_kind": args.kind}} if args.kind else None)
    env_applied = apply_env(cfg, args.pipeline_config, args.env)
    if args.raw:  # до протокола: обе руки абляции фиксируют одинаковое окружение
        raw_env = {k: str(v) for k, v in ((cfg.get("runner") or {}).get("raw_env") or {}).items()}
        os.environ.update(raw_env)
        env_applied.update(raw_env)
    mode = "api" if args.api else args.mode
    out_dir = Path(args.out or f"runs/eval/{_dt.datetime.now():%Y%m%d-%H%M%S}")
    subset_ids = None
    if args.subset:
        from .split import ids_for, load_splits, log_holdout_access

        if not Path(args.splits).exists():
            ap.error(f"нет {args.splits} — сначала python -m eval.split")
        if "holdout" in args.subset:
            if not args.final:
                ap.error("holdout — только для финальных цифр: добавьте --final (обращение будет записано)")
            from .metrics import git_sha

            prev = log_holdout_access(Path(args.splits).with_name("holdout_access.jsonl"),
                                      {"git_sha": git_sha(), "out": str(out_dir), "mode": mode})
            if prev:
                print(f"ВНИМАНИЕ: holdout уже открывали {prev} раз(а) — цифры становятся оптимистичными.")
        subset_ids = ids_for(load_splits(args.splits), args.subset)
    if args.preds:
        preds_dir = Path(args.preds)
    else:
        vids = select_videos(args.videos, args.labels, cfg, subset_ids,
                             set(x.strip() for x in args.only.split(",")) if args.only else None,
                             args.limit, args.all_members)
        if not vids:
            print("нет видео для прогона")
            return 1
        print(f"{len(vids)} видео → {out_dir} (mode={mode})")
        from .protocol import build_protocol

        prot = build_protocol(label=args.label or out_dir.name, cfg=cfg, videos=vids, labels_csv=args.labels, mode=mode,
                              subset=args.subset, pipeline_config=args.pipeline_config, env_overrides=env_applied,
                              run_id=out_dir.name)
        write_json(out_dir / "protocol.json", prot)
        run_folder(vids, out_dir, cfg, mode, args.api or "", args.key, args.raw, args.skip_existing)
        preds_dir = out_dir / "preds"
        if args.raw:
            p = write_scores_raw(out_dir, args.labels)
            print(f"raw → {p}")
    if not Path(args.labels).exists():
        print(f"нет {args.labels}: метрики пропущены (python -m eval.labels ...)")
        return 0
    from .metrics import evaluate_dirs, format_scorecard, format_summary, public_summary
    from .score_like_jury import format_score, score

    prot_path = preds_dir.parent / "protocol.json"
    pc_sha = read_json(prot_path).get("pipeline_config_sha") if prot_path.exists() else None
    res = evaluate_dirs(args.labels, preds_dir, args.types, cfg, subset_ids,
                        {"subset": args.subset or "all", "mode": mode, "pipeline_config_sha": pc_sha})
    mdir = preds_dir.parent
    write_json(mdir / "metrics.json", res)
    rob = read_json(args.robustness) if args.robustness else None
    js = read_json(args.jury_sim) if args.jury_sim else None
    sc = score(res, rob, js, cfg)
    write_json(mdir / "proxy_score.json", sc)
    write_json(mdir / "summary_public.json", public_summary(res, sc))
    print(format_scorecard(res["scorecard"]))
    print(format_summary(res["summary"]))
    print(format_score(sc).split("\n", 1)[-1] if res.get("scorecard") else format_score(sc))
    if not args.no_log:
        from .runs import append, record_from_run

        append(args.log, record_from_run(mdir, args.label, args.role, args.ablation_id))
    print(f"→ {mdir}/metrics.json, proxy_score.json, summary_public.json (наружу — только summary_public.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
