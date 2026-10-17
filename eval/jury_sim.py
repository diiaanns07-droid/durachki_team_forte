"""Black-box приёмка по публичному URL (Cloudflare Tunnel): сначала ПОСЛЕДОВАТЕЛЬНО 5+ видео, потом ограниченная пачка.

Чек-лист (каждый пункт — PASS / WARN / FAIL / NOT RUN):
* auth: без ключа и с неверным ключом → 401 (POST и GET); неизвестный job → 404 (мягко); .txt → 415;
* по каждой задаче: загрузка < таймаута Cloudflare (~100 с; HTTP 524/522 — FAIL), файл MOV, схема ТЗ,
  ≤ 300 с от приёма ВКЛЮЧАЯ очередь, таймкоды и кадры в пределах длительности по PTS (ffprobe локально),
  кадры — JPEG в ориентации видео, настоящие модели (meta не mock), failed — с причиной,
  no_violation при неполном анализе (meta.coverage) — FAIL, нет незавершённых задач;
* большой файл → 413/4xx или job=failed, но не 5xx.
Итог: FAIL, если есть FAIL; INCOMPLETE, если критичный пункт NOT RUN; иначе PASS. В отчёт пишутся только
статусы и тайминги (без кадров и описаний).

CLI::

    python -m eval.jury_sim --url https://<tunnel> --key $SENTINEL_API_KEY --videos data/raw --n 6 --concurrency 3
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import ffprobe_info, list_videos, load_config, video_id_for_path, write_json
from .schema import check_api_response, coverage_problem, coverage_ratio

PASS, WARN, FAIL, NOT_RUN = "PASS", "WARN", "FAIL", "NOT RUN"
CF_ERRORS = (520, 521, 522, 523, 524)
MOCK_MARKERS = ("mock", "fake", "dummy", "stub")


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    seconds: Optional[float] = None
    critical: bool = True
    data: dict[str, Any] = field(default_factory=dict)


def jpeg_size(raw: bytes) -> Optional[tuple[int, int]]:
    """(w, h) из SOF-маркера JPEG без декодеров."""
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(raw):
        if raw[i] != 0xFF:
            i += 1
            continue
        m = raw[i + 1]
        if m in (0xD8, 0x01, 0xFF) or 0xD0 <= m <= 0xD7:
            i += 2 if m != 0xFF else 1
            continue
        seg = int.from_bytes(raw[i + 2:i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return int.from_bytes(raw[i + 7:i + 9], "big"), int.from_bytes(raw[i + 5:i + 7], "big")
        i += 2 + seg
    return None


def inspect_result(res: dict, video_info: dict, cfg: Optional[dict] = None) -> dict:
    """Проверки одного ответа (без сети): PTS-границы, JPEG/ориентация, mock, явные ошибки, partial."""
    out: dict[str, Any] = {"pts_issues": [], "jpeg_issues": [], "mock": None, "models_known": False,
                           "error_explained": None, "partial_as_clean": False, "coverage": None}
    dur = float(video_info.get("duration") or 0)
    w, h = int(video_info.get("width") or 0), int(video_info.get("height") or 0)
    for v in res.get("violations") or []:
        if not isinstance(v, dict):
            continue
        tc = v.get("timecode_sec")
        if dur and isinstance(tc, (int, float)) and not (0 <= tc <= dur + 0.5):
            out["pts_issues"].append(f"timecode {tc} вне [0, {dur:.1f}]")
        for f in v.get("evidence_frames") or []:
            if not isinstance(f, dict):
                continue
            ts = f.get("timestamp_sec")
            if dur and isinstance(ts, (int, float)) and not (0 <= ts <= dur + 0.5):
                out["pts_issues"].append(f"кадр {ts} вне [0, {dur:.1f}]")
            try:
                b = f.get("image_base64") or ""
                raw = base64.b64decode(b.split(",", 1)[-1] if b.startswith("data:") else b, validate=True)
            except Exception:  # noqa: BLE001
                raw = b""
            size = jpeg_size(raw)
            if size is None:
                out["jpeg_issues"].append("не JPEG")
            elif w and h and size[0] and size[1]:
                if abs((size[0] / size[1]) / (w / h) - 1) > 0.05:
                    out["jpeg_issues"].append(f"ориентация/пропорции кадра {size[0]}×{size[1]} ≠ видео {w}×{h}")
    meta = res.get("meta") or {}
    models = meta.get("models") or meta.get("model") or []
    names = [str(m.get("name") if isinstance(m, dict) else m).lower() for m in (models if isinstance(models, list)
                                                                              else [models])]
    out["models_known"] = bool(names)
    out["mock"] = bool(meta.get("mock")) or any(k in n for n in names for k in MOCK_MARKERS) \
        or str(meta.get("backend") or "").lower() in MOCK_MARKERS
    if res.get("status") == "failed":
        out["error_explained"] = bool(res.get("error") or res.get("detail") or meta.get("error") or meta.get("errors"))
    out["coverage"] = coverage_ratio(meta)                       # канон A01: meta.coverage.ratio
    out["coverage_reported"] = meta.get("coverage") is not None or meta.get("complete") is not None
    if res.get("verdict") == "no_violation" and out["coverage_reported"]:
        out["coverage_problem"] = coverage_problem(meta, cfg)       # та же семантика, что у сериализатора A01
        out["partial_as_clean"] = out["coverage_problem"] is not None
    return out


def judge_accept_to_done(accepted: list[dict], limit_s: float) -> tuple[str, str]:
    """≤ limit от приёма (вкл. очередь). 0.0 с (мгновенный ответ, dedup/кэш) — валидное время, не таймаут;
    None — время не измерено: такой job не PASS и не таймаут, пункт NOT RUN (критичный)."""
    if not accepted:
        return NOT_RUN, "нет принятых задач"
    known = [r for r in accepted if r.get("accept_to_done_s") is not None]
    unknown = [r["video"] for r in accepted if r.get("accept_to_done_s") is None]
    slow = [r["video"] for r in known if float(r["accept_to_done_s"]) > limit_s]
    mx = max((float(r["accept_to_done_s"]) for r in known), default=None)
    detail = (f"max {mx:.1f} с" if mx is not None else "время не измерено") + \
        f"; > лимита: {slow or '—'}; не измерено: {unknown or '—'}"
    if slow:
        return FAIL, detail
    if unknown:
        return NOT_RUN, detail
    return PASS, detail


class JurySim:
    def __init__(self, url: str, key: str, cfg: Optional[dict] = None, session: Any = None) -> None:
        import requests  # лениво

        full = cfg or load_config()
        self.url, self.key = url.rstrip("/"), key
        self.cfg = full.get("jury_sim") or {}
        self.full_cfg = full
        self.s = session or requests.Session()
        self.req_timeout = float(self.cfg.get("request_timeout_s", 120))
        self.poll_s = float(self.cfg.get("poll_s", 2.0))
        self.limit_s = float(self.cfg.get("total_limit_s", 300.0))
        self.cf_s = float(self.cfg.get("cloudflare_timeout_s", 100.0))
        self.min_seq = int(self.cfg.get("min_sequential", 5))
        self.checks: list[Check] = []
        self._probe: dict[str, dict] = {}

    # ------------------------------------------------------------------ http
    def _post(self, path: Path, key: Optional[str], ctype: str = "video/quicktime") -> Any:
        headers = {"X-API-Key": key} if key is not None else {}
        with open(path, "rb") as f:
            return self.s.post(f"{self.url}/api/v1/analyze", headers=headers, files={"file": (path.name, f, ctype)},
                               timeout=self.req_timeout)

    def _get(self, job_id: str, key: Optional[str]) -> Any:
        headers = {"X-API-Key": key} if key is not None else {}
        return self.s.get(f"{self.url}/api/v1/jobs/{job_id}", headers=headers, timeout=self.req_timeout)

    def add(self, *a: Any, **kw: Any) -> Check:
        c = Check(*a, **kw)
        self.checks.append(c)
        return c

    def probe(self, p: Path) -> dict:
        if str(p) not in self._probe:
            self._probe[str(p)] = ffprobe_info(p)
        return self._probe[str(p)]

    # ------------------------------------------------------------------ фазы
    def check_auth(self, sample: Path) -> None:
        for name, key in (("auth: POST без ключа → 401", None), ("auth: POST с неверным ключом → 401", "wrong-key-xyz")):
            try:
                r = self._post(sample, key)
                self.add(name, PASS if r.status_code == 401 else FAIL, f"HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001
                self.add(name, FAIL, repr(e)[:200])
        try:
            r = self._get("nonexistent-job", None)
            self.add("auth: GET jobs без ключа → 401", PASS if r.status_code == 401 else FAIL, f"HTTP {r.status_code}")
            r = self._get("nonexistent-job-000", self.key)
            self.add("GET неизвестного job → 404", PASS if r.status_code == 404 else WARN, f"HTTP {r.status_code}",
                     critical=False)
        except Exception as e:  # noqa: BLE001
            self.add("auth: GET jobs", FAIL, repr(e)[:200])

    def check_wrong_file(self, tmp: Path) -> None:
        p = tmp / "not_a_video.txt"
        p.write_text("это не видео", encoding="utf-8")
        try:
            r = self._post(p, self.key, "text/plain")
            st = PASS if r.status_code == 415 else (WARN if r.status_code in (400, 422) else FAIL)
            self.add("неверный файл (.txt) → 415", st, f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            self.add("неверный файл (.txt) → 415", FAIL, repr(e)[:200])

    def check_big_file(self, tmp: Path, mb: int) -> None:
        name = f"большой файл ({mb} МБ мусора) → 413/4xx или failed, не 5xx"
        if mb <= 0:
            self.add(name, NOT_RUN, "big_file_mb=0", critical=False)
            return
        p = tmp / "big_random.mov"
        with open(p, "wb") as f:
            for _ in range(mb):
                f.write(os.urandom(1 << 20))
        t0 = time.perf_counter()
        try:
            r = self._post(p, self.key)
            if r.status_code in (413, 400, 415, 422):
                self.add(name, PASS, f"HTTP {r.status_code}", time.perf_counter() - t0)
            elif r.status_code < 300:
                job = r.json().get("job_id")
                st = self._wait(job, time.perf_counter())
                self.add(name, PASS if st.get("status") == "failed" else WARN, f"принят, job → {st.get('status')}",
                         time.perf_counter() - t0)
            else:
                self.add(name, FAIL, f"HTTP {r.status_code}", time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            self.add(name, FAIL, repr(e)[:200], time.perf_counter() - t0)
        finally:
            p.unlink(missing_ok=True)

    def _wait(self, job_id: str, t_accept: float) -> dict:
        last: dict = {}
        while time.perf_counter() - t_accept < self.limit_s + float(self.cfg.get("grace_s", 60)):
            try:
                r = self._get(job_id, self.key)
                if r.ok:
                    last = r.json()
            except Exception:  # noqa: BLE001
                pass
            if last.get("status") in ("done", "failed"):
                break
            time.sleep(self.poll_s)
        last["_elapsed_from_accept_s"] = round(time.perf_counter() - t_accept, 2)
        return last

    def _job(self, p: Path) -> dict:
        rec: dict[str, Any] = {"video": video_id_for_path(p), "is_mov": p.suffix.lower() == ".mov",
                               "size_mb": round(p.stat().st_size / 2 ** 20, 2)}
        t0 = time.perf_counter()
        try:
            r = self._post(p, self.key)
            rec["upload_http"] = r.status_code
            rec["upload_s"] = round(time.perf_counter() - t0, 2)
            if r.status_code >= 300:
                rec["error"] = f"HTTP {r.status_code}" + (" (Cloudflare)" if r.status_code in CF_ERRORS else "")
                rec["state"] = "not_accepted"
                return rec
            job_id = r.json()["job_id"]
        except Exception as e:  # noqa: BLE001
            rec["upload_s"] = round(time.perf_counter() - t0, 2)
            rec["error"] = repr(e)[:200]
            rec["state"] = "not_accepted"
            return rec
        t_acc = time.perf_counter()
        res = self._wait(job_id, t_acc)
        rec["accept_to_done_s"] = res.pop("_elapsed_from_accept_s", None)
        rec["status"] = res.get("status")
        rec["state"] = res.get("status") if res.get("status") in ("done", "failed") else "unfinished"
        rec["verdict"] = res.get("verdict")
        rec["n_violations"] = len(res.get("violations") or [])
        rec["schema_errors"] = check_api_response(res, require_done=False).errors[:5]
        rec.update(inspect_result(res, self.probe(p), self.full_cfg))
        return rec

    def _judge(self, phase: str, recs: list[dict], wall: float) -> None:
        acc = [r for r in recs if r.get("state") != "not_accepted"]
        fin = [r for r in acc if r.get("state") in ("done", "failed")]

        def ids(pred) -> list[str]:  # noqa: ANN001
            return [r["video"] for r in recs if pred(r)]

        up = [r.get("upload_s") for r in recs]                  # None — не измерено (не 0)
        up_known = [u for u in up if u is not None]
        cf_bad = ids(lambda r: (r.get("upload_s") is not None and r["upload_s"] >= self.cf_s)
                     or r.get("upload_http") in CF_ERRORS)
        cf_warn = ids(lambda r: r.get("upload_s") is not None and 0.6 * self.cf_s <= r["upload_s"] < self.cf_s)
        up_none = ids(lambda r: r.get("upload_s") is None)
        self.add(f"{phase}: загрузка < {self.cf_s:g} с (таймаут Cloudflare)",
                 FAIL if cf_bad else (NOT_RUN if not up_known else (WARN if cf_warn or up_none else PASS)),
                 (f"max {max(up_known):.1f} с" if up_known else "время загрузки не измерено")
                 + f"; ≥ лимита: {cf_bad or '—'}; ≥ 60%: {cf_warn or '—'}; не измерено: {up_none or '—'}", wall,
                 data={"upload_s": up})
        not_acc = ids(lambda r: r.get("state") == "not_accepted")
        self.add(f"{phase}: все файлы приняты", FAIL if not_acc else PASS, f"не приняты: {not_acc or '—'}")
        non_mov = ids(lambda r: not r.get("is_mov"))
        self.add(f"{phase}: формат MOV", WARN if non_mov else PASS, f"не .mov: {non_mov or '—'}", critical=False)
        unfinished = ids(lambda r: r.get("state") == "unfinished")
        self.add(f"{phase}: все принятые задачи завершены", FAIL if unfinished else (PASS if acc else NOT_RUN),
                 f"незавершённых: {unfinished or '—'} из {len(acc)}")
        verdict_t, detail_t = judge_accept_to_done(acc, self.limit_s)
        self.add(f"{phase}: ≤ {self.limit_s:g} с от приёма, включая очередь", verdict_t, detail_t,
                 data={"accept_to_done_s": [r.get("accept_to_done_s") for r in recs]})
        if not fin:
            for name in ("схема ТЗ", "таймкоды и кадры в пределах длительности (PTS)", "кадры — JPEG в ориентации видео",
                         "настоящие модели (не mock)", "failed — с явной причиной",
                         "неполный анализ не выдан за no_violation"):
                self.add(f"{phase}: {name}", NOT_RUN, "нет завершённых задач")
            return
        bad_schema = [r["video"] for r in fin if r.get("schema_errors")]
        self.add(f"{phase}: схема ТЗ", FAIL if bad_schema else PASS, f"ошибки: {bad_schema or '—'}")
        no_probe = [r["video"] for r in fin if not self.probe_ok(r)]
        pts = [r["video"] for r in fin if r.get("pts_issues")]
        self.add(f"{phase}: таймкоды и кадры в пределах длительности (PTS)",
                 FAIL if pts else (NOT_RUN if len(no_probe) == len(fin) else PASS),
                 f"вне длительности: {pts or '—'}" + (f"; нет ffprobe: {no_probe}" if no_probe else ""))
        jp = [r["video"] for r in fin if r.get("jpeg_issues")]
        not_jpeg = [r["video"] for r in fin if any(x == "не JPEG" for x in r.get("jpeg_issues") or [])]
        self.add(f"{phase}: кадры — JPEG в ориентации видео", FAIL if not_jpeg else (WARN if jp else PASS),
                 f"проблемы: {jp or '—'}")
        mock = [r["video"] for r in fin if r.get("mock")]
        unknown = [r["video"] for r in fin if not r.get("models_known")]
        self.add(f"{phase}: настоящие модели (не mock)", FAIL if mock else (WARN if unknown else PASS),
                 f"mock: {mock or '—'}; meta.models не сообщены: {unknown or '—'}")
        failed = [r for r in fin if r.get("state") == "failed"]
        unexplained = [r["video"] for r in failed if not r.get("error_explained")]
        self.add(f"{phase}: failed — с явной причиной", FAIL if unexplained else PASS,
                 f"failed: {len(failed)}; без причины: {unexplained or '—'}")
        pc = [r["video"] for r in fin if r.get("partial_as_clean")]
        cov_unknown = [r["video"] for r in fin if r.get("verdict") == "no_violation" and not r.get("coverage_reported")]
        self.add(f"{phase}: неполный анализ не выдан за no_violation", FAIL if pc else (WARN if cov_unknown else PASS),
                 f"partial→no_violation: {pc or '—'}; no_violation без meta.coverage: {cov_unknown or '—'}",
                 data={"jobs": recs})

    def probe_ok(self, rec: dict) -> bool:
        info = next((v for k, v in self._probe.items() if video_id_for_path(k) == rec["video"]), {})
        return bool(info.get("duration"))

    def sequential(self, videos: list[Path]) -> None:
        if len(videos) < self.min_seq:
            self.add(f"sequential: ≥ {self.min_seq} видео подряд", NOT_RUN, f"дано только {len(videos)}")
        else:
            self.add(f"sequential: ≥ {self.min_seq} видео подряд", PASS, f"{len(videos)} видео")
        t0 = time.perf_counter()
        recs = [self._job(p) for p in videos]
        self._judge("sequential", recs, time.perf_counter() - t0)

    def stress(self, videos: list[Path], concurrency: int) -> None:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            recs = list(ex.map(self._job, videos))
        self._judge(f"batch (≤ {concurrency} параллельно)", recs, time.perf_counter() - t0)

    def summary(self) -> dict:
        counted = [c for c in self.checks if c.status != NOT_RUN]
        n_fail = sum(c.status == FAIL for c in self.checks)
        crit_not_run = [c.name for c in self.checks if c.status == NOT_RUN and c.critical]
        overall = FAIL if n_fail else ("INCOMPLETE" if crit_not_run else PASS)
        return {"overall": overall, "n_checks": len(self.checks), "n_pass": sum(c.status == PASS for c in self.checks),
                "n_warn": sum(c.status == WARN for c in self.checks), "n_fail": n_fail,
                "n_not_run": sum(c.status == NOT_RUN for c in self.checks), "critical_not_run": crit_not_run,
                "pass_rate": round(sum(c.status == PASS for c in counted) / len(counted), 3) if counted else None,
                "limit_s": self.limit_s, "cloudflare_timeout_s": self.cf_s,
                "url_host": self.url.split("//")[-1].split("/")[0]}


def run_jury_sim(url: str, key: str, videos: list[Path], cfg: Optional[dict] = None, concurrency: int = 3,
                 skip_stress: bool = False, big_mb: Optional[int] = None, session: Any = None) -> dict:
    sim = JurySim(url, key, cfg, session)
    tmp = Path(tempfile.mkdtemp(prefix="jury_sim_"))
    if videos:
        sim.check_auth(videos[0])
    else:
        sim.add("auth", NOT_RUN, "нет видео")
    sim.check_wrong_file(tmp)
    if videos:
        sim.sequential(videos)          # сначала по одному
        if skip_stress:
            sim.add("batch", NOT_RUN, "--skip-stress", critical=False)
        else:
            sim.stress(videos, concurrency)
    else:
        sim.add("sequential", NOT_RUN, "нет видео")
    sim.check_big_file(tmp, int(sim.cfg.get("big_file_mb", 150) if big_mb is None else big_mb))
    return {"summary": sim.summary(), "checks": [asdict(c) for c in sim.checks]}


def format_table(res: dict) -> str:
    lines = [f"{'статус':8}  {'сек':>7}  проверка — детали"]
    for c in res["checks"]:
        sec = f"{c['seconds']:.1f}" if c.get("seconds") is not None else "—"
        lines.append(f"{c['status']:8}  {sec:>7}  {c['name']} — {c['detail']}")
    s = res["summary"]
    lines.append(f"ИТОГ: {s['overall']}  PASS {s['n_pass']} / WARN {s['n_warn']} / FAIL {s['n_fail']} / "
                 f"NOT RUN {s['n_not_run']}" + (f"; критичные NOT RUN: {s['critical_not_run']}" if s["critical_not_run"]
                                                 else ""))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Black-box приёмка по публичному URL")
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("SENTINEL_API_KEY", ""))
    ap.add_argument("--videos", default="data/raw", help="папка видео (L1: data/raw)")
    ap.add_argument("--n", type=int, default=None, help="сколько видео (по умолчанию config jury_sim.n_videos)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--big-mb", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/jury_sim.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    n = args.n or int((cfg.get("jury_sim") or {}).get("n_videos", 6))
    vids = list_videos(args.videos) if Path(args.videos).exists() else []
    random.Random(args.seed).shuffle(vids)
    vids = sorted(vids[:n], key=lambda p: p.stat().st_size, reverse=True)  # самые длинные — первыми
    res = run_jury_sim(args.url, args.key, vids, cfg, args.concurrency, args.skip_stress, args.big_mb)
    write_json(args.out, res)
    print(format_table(res))
    print(f"→ {args.out}")
    return 0 if res["summary"]["overall"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
