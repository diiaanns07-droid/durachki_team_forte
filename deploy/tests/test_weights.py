import hashlib
import json
from pathlib import Path

import pytest

from deploy import license_policy as lp
from deploy import weights as w

COMMIT = "a" * 40


class FakeHub:
    """Имитирует хаб: метаданные ревизии + раскладка кэша huggingface_hub."""

    def __init__(self, files: dict[str, bytes], license: str = "apache-2.0", lfs: bool = True):
        self.files, self.license, self.lfs, self.downloads = files, license, lfs, []

    def info(self, repo_id, revision):
        meta = []
        for p, data in self.files.items():
            meta.append({"path": p, "size": len(data),
                         "sha256": hashlib.sha256(data).hexdigest() if self.lfs else None,
                         "blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()})
        return {"sha": COMMIT, "license": self.license, "files": meta}

    def download(self, repo_id, commit, files, cache_dir):
        self.downloads.append((repo_id, commit, tuple(files)))
        snap = w.snapshot_dir(Path(cache_dir), repo_id, commit)
        for p in files:
            (snap / p).parent.mkdir(parents=True, exist_ok=True)
            (snap / p).write_bytes(self.files[p])
        return snap


def _models(root: Path, module: str, items) -> None:
    d = root / module
    d.mkdir(parents=True, exist_ok=True)
    (d / "models.json").write_text(json.dumps(items), encoding="utf-8")


@pytest.fixture(autouse=True)
def _no_hf_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)


def test_load_specs_lenient_and_errors(tmp_path):
    _models(tmp_path, "ml/detect", [{"name": "dfine", "hf_id_or_url": "ustc-community/dfine-small-coco",
                                     "license": "Apache-2.0", "size_mb": 40, "purpose": "det"}])
    _models(tmp_path, "ml/vlm", {"models": [{"Name": "qwen", "HF_ID_OR_URL": "https://huggingface.co/Qwen/X/tree/v1",
                                             "license": "apache-2.0"}]})
    (tmp_path / "ml/bad").mkdir(parents=True)
    (tmp_path / "ml/bad/models.json").write_text("{oops", encoding="utf-8")
    errors = []
    specs = w.load_specs(tmp_path, errors)
    assert [s.name for s in specs] == ["dfine", "qwen"]
    assert specs[1].kind == "hf" and specs[1].repo_id == "Qwen/X" and specs[1].revision == "v1"
    assert len(errors) == 1 and "ml/bad" in errors[0]


@pytest.mark.parametrize("src,kind,repo,fname", [
    ("google/siglip2-base-patch16-224", "hf", "google/siglip2-base-patch16-224", ""),
    ("https://huggingface.co/org/rep/resolve/main/m.onnx", "hf_file", "org/rep", "m.onnx"),
    ("https://github.com/x/y/releases/download/v1/rtm.onnx", "url", "", "rtm.onnx"),
    ("bundled:mediapipe", "bundled", "", ""),
    ("https://pypi.org/project/mediapipe/", "bundled", "", ""),
])
def test_classify_source(src, kind, repo, fname):
    k, r, f, _ = w.classify_source(src)
    assert (k, r, f) == (kind, repo, fname)


def test_license_policy_specific_not_family():
    assert lp.is_allowed("Apache 2.0") and lp.is_allowed("MIT") and lp.is_allowed("BSD-3-Clause")
    assert not lp.is_allowed("cc-by-nc-4.0") and not lp.is_allowed("") and not lp.is_allowed("other")
    assert lp.is_allowed("MIT OR GPL-3.0") and not lp.is_allowed("MIT AND GPL-3.0")
    # имя «разрешённого» семейства ничего не разрешает — решает лицензия ревизии на хабе
    ok, why = w.license_verdict("apache-2.0", "qwen-research", [])
    assert not ok and "хаб" in why
    ok, _ = w.license_verdict("apache-2.0", "", [])
    assert not ok
    ok, why = w.license_verdict("mit", "apache-2.0", [])
    assert not ok and "заявлено" in why
    assert lp.banned_reasons("Qwen/Qwen2.5-VL-3B-Instruct") and not lp.banned_reasons("Qwen/Qwen2.5-VL-7B-Instruct")
    assert lp.banned_reasons("Ultralytics/YOLOv8") and not lp.banned_reasons("Megvii/YOLOX-s")
    assert lp.banned_reasons("MCG-NJU/videomae-base") and not lp.banned_reasons("Qwen/Qwen2.5-1.5B-Instruct")


def test_select_files_defaults():
    files = ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5", "onnx/model.onnx",
             "tokenizer.json", ".gitattributes"]
    assert w.select_files(files, []) == ["config.json", "model.safetensors", "tokenizer.json"]
    assert w.select_files(files, ["onnx/*"]) == ["onnx/model.onnx"]
    with pytest.raises(ValueError):
        w.select_files(["a.Q4_K_M.gguf", "a.Q8_0.gguf"], [])


def test_prepare_pins_commit_hashes_and_verify(tmp_path):
    root, weights = tmp_path / "repo", tmp_path / "weights"
    _models(root, "ml/scene", [{"name": "siglip", "hf_id_or_url": "google/siglip2", "license": "Apache-2.0",
                                "size_mb": 1, "purpose": "emb"}])
    hub = FakeHub({"config.json": b"{}", "model.safetensors": b"\x00" * 1000, "pytorch_model.bin": b"dup"})
    lock_path = root / "deploy" / "weights.lock.json"
    lock_path.parent.mkdir(parents=True)
    code, lock = w.prepare(w.load_specs(root), weights, hub, lock_path=lock_path, log=lambda *_: None)
    assert code == 0
    m = lock["models"][0]
    assert m["revision"] == COMMIT and m["license_ok"] and m["license_hub"] == "apache-2.0"
    assert [f["path"] for f in m["files"]] == ["config.json", "model.safetensors"]
    assert all(len(f["sha256"]) == 64 for f in m["files"])
    assert hub.downloads == [("google/siglip2", COMMIT, ("config.json", "model.safetensors"))]
    ref = w.repo_cache_dir(weights / "hf" / "hub", "google/siglip2") / "refs" / "main"
    assert ref.read_text() == COMMIT  # from_pretrained(repo) офлайн найдёт снапшот
    assert json.loads((weights / "MANIFEST.json").read_text())["models"][0]["status"] == "ok"

    code, res = w.verify(weights, w.load_lock(lock_path), log=lambda *_: None)
    assert code == 0 and res[0]["ok"]
    # порча файла ловится по sha256, но не в --fast (размер тот же)
    snap = w.snapshot_dir(weights / "hf" / "hub", "google/siglip2", COMMIT)
    (snap / "model.safetensors").write_bytes(b"\x01" * 1000)
    assert w.verify(weights, w.load_lock(lock_path), fast=True, log=lambda *_: None)[0] == 0
    code, res = w.verify(weights, w.load_lock(lock_path), log=lambda *_: None)
    assert code == 1 and "sha256" in res[0]["problems"][0]
    (snap / "config.json").unlink()
    assert "нет файла" in " ".join(w.verify(weights, w.load_lock(lock_path), log=lambda *_: None)[1][0]["problems"])


def test_prepare_reuses_locked_revision(tmp_path):
    root, weights = tmp_path / "repo", tmp_path / "weights"
    _models(root, "ml/x", [{"name": "m", "hf_id_or_url": "o/m", "license": "MIT"}])
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps({"version": 1, "models": [
        {"name": "m", "source": "o/m", "revision": "b" * 40, "module": "ml/x", "files": []}]}))
    seen = []

    class Hub(FakeHub):
        def info(self, repo_id, revision):
            seen.append(revision)
            return super().info(repo_id, revision)

    w.prepare(w.load_specs(root), weights, Hub({"model.safetensors": b"1"}, license="mit"),
              lock_path=lock_path, dry_run=True, log=lambda *_: None)
    assert seen == ["b" * 40]


def test_prepare_refuses_red_license(tmp_path):
    root = tmp_path / "repo"
    _models(root, "ml/t", [{"name": "vmae", "hf_id_or_url": "MCG-NJU/videomae-base", "license": "cc-by-nc-4.0"}])
    hub = FakeHub({"model.safetensors": b"x"}, license="cc-by-nc-4.0")
    code, _ = w.prepare(w.load_specs(root), tmp_path / "w", hub, lock_path=tmp_path / "l.json", log=lambda *_: None)
    assert code == 2 and not hub.downloads


def test_url_model_sha256_checked(tmp_path, monkeypatch):
    payload = b"onnx-bytes"
    src = tmp_path / "src.onnx"
    src.write_bytes(payload)
    root = tmp_path / "repo"
    _models(root, "ml/pose", [{"name": "rtm", "hf_id_or_url": src.as_uri().replace("file://", "https://x"),
                               "license": "Apache-2.0", "sha256": hashlib.sha256(payload).hexdigest()}])
    specs = w.load_specs(root)
    assert specs[0].kind == "url"
    monkeypatch.setattr(w, "http_download",
                        lambda url, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(payload)))
    code, lock = w.prepare(specs, tmp_path / "w", None, lock_path=tmp_path / "l.json",
                           allow_license_flags=True, log=lambda *_: None)
    m = lock["models"][0]
    assert code == 0 and m["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not m["license_ok"]  # URL: лицензию хаб не подтверждает → ручная проверка
    assert w.verify(tmp_path / "w", lock, log=lambda *_: None)[1][0]["problems"][0].startswith("лицензия")


def test_licenses_md(tmp_path):
    rows = [
        {"name": "dfine", "module": "ml/detect", "source": "a/b", "revision": COMMIT, "license_declared": "Apache-2.0",
         "license_hub": "apache-2.0", "license_ok": True, "license_note": "ok"},
        {"name": "bad|name", "module": "ml/t", "source": "c/d", "license_declared": "cc-by-nc-4.0",
         "license_hub": "cc-by-nc-4.0", "license_ok": False, "license_note": "нет"},
    ]
    md = w.render_licenses(rows)
    assert "✅ да" in md and "🔴 НЕТ" in md and "bad\\|name" in md and "a/b@aaaaaaaaaaaa" in md
    assert "красных флагов: 1" in md
    assert "models.json модулей ещё не смерджены" in w.render_licenses([])


def test_cli_licenses_and_verify_empty(tmp_path, capsys):
    root = tmp_path / "repo"
    _models(root, "ml/detect", [{"name": "dfine", "hf_id_or_url": "ustc-community/dfine-small-coco",
                                 "license": "Apache-2.0"}])
    (root / "deploy").mkdir()
    assert w.main(["licenses", "--root", str(root)]) == 1  # не подтверждено хабом → красный
    assert "weights.py prepare" in (root / "deploy" / "LICENSES.md").read_text(encoding="utf-8")
    # модель объявлена, но не подготовлена → verify не зелёный
    mj = ["--models-from", "models-json"]
    assert w.main(["verify", "--root", str(root), "--weights", str(tmp_path / "w")] + mj) == 1
    (root / "ml/detect/models.json").write_text("[]", encoding="utf-8")
    assert w.main(["verify", "--root", str(root), "--weights", str(tmp_path / "w")] + mj) == 0
    # по умолчанию обязательный список — из deploy.yaml: пустой lock → FAIL
    assert w.main(["verify", "--root", str(root), "--weights", str(tmp_path / "w")]) == 1


def _fake_cache_repo(hub: Path, repo: str, files: dict[str, bytes], commit: str = COMMIT, lfs=(".safetensors",),
                     readme_license: str | None = "apache-2.0", symlinks: bool = True) -> Path:
    """Раскладка huggingface_hub: blobs/<sha256|sha1>, snapshots/<commit>/<file> → ../../blobs/..., refs/main."""
    rdir = w.repo_cache_dir(hub, repo)
    snap = rdir / "snapshots" / commit
    (rdir / "blobs").mkdir(parents=True, exist_ok=True)
    (rdir / "refs").mkdir(parents=True, exist_ok=True)
    (rdir / "refs" / "main").write_text(commit)
    if readme_license is not None:
        files = {**files, "README.md": f"---\nlicense: {readme_license}\ntags: [x]\n---\n# card\n".encode()}
    for rel, data in files.items():
        if rel.endswith(lfs):
            blob = hashlib.sha256(data).hexdigest()
        else:
            blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        (rdir / "blobs" / blob).write_bytes(data)
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if symlinks:
            dst.symlink_to(Path("../../blobs") / blob if "/" not in rel else Path("../../../blobs") / blob)
        else:
            dst.write_bytes(data)
    return rdir


def test_adopt_existing_cache_and_verify(tmp_path):
    weights = tmp_path / "weights"
    hub = weights / "hf" / "hub"
    shard_idx = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                           "b": "model-00002-of-00002.safetensors"}}).encode()
    _fake_cache_repo(hub, "Qwen/Qwen3-VL-4B-Instruct", {
        "config.json": b"{}", "model.safetensors.index.json": shard_idx,
        "model-00001-of-00002.safetensors": b"\x01" * 64, "model-00002-of-00002.safetensors": b"\x02" * 64})
    _fake_cache_repo(hub, "PekingU/rtdetr_v2_r18vd", {"config.json": b"{}", "model.safetensors": b"\x03" * 32},
                     symlinks=False)  # Windows без symlink: файлы прямо в снапшоте
    lock_path = tmp_path / "lock.json"
    models = [{"repo": "Qwen/Qwen3-VL-4B-Instruct", "module": "ml/vlm"},
              {"repo": "PekingU/rtdetr_v2_r18vd", "module": "ml/detect"},
              {"repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF", "module": "ml/vlm"}]
    code, lock = w.adopt(models, weights, lock_path, exclude=["*GGUF*"], log=lambda *_: None)
    assert code == 0
    by = {m["name"]: m for m in lock["models"]}
    assert set(by) == {"Qwen/Qwen3-VL-4B-Instruct", "PekingU/rtdetr_v2_r18vd"}   # GGUF исключён
    q = by["Qwen/Qwen3-VL-4B-Instruct"]
    assert q["revision"] == COMMIT and q["license_ok"] and q["license_hub"] == "apache-2.0"
    assert all(len(f["sha256"]) == 64 for f in q["files"] if f["path"].endswith(".safetensors"))
    assert w.verify(weights, lock, log=lambda *_: None)[0] == 0
    # порча шарда: размер тот же, sha256 — нет
    snap = w.snapshot_dir(hub, "Qwen/Qwen3-VL-4B-Instruct", COMMIT)
    (snap / "model-00002-of-00002.safetensors").resolve().write_bytes(b"\x09" * 64)
    code, res = w.verify(weights, lock, log=lambda *_: None)
    assert code == 1 and any("sha256" in p for r in res for p in r["problems"])


def test_adopt_detects_missing_shard_incomplete_and_bad_license(tmp_path):
    weights = tmp_path / "weights"
    hub = weights / "hf" / "hub"
    idx = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                     "b": "model-00002-of-00002.safetensors"}}).encode()
    rdir = _fake_cache_repo(hub, "Qwen/Qwen3-VL-4B-Instruct", {
        "config.json": b"{}", "model.safetensors.index.json": idx,
        "model-00001-of-00002.safetensors": b"\x01" * 8})
    (rdir / "blobs" / ("f" * 64 + ".incomplete")).write_bytes(b"part")
    _fake_cache_repo(hub, "MCG-NJU/videomae-base", {"model.safetensors": b"x"}, readme_license="cc-by-nc-4.0")
    code, lock = w.adopt([{"repo": "Qwen/Qwen3-VL-4B-Instruct"}, {"repo": "MCG-NJU/videomae-base"}],
                         weights, tmp_path / "l.json", log=lambda *_: None)
    assert code == 1
    by = {m["name"]: m for m in lock["models"]}
    assert "нет шардов" in by["Qwen/Qwen3-VL-4B-Instruct"]["status"]
    assert not by["MCG-NJU/videomae-base"]["license_ok"]
    info = w.inspect_snapshot(hub, "Qwen/Qwen3-VL-4B-Instruct")
    assert info["incomplete_blobs"] == 1
    assert w.inspect_snapshot(hub, "nobody/absent")["problems"]


def test_classify_local_and_free_text():
    assert w.classify_source("local: weights/scene/scene_signals.json — python -m ml.scene")[:3] == (
        "local", "", "weights/scene/scene_signals.json")
    assert w.classify_source("Qwen/Qwen3-4B-Instruct-2507 (GGUF Q4_K_M: конвертация)")[0] == "unknown"


def test_license_rows_match_adopted_by_repo_id(tmp_path):
    root = tmp_path / "repo"
    _models(root, "ml/detect", [{"name": "RT-DETRv2-S (r18vd), COCO", "hf_id_or_url": "PekingU/rtdetr_v2_r18vd",
                                 "license": "apache-2.0"}])
    lock = {"models": [
        {"name": "PekingU/rtdetr_v2_r18vd", "source": "PekingU/rtdetr_v2_r18vd", "repo_id": "PekingU/rtdetr_v2_r18vd",
         "revision": COMMIT, "license_hub": "apache-2.0", "license_ok": True, "license_note": "ok", "module": "ml/detect"},
        {"name": "ustc-community/dfine-medium-obj2coco", "source": "ustc-community/dfine-medium-obj2coco",
         "repo_id": "ustc-community/dfine-medium-obj2coco", "revision": COMMIT, "license_hub": "apache-2.0",
         "license_ok": True, "license_note": "ok", "module": "ml/detect"}]}
    rows = w.license_rows(w.load_specs(root), lock)
    assert len(rows) == 2 and all(r["license_ok"] for r in rows)
    md = w.render_licenses(rows)
    assert "красных флагов: 0" in md and "dfine-medium" in md
