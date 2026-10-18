import json
import subprocess
from pathlib import Path

from deploy import release as rl


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "deploy/config").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(root))
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("runs/\ndata/\ndeploy/.env\n")
    (root / "deploy/config/deploy.yaml").write_text("a: 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v1")
    return root


def test_mark_verified_diff_and_rollback_keep_work(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_LOG_DIR", raising=False)
    monkeypatch.delenv("SENTINEL_PIPELINE_CONFIG", raising=False)
    root = _repo(tmp_path)
    (root / "deploy/.env").write_text("SENTINEL_API_KEYS=supersecretvalue\n")
    rec = rl.mark_verified(root, note="A16 ok")
    v1 = rec["sha"]
    saved = (root / "runs/deploy/verified/latest.json").read_text()
    assert "supersecretvalue" not in saved and "deploy/.env" in json.loads(saved)["configs"]  # только хэш
    (root / "deploy/config/deploy.yaml").write_text("a: 2\n")
    _git(root, "commit", "-qam", "v2")
    (root / "deploy/config/deploy.yaml").write_text("a: 3  # незакоммиченная правка\n")
    (root / "data").mkdir()
    (root / "data/job.json").write_text("{}")  # данные вне git
    d = rl.diff(root, rl.load_record(root))
    assert d["verified_sha"] == v1 and d["commits_since"] == "1" and "deploy/config/deploy.yaml" in d["changed_configs"]
    plan = rl.rollback(root, rl.load_record(root), dry_run=True)
    assert plan["dry_run"] and any("stash" in s for s in plan["steps"])
    out = rl.rollback(root, rl.load_record(root))
    assert _git(root, "rev-parse", "HEAD") == v1 and out["config_mismatch"] == []
    assert "rollback-to-" in _git(root, "stash", "list")           # правка не потеряна
    assert (root / "data/job.json").is_file()                       # данные не тронуты
    assert (root / "deploy/config/deploy.yaml").read_text() == "a: 1\n"


def test_rollback_without_record(tmp_path, monkeypatch):
    monkeypatch.delenv("SENTINEL_LOG_DIR", raising=False)
    root = _repo(tmp_path)
    assert rl.main(["rollback", "--root", str(root)]) == 1
