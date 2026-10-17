from pathlib import Path

from deploy import deploy_config as dc
from deploy import envcheck as ec
from deploy import merge_requirements as mr

LOCK = """# L1 lock
torch==2.11.0+cu128
transformers==5.20.1
pillow==13.0.0
pyyaml==6.0.3
--extra-index-url https://download.pytorch.org/whl/cu128
"""


def test_parse_lock_keeps_local_labels():
    pins = ec.parse_lock(LOCK)
    assert pins == {"torch": "2.11.0+cu128", "transformers": "5.20.1", "pillow": "13.0.0", "pyyaml": "6.0.3"}


def test_check_blocks_critical_mismatch_and_missing_runtime():
    pins = ec.parse_lock(LOCK)
    installed = {"torch": "2.11.0", "transformers": "5.20.1", "pillow": "12.0.0", "pyyaml": "6.0.3"}
    rep = ec.check(pins, installed, ["torch", "transformers"], ["fastapi>=0.115", "pyyaml>=6"])
    assert not rep.ok
    assert any("torch" in b and "2.11.0+cu128" in b for b in rep.blocking)  # CPU-сборка вместо +cu128
    assert any("pillow" in w for w in rep.warnings)                          # не критичный — предупреждение
    assert rep.missing_runtime == ["fastapi>=0.115"]


def test_check_ok_when_exact():
    pins = ec.parse_lock(LOCK)
    rep = ec.check(pins, dict(pins), ["torch"], ["pyyaml>=6.0"])
    assert rep.ok and not rep.warnings and rep.checked == 4


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def test_merge_reports_only_violating_specs_against_env_lock(tmp_path):
    root = _tree(tmp_path, {
        "tools/dataprep/requirements-l1.lock.txt": LOCK,
        "ml/vlm/requirements.txt": "transformers>=4.57.1,<5\ntorch>=2.4\n",
        "ml/fusion/requirements.txt": "pillow>=10,<13\n",
        "eval/requirements.txt": "pyyaml==6.0.1\nfastapi>=0.115\n",
    })
    code, report, text = mr.run(root, "gpu", None)  # lock найден автоматически
    assert code == 1
    joined = "\n".join(report.conflicts)
    for name in ("transformers", "pillow", "pyyaml"):
        assert name in joined
    assert "torch" not in joined                       # >=2.4 совместимо с 2.11.0+cu128
    assert str(root / "tools/dataprep/requirements-l1.lock.txt") in report.constraints
    assert "fastapi>=0.115" in text                     # недостающее — добавляется, пины lock не трогаются
    code, report, _ = mr.run(root, "gpu", None, env_lock="none")
    assert not any("pillow" in c for c in report.conflicts)


def test_required_models_demoted_by_env():
    cfg = dc.load()
    req, opt = dc.required_models(cfg, environ={})
    assert "Qwen/Qwen3-VL-4B-Instruct" in [m["repo"] for m in req]
    req, opt = dc.required_models(cfg, environ={"SENTINEL_NO_VLM": "1"})
    assert "Qwen/Qwen3-VL-4B-Instruct" not in [m["repo"] for m in req]
    assert any(m.get("demoted_by_env") for m in opt)
    assert all("GGUF" not in m["repo"] for m in req + opt)


def test_markers_and_urls_do_not_mask_lock_conflicts(tmp_path):
    root = _tree(tmp_path, {
        "tools/dataprep/requirements-l1.lock.txt": "pillow==13.0.0\nnumpy==2.4.6\n",
        "ml/fusion/requirements.txt": 'Pillow>=10.0,<13 ; python_version >= "3.8"\n',   # маркер истинен на py3.11
        "ml/x/requirements.txt": "numpy @ https://example.invalid/numpy-1.26.0-cp311-none-any.whl\n",
        "ml/y/requirements.txt": 'pillow<12 ; sys_platform == "linux"\n',              # к Windows не относится
    })
    code, report, _ = mr.run(root, "gpu", None, check_only=True)
    joined = "\n".join(report.conflicts)
    assert code == 1
    assert "pillow" in joined and "ml/fusion" in joined and "ml/y" not in joined
    assert "numpy" in joined and "прямую ссылку" in joined
    code, report, _ = mr.run(root, "gpu", None, check_only=True, target="linux")
    assert "ml/y" in "\n".join(report.conflicts)


def test_marker_groups_checked_jointly_without_lock(tmp_path):
    root = _tree(tmp_path, {
        "a/requirements.txt": "pillow<13\n",
        "b/requirements.txt": 'pillow>=13 ; python_version >= "3.8"\n',
    })
    code, report, _ = mr.run(root, "gpu", None, check_only=True, env_lock="none")
    assert code == 1 and "pillow" in report.conflicts[0]
