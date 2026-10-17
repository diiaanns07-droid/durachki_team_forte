from pathlib import Path

from deploy import preflight as pf
from deploy.tests.test_weights import COMMIT, _fake_cache_repo

LOCK = "torch==2.11.0+cu128\ntransformers==5.20.1\npyyaml==6.0.3\n"


class FakeProbes(pf.Probes):
    def __init__(self, installed=None, torch=None, gpus=None, busy=False, tools=("ffmpeg", "ffprobe", "cloudflared")):
        self._installed = installed if installed is not None else {"torch": "2.11.0+cu128", "transformers": "5.20.1",
                                                                   "pyyaml": "6.0.3", "fastapi": "0.120.0"}
        self._torch = torch or {"installed": True, "version": "2.11.0+cu128", "cuda_build": "12.8",
                                "cuda_available": True, "device_count": 1}
        self._gpus = gpus if gpus is not None else [{"name": "RTX 4060 Laptop", "free_mib": "5427"}]
        self._busy, self._tools = busy, tools

    def installed_versions(self):
        return dict(self._installed)

    def torch_info(self):
        return dict(self._torch)

    def nvidia_smi(self):
        return self._gpus

    def port_in_use(self, host, port):
        return self._busy

    def which(self, name):
        return f"/usr/bin/{name}" if name in self._tools else None


def _setup(tmp_path: Path) -> tuple[dict, Path]:
    root = tmp_path / "repo"
    (root / "tools/dataprep").mkdir(parents=True)
    (root / "tools/dataprep/requirements-l1.lock.txt").write_text(LOCK)
    (root / "backend/api").mkdir(parents=True)
    (root / "backend/pipeline").mkdir(parents=True)
    (root / "backend/api/config.yaml").write_text("workers: 1\ngpu_slots: 1\nwarmup: true\nallow_mock: false\n"
                                                  "fake_pipeline: false\npipeline: backend.pipeline\n")
    (root / "backend/pipeline/config.yaml").write_text("mock: false\nexecutor:\n  mode: process\n")
    (root / "backend/api/requirements.txt").write_text("fastapi>=0.115\n")
    hub = root / "weights/hf/hub"
    _fake_cache_repo(hub, "PekingU/rtdetr_v2_r18vd", {"config.json": b"{}", "model.safetensors": b"\x01" * 16})
    cfg = {
        "env": {"python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "lock": "tools/dataprep/requirements-l1.lock.txt", "torch_backend": "cu128",
                "critical": ["torch", "transformers"]},
        "paths": {"weights": "weights", "hf_hub_cache": "weights/hf/hub", "data": "data", "cache": "cache",
                  "logs": "runs/deploy"},
        "models": {"required": [{"repo": "PekingU/rtdetr_v2_r18vd"},
                                {"repo": "Qwen/Qwen3-VL-4B-Instruct", "unless_env": ["SENTINEL_NO_VLM=1"]}],
                   "optional": [{"repo": "intfloat/multilingual-e5-small"}]},
        "preflight": {"min_free_disk_gb": 0, "min_free_vram_gb": 5.0, "configs": ["backend/api/config.yaml"],
                      "forbid_env": {"SENTINEL_FAKE_PIPELINE": "1", "SENTINEL_MOCK": "1"}},
        "service": {"host": "127.0.0.1", "port": 8000}, "tunnel": {"mode": "quick"},
    }
    return cfg, root


def _by(checks):
    return {c.name: c for c in checks}


def test_preflight_blocks_missing_required_model_and_passes_when_demoted(tmp_path):
    cfg, root = _setup(tmp_path)
    env = {"SENTINEL_API_KEYS": "k" * 24}
    checks = _by(pf.run_checks(cfg, root, env, FakeProbes()))
    assert checks["weights:PekingU/rtdetr_v2_r18vd"].status == "ok"
    assert COMMIT[:12] in checks["weights:PekingU/rtdetr_v2_r18vd"].detail
    assert checks["weights:Qwen/Qwen3-VL-4B-Instruct"].status == "fail"     # обязательная — нет в кэше
    assert checks["weights:intfloat/multilingual-e5-small (опц.)"].status == "warn"
    checks = _by(pf.run_checks(cfg, root, {**env, "SENTINEL_NO_VLM": "1"}, FakeProbes()))
    assert checks["weights:Qwen/Qwen3-VL-4B-Instruct (опц.)"].status == "warn"   # VLM выключен флагом
    assert all(c.status != "fail" for c in checks.values()), [c for c in checks.values() if c.status == "fail"]


def test_preflight_env_cuda_vram_keys_and_forbidden_modes(tmp_path):
    cfg, root = _setup(tmp_path)
    cfg["models"]["required"] = cfg["models"]["required"][:1]
    probes = FakeProbes(installed={"torch": "2.11.0", "transformers": "5.20.1", "pyyaml": "6.0.3"},
                        torch={"installed": True, "version": "2.11.0+cpu", "cuda_build": None,
                               "cuda_available": False, "device_count": 0},
                        gpus=[{"name": "RTX", "free_mib": "3000"}])
    env = {"SENTINEL_API_KEYS": "change-me", "SENTINEL_MOCK": "1"}
    c = _by(pf.run_checks(cfg, root, env, probes))
    assert c["env-lock"].status == "fail" and "torch" in c["env-lock"].detail and "fastapi" in c["env-lock"].detail
    assert c["cuda"].status == "fail" and c["vram"].status == "fail"
    assert c["api-key"].status == "fail" and "change-me" not in c["api-key"].detail.replace("(change-me)", "")
    assert c["env:SENTINEL_MOCK"].status == "fail"


def test_preflight_never_prints_key_and_port_busy(tmp_path):
    cfg, root = _setup(tmp_path)
    secret = "S3cretValue-ABCDEFGHIJKLMNOP"
    checks = pf.run_checks(cfg, root, {"SENTINEL_API_KEYS": secret}, FakeProbes(busy=True))
    assert all(secret not in c.detail for c in checks)
    c = _by(checks)
    assert c["port"].status == "fail"
    assert c["vram"].status == "ok"


def test_preflight_key_file_and_named_tunnel(tmp_path):
    cfg, root = _setup(tmp_path)
    (root / "data").mkdir()
    (root / "data/api_keys.txt").write_text("x" * 32)
    cfg["tunnel"]["mode"] = "named"
    c = _by(pf.run_checks(cfg, root, {}, FakeProbes(tools=("ffmpeg",))))
    assert c["api-key"].status == "ok"
    assert c["tunnel-token"].status == "fail" and c["tool:cloudflared"].status == "fail"
    assert c["tool:ffprobe"].status == "warn"


def _a02_a03(root: Path, api: str = "workers: 1\ngpu_slots: 1\nwarmup: true\nallow_mock: false\nfake_pipeline: false\n"
             "pipeline: backend.pipeline\nbudget_s: 270\nwatchdog_s: 285\njob_cap_s: 120\n",
             pipe: str = "mock: false\nexecutor:\n  mode: process\nwarmup:\n  deadline_s: 120\n") -> None:
    (root / "backend/api").mkdir(parents=True, exist_ok=True)
    (root / "backend/pipeline").mkdir(parents=True, exist_ok=True)
    (root / "backend/api/config.yaml").write_text(api)
    (root / "backend/pipeline/config.yaml").write_text(pipe)


def test_preflight_gates_single_worker_warmup_and_real_pipeline(tmp_path):
    cfg, root = _setup(tmp_path)
    cfg["models"]["required"] = cfg["models"]["required"][:1]
    _a02_a03(root)
    env = {"SENTINEL_API_KEYS": "k" * 24}
    c = _by(pf.run_checks(cfg, root, env, FakeProbes()))
    for name in ("a02:single-worker", "a02:warmup", "a02:real-pipeline", "a03:executor", "a03:mock"):
        assert c[name].status == "ok", (name, c[name].detail)
    assert c["budgets"].status == "warn" and "UNMEASURED" in c["budgets"].detail and "budget_s=270" in c["budgets"].detail
    bad = {**env, "SENTINEL_WORKERS": "2", "SENTINEL_WARMUP": "0", "SENTINEL_FAKE_PIPELINE": "1", "SENTINEL_EXECUTOR": "inline"}
    c = _by(pf.run_checks(cfg, root, bad, FakeProbes()))
    for name in ("a02:single-worker", "a02:warmup", "a02:real-pipeline", "a03:executor"):
        assert c[name].status == "fail", name
    over = root / "over.yaml"
    over.write_text("executor:\n  mode: inline\n")
    c = _by(pf.run_checks(cfg, root, {**env, "SENTINEL_PIPELINE_CONFIG": str(over)}, FakeProbes()))
    assert c["a03:executor"].status == "fail"
    cfg["service"]["budgets_measured_on_gpu"] = True
    assert _by(pf.run_checks(cfg, root, env, FakeProbes()))["budgets"].status == "ok"


def test_preflight_fails_without_backend_configs(tmp_path):
    cfg, root = _setup(tmp_path)
    (root / "backend/api/config.yaml").unlink()
    (root / "backend/pipeline/config.yaml").unlink()
    c = _by(pf.run_checks(cfg, root, {"SENTINEL_API_KEYS": "k" * 24}, FakeProbes()))
    assert c["a02:config"].status == "fail" and c["a03:config"].status == "fail"
