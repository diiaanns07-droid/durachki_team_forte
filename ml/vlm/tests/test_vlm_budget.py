from ml.vlm.backends.mock import MockBackend
from ml.vlm.budget import Budget, FakeClock, VLMLimits, visual_tokens
from ml.vlm.config import load_config
from ml.vlm.runtime import call_json, fit_images
from ml.vlm.parsing import normalize_verify


def _imgs(n=5, size=(245, 448)):
    from PIL import Image
    return [Image.new("RGB", size) for _ in range(n)]


def test_visual_tokens_estimate():
    assert visual_tokens((448, 448), 28) == 256
    assert visual_tokens((245, 448), 32) == 8 * 14


def test_limits_from_cfg_and_overrides():
    cfg = load_config(use_env=False)
    lim = VLMLimits.from_cfg(cfg, "verify", {"max_calls": 2})
    assert lim.max_calls == 2 and lim.max_frames_per_call == 6 and lim.max_retries_per_item == 1
    assert VLMLimits.from_cfg(cfg, "discover").max_calls == 4


def test_budget_time_gate_and_ema():
    clk = FakeClock()
    b = Budget(10.0, VLMLimits(est_call_s=2.0, safety=1.0), clock=clk)
    assert b.can_call()
    clk.advance(8.5)
    assert not b.can_call() and b.stop_reason == "time"
    b2 = Budget(10.0, VLMLimits(est_call_s=1.0, safety=1.0), clock=FakeClock())
    b2.record(3.0)
    assert b2.est_call_s() >= 3.0


def test_budget_count_limits():
    b = Budget(100.0, VLMLimits(max_calls=1, est_call_s=0.1), clock=FakeClock())
    assert b.can_call()
    b.record(0.1)
    assert not b.can_call() and b.stop_reason == "max_calls"
    b = Budget(100.0, VLMLimits(max_visual_tokens_total=100, est_call_s=0.1), clock=FakeClock())
    assert not b.can_call(vis_tokens=101) and b.stop_reason == "max_visual_tokens_total"
    b = Budget(100.0, VLMLimits(max_output_tokens_total=50, max_output_tokens_per_call=40, est_call_s=0.1),
               clock=FakeClock())
    assert b.output_tokens_for_call(220) == 40
    b.record(0.1, out_tokens=30)
    assert b.output_tokens_for_call(220) == 20


def test_call_json_retry_once_on_garbage():
    clk = FakeClock()
    be = MockBackend(responses=["oops not json", '{"visible": true, "type_id": "V05", "confidence": 0.9}'],
                     latency_s=1.0, clock=clk)
    b = Budget(30.0, VLMLimits(est_call_s=1.0), clock=clk)
    out = call_json(be, _imgs(), "sys", "user", b, normalize=lambda d: normalize_verify(d), max_new_tokens=100,
                    retry_suffix="ONLY JSON")
    assert out.ok and out.attempts == 2 and b.retries == 1
    assert be.calls[1]["prompt"].endswith("ONLY JSON")


def test_call_json_retry_respects_total_retry_limit():
    clk = FakeClock()
    be = MockBackend(responses=["bad", "bad again"], latency_s=1.0, clock=clk)
    b = Budget(30.0, VLMLimits(est_call_s=1.0, max_retries_total=0), clock=clk)
    out = call_json(be, _imgs(), "sys", "user", b, normalize=lambda d: normalize_verify(d), max_new_tokens=100)
    assert not out.ok and out.status == "parse_failed" and out.attempts == 1


def test_call_json_timeout_not_retried():
    clk = FakeClock()
    be = MockBackend(latency_s=50.0, clock=clk)
    b = Budget(30.0, VLMLimits(est_call_s=1.0, call_timeout_s=5.0), clock=clk)
    out = call_json(be, _imgs(), "sys", "user", b, normalize=lambda d: normalize_verify(d), max_new_tokens=100)
    assert out.status == "timeout" and out.attempts == 1
    assert clk() <= 5.0 + 1e-6  # вызов ограничен call_timeout_s


def test_call_json_skipped_when_budget_exhausted():
    be = MockBackend()
    b = Budget(0.1, VLMLimits(est_call_s=1.0), clock=FakeClock())
    out = call_json(be, _imgs(), "sys", "user", b, normalize=lambda d: normalize_verify(d), max_new_tokens=100)
    assert out.status == "skipped_budget" and not be.calls


def test_fit_images_caps_frames_and_tokens():
    imgs = _imgs(8, (448, 448))
    out = fit_images(imgs, max_frames=4, max_vis_tokens=400, px_per_token=28, keep_index=7)
    assert len(out) == 4
    assert sum(visual_tokens(i.size, 28) for i in out) <= 400


class _SlowLoad(MockBackend):
    """Бэкенд с «холодной» загрузкой: load() двигает часы, может падать."""

    def __init__(self, clock, load_s=30.0, fail=False):
        super().__init__(mode="confirm", clock=clock)
        self._loaded = False
        self.load_s, self.fail, self.load_calls = load_s, fail, 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self.clock.advance(self.load_s)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self._loaded = True


def test_cold_load_counts_against_budget(clip, cfg, clock):
    from ml.vlm.verifier import verify_events
    from ml.vlm._contract import Event
    be = _SlowLoad(clock, load_s=30.0)
    cfg["runtime"]["lazy_load_min_budget_s"] = 10.0
    out = verify_events(clip, [Event("V05", 1, 2, 1.5, 0.6, "rules")], 32.0, cfg=cfg, backend=be, clock=clock,
                        limits={"est_call_s": 3.0, "safety": 1.0})
    assert be.load_calls == 1 and out[0].extra["vlm"]["status"] == "skipped_budget"  # 2 с после загрузки мало


def test_cold_load_skipped_when_budget_small(clip, cfg, clock):
    from ml.vlm.verifier import verify_events
    from ml.vlm._contract import Event
    be = _SlowLoad(clock)
    cfg["runtime"]["lazy_load_min_budget_s"] = 45.0
    out = verify_events(clip, [Event("V05", 1, 2, 1.5, 0.6, "rules")], 20.0, cfg=cfg, backend=be, clock=clock)
    assert be.load_calls == 0 and out[0].extra["vlm"]["status"] == "unavailable"
    assert "not warm" in out[0].extra["vlm"]["reason"]


def test_failed_load_has_cooldown(clip, cfg, clock):
    from ml.vlm.verifier import verify_events
    from ml.vlm._contract import Event
    be = _SlowLoad(clock, load_s=1.0, fail=True)
    ev = [Event("V05", 1, 2, 1.5, 0.6, "rules")]
    for _ in range(3):
        out = verify_events(clip, ev, 100.0, cfg=cfg, backend=be, clock=clock)
        assert out[0].extra["vlm"]["status"] == "unavailable" and out[0].confidence == 0.6
    assert be.load_calls == 1  # повторно не грузим в течение load_retry_cooldown_s


def test_regex_fallback_only_after_retry():
    from ml.vlm.parsing import regex_fallback_verify
    clk = FakeClock()
    good = '{"visible": true, "type_id": "V05", "observed_facts": ["Кассир держит телефон"], "confidence": 0.9}'
    be = MockBackend(responses=["visible: yes, confidence 0.7 (broken", good], clock=clk)
    b = Budget(30.0, VLMLimits(est_call_s=0.5), clock=clk)
    out = call_json(be, _imgs(), "s", "u", b, normalize=lambda d: normalize_verify(d), max_new_tokens=100,
                    fallback=regex_fallback_verify)
    assert out.ok and out.attempts == 2 and out.data["observed_facts"]  # ретрай дал полноценный JSON
    be = MockBackend(responses=["visible: yes, confidence 0.7 (broken", "still broken"], clock=clk)
    out = call_json(be, _imgs(), "s", "u", Budget(30.0, VLMLimits(est_call_s=0.5), clock=clk),
                    normalize=lambda d: normalize_verify(d), max_new_tokens=100, fallback=regex_fallback_verify)
    assert out.ok and out.attempts == 2 and "regex_fallback" in out.data["flags"]


def test_partial_answer_on_timeout_not_retried():
    from ml.vlm.backends.base import GenResult

    class Partial(MockBackend):
        def generate(self, images, system, prompt, **kw):
            self.calls.append(prompt)
            return GenResult(text='{"type_id": "V05", "observ', latency_s=5.0, timed_out=True)

    be = Partial()
    out = call_json(be, _imgs(), "s", "u", Budget(60.0, VLMLimits(est_call_s=0.5), clock=FakeClock()),
                    normalize=lambda d: normalize_verify(d), max_new_tokens=100)
    assert out.status == "timeout" and len(be.calls) == 1


def test_budget_rechecked_after_gpu_wait():
    clk = FakeClock()

    class SlowLock:
        def __enter__(self):
            clk.advance(29.0)  # другой поток держал GPU 29 с

        def __exit__(self, *a):
            return False

    from ml.vlm import runtime
    runtime.set_gpu_lock(SlowLock())
    be = MockBackend(clock=clk)
    out = call_json(be, _imgs(), "s", "u", Budget(30.0, VLMLimits(est_call_s=2.0), clock=clk),
                    normalize=lambda d: normalize_verify(d), max_new_tokens=100)
    assert out.status == "skipped_budget" and not be.calls
