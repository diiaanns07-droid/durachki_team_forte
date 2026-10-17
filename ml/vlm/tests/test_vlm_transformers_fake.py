"""transformers-бэкенд на поддельных torch/модели/процессоре (numpy): проверяем НАШУ логику — сборку входов,
finish_reason, тайминги, softmax по вариантам, переход на буквы при совпадении первых токенов, чат.
Реальная модель/скорость здесь не проверяются (это делает L2 через python -m ml.vlm.bench)."""
import contextlib
import sys
import types

import numpy as np
import pytest

from ml.vlm.backends.transformers_backend import TransformersBackend


class T(np.ndarray):
    def float(self):
        return np.asarray(self, dtype=np.float64).view(T)


def t(a):
    return np.asarray(a).view(T)


def _fake_torch():
    m = types.ModuleType("torch")

    class OOM(RuntimeError):
        pass

    m.cuda = types.SimpleNamespace(is_available=lambda: False, reset_peak_memory_stats=lambda: None,
                                   synchronize=lambda: None, empty_cache=lambda: None, OutOfMemoryError=OOM,
                                   max_memory_allocated=lambda: 0, mem_get_info=lambda: (0, 0))
    m.inference_mode = contextlib.nullcontext

    def softmax(x, dim=-1):
        e = np.exp(np.asarray(x) - np.max(x))
        return (e / e.sum()).view(T)

    m.softmax = softmax
    return m


class Batch(dict):
    def to(self, device):
        return self


VOCAB = {"Да": 5, "да": 5, " Да": 6, " да": 6, "Нет": 7, "нет": 7, " Нет": 8, " нет": 8, "Не": 9, " Не": 9,
         "A": 20, " A": 20, "B": 21, " B": 21, "C": 22, " C": 22}


class Tok:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        if text in VOCAB:
            return [VOCAB[text]]
        head = text.split(" ")[0] if not text.startswith(" ") else " " + text.split(" ")[1]
        return [VOCAB.get(head, 99), 100]  # «Не видно» -> [9, 100]


class Proc:
    tokenizer = Tok()

    def __init__(self, fail_tokenize=False):
        self.fail_tokenize = fail_tokenize
        self.seen = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_dict=False,
                            return_tensors=None):
        self.seen.append(messages)
        if tokenize and self.fail_tokenize:
            raise TypeError("old transformers")
        if not tokenize:
            return "TEMPLATE"
        return Batch(input_ids=t(np.zeros((1, 50), dtype=np.int64)))

    def __call__(self, text=None, images=None, return_tensors=None, padding=True):
        return Batch(input_ids=t(np.zeros((1, 40), dtype=np.int64)))

    def decode(self, ids, skip_special_tokens=True):
        return "{" + ",".join(str(int(i)) for i in ids) + "}"


class Model:
    device = "cpu"
    generation_config = types.SimpleNamespace(eos_token_id=[2, 3])

    def __init__(self, gen_tail, logits=None):
        self.gen_tail = gen_tail
        self.logits = logits
        self.kw = None

    def generate(self, **kw):
        self.kw = kw
        n_in = kw["input_ids"].shape[1]
        return t(np.concatenate([np.zeros(n_in, dtype=np.int64), np.array(self.gen_tail)])[None, :])

    def __call__(self, **kw):
        self.kw = kw
        return types.SimpleNamespace(logits=t(np.asarray(self.logits, dtype=np.float64)[None, None, :]))


@pytest.fixture
def be(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    b = TransformersBackend(model="Qwen/Qwen3-VL-4B-Instruct", quant="bnb4", cache_dir="/nonexistent")
    b._processor = Proc()
    return b


def test_generate_eos_complete_and_timings(be):
    be._model = Model([11, 12, 2])
    r = be.generate([], "sys", "prompt", max_new_tokens=10, timeout_s=5)
    assert r.finish_reason == "eos" and not r.truncated and r.completion_tokens == 3 and r.prompt_tokens == 50
    assert set(r.meta["timings"]) == {"prep_s", "gen_s", "decode_s"}
    assert be._model.kw["do_sample"] is False and be._model.kw["max_new_tokens"] == 10
    assert be._model.kw["max_time"] == 5
    msgs = be._processor.seen[-1]
    assert msgs[0] == {"role": "system", "content": "sys"} and msgs[1]["content"][-1] == {"type": "text", "text": "prompt"}


def test_generate_length_truncated(be):
    be._model = Model([11, 12, 13, 14])
    r = be.generate([], "", "p", max_new_tokens=4, timeout_s=5)
    assert r.finish_reason == "length" and r.truncated


def test_images_go_inside_messages(be):
    from PIL import Image
    be._model = Model([2])
    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    be.generate(imgs, "s", "p", max_new_tokens=4)
    content = be._processor.seen[-1][1]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"] and content[0]["image"] is imgs[0]


def test_fallback_input_path_for_old_transformers(be):
    from PIL import Image
    be._processor = Proc(fail_tokenize=True)
    be._model = Model([2])
    r = be.generate([Image.new("RGB", (8, 8))], "s", "p", max_new_tokens=4)
    assert r.prompt_tokens == 40  # путь text + processor(text, images)


def test_option_probs_words(be):
    logits = np.full(120, -10.0)
    logits[5], logits[6], logits[7], logits[9] = 3.0, 1.0, 1.0, 0.0  # Да/ Да/Нет/Не(видно)
    be._model = Model([], logits=logits)
    sc = be.option_probs([], "s", "Вопрос?", {"yes": ["Да"], "no": ["Нет"], "unknown": ["Не видно"]})
    assert sc.mode == "words" and abs(sum(sc.probs.values()) - 1) < 1e-9
    assert sc.probs["yes"] > sc.probs["no"] > 0 and 0 < sc.mass <= 1
    assert be._model.kw.get("logits_to_keep") == 1


def test_option_probs_letters_when_first_tokens_collide(be):
    logits = np.full(120, -10.0)
    logits[21] = 4.0  # «B»
    be._model = Model([], logits=logits)
    sc = be.option_probs([], "s", "Вопрос?", {"yes": ["Не знаю"], "no": ["Нет"], "unknown": ["Не видно"]})
    assert sc.mode == "letters" and max(sc.probs, key=sc.probs.get) == "no"
    assert "Вместо слова ответь одной буквой" in be._processor.seen[-1][-1]["content"][-1]["text"]


def test_chat_converts_openai_messages(be):
    from PIL import Image
    be._model = Model([7, 2])
    r = be.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                 {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}],
                images=[Image.new("RGB", (8, 8))], max_new_tokens=8)
    conv = be._processor.seen[-1]
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "user"]
    assert conv[3]["content"][0]["type"] == "image" and conv[1]["content"] == [{"type": "text", "text": "U1"}]
    assert conv[0] == {"role": "system", "content": "S"} and conv[2] == {"role": "assistant", "content": "A1"}
    assert r.finish_reason == "eos"


def test_oom_returns_error_not_raise(be):
    class OOMModel(Model):
        def generate(self, **kw):
            raise sys.modules["torch"].cuda.OutOfMemoryError("oom")

    be._model = OOMModel([])
    r = be.generate([], "s", "p", max_new_tokens=4)
    assert r.error == "cuda_oom" and r.text == ""


# ---------------------------------------------------------------- verify на поддельном transformers-адаптере
# Полный путь: verify_events -> call_json -> TransformersBackend._run (поддельные torch/модель/процессор).

class TextProc(Proc):
    """decode возвращает заданный текст ответа (как настоящий процессор декодирует токены)."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text


def _verify_with_fake(be, clip, cfg, conf=0.5):
    from ml.vlm._contract import Event
    from ml.vlm.verifier import verify_events
    ev = Event("V05", 4.0, 6.0, 5.0, conf, "rules", [], {})
    return verify_events(clip, [ev], 60.0, cfg=cfg, backend=be)[0]


def test_fake_tf_truncated_json_gives_unsure(be, clip, cfg):
    be._processor = TextProc('{"visible": true, "type_id": "V05", "confidence": 0.9, "observed_facts": ["Кассир')
    be._model = Model(list(range(10, 10 + 180)))  # 180 токенов без EOS = max_new_tokens_verify -> length
    out = _verify_with_fake(be, clip, cfg)
    v = out.extra["vlm"]
    assert v["status"] == "truncated" and v["verdict"] == "unsure" and out.confidence == 0.5


def test_fake_tf_deadline_gives_unsure_without_retry(be, clip, cfg, monkeypatch):
    import time as _time

    class Slow(Model):
        def generate(self, **kw):
            _time.sleep(kw["max_time"] + 0.05)  # упёрлись в max_time
            return super().generate(**kw)

    be._processor = TextProc('{"visible": true, "type_id": "V05", "confidence": 0.9}')
    be._model = Slow([11, 12, 13])  # без EOS, меньше max_new_tokens -> finish=time
    cfg["generation"]["call_timeout_s"] = 0.5
    out = _verify_with_fake(be, clip, cfg)
    v = out.extra["vlm"]
    assert v["status"] == "timeout" and v["verdict"] == "unsure"
    assert v["attempts"] == 1 and out.confidence == 0.5  # время не ретраим


def test_fake_tf_oom_gives_error_unsure(be, clip, cfg):
    class OOMModel(Model):
        def generate(self, **kw):
            raise sys.modules["torch"].cuda.OutOfMemoryError("oom")

    be._model = OOMModel([])
    out = _verify_with_fake(be, clip, cfg, conf=0.7)
    v = out.extra["vlm"]
    assert v["status"] == "error" and v["error"] == "cuda_oom" and v["verdict"] == "unsure"
    assert out.confidence == 0.7  # ни подтверждения, ни отказа


def test_fake_tf_complete_json_confirms(be, clip, cfg):
    import json as _json
    be._processor = TextProc(_json.dumps({"visible": True, "type_id": "V05", "observed_facts": [],
                                          "confidence": 0.9, "best_frame": 2, "visible_frames": [2],
                                          "uncertain_because": ""}))
    be._model = Model([11, 12, 2])  # EOS -> complete
    out = _verify_with_fake(be, clip, cfg)
    v = out.extra["vlm"]
    assert v["status"] == "ok" and v["finish_reason"] == "eos" and v["complete"] is True and v["verdict"] == "confirm"
