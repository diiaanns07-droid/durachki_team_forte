"""transformers-бэкенд: Qwen3-VL-4B/8B-Instruct (bnb 4-bit) или Qwen2.5-VL-7B-Instruct-AWQ. Apache-2.0.

Тяжёлые импорты (torch/transformers) — только внутри методов. Модель грузится лениво и держится тёплой;
unload() освобождает VRAM (оркестратор может выгружать VLM между видео при нехватке памяти).
"""
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import threading
import time
from typing import Any, Optional

from .base import BackendUnavailable, GenResult, OptionScores, VLMBackend

log = logging.getLogger(__name__)


def _hf_offline() -> bool:
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
    """HF hub-кэш: явный параметр > env SENTINEL_HF_CACHE > <repo>/weights/hf/hub (раскладка L1/A17), если есть.
    None — пусть huggingface_hub решает сам (HF_HUB_CACHE / HF_HOME)."""
    for cand in (cache_dir, os.environ.get("SENTINEL_HF_CACHE")):
        if cand:
            return str(cand)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for cand in (os.path.join(os.getcwd(), "weights", "hf", "hub"), os.path.join(repo_root, "weights", "hf", "hub")):
        if os.path.isdir(cand):
            return cand
    return None


def read_model_license(model_id: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """license из YAML front-matter README.md локальной папки или снапшота в HF-кэше (без сети)."""
    readme: Optional[str] = None
    if os.path.isdir(model_id):
        cand = os.path.join(model_id, "README.md")
        readme = cand if os.path.exists(cand) else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            r = try_to_load_from_cache(model_id, "README.md", cache_dir=cache_dir)
            readme = r if isinstance(r, str) and os.path.exists(r) else None
        except Exception:  # noqa: BLE001
            readme = None
    if not readme:
        return None
    with open(readme, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(8192)
    if not text.startswith("---"):
        return None
    fm = text.split("---", 2)[1] if text.count("---") >= 2 else ""
    for line in fm.splitlines():
        if line.strip().lower().startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"").lower() or None
    return None


def check_license(model_id: str, mode: str = "warn", allowed: Optional[list[str]] = None,
                  cache_dir: Optional[str] = None) -> tuple[bool, str]:
    """strict: лицензия должна быть найдена и разрешена; warn: запрещённая — блок, не найдена — предупреждение."""
    mode = (mode or "warn").lower()
    if mode == "off":
        return True, "license check off"
    allowed_set = {a.lower() for a in (allowed or ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"])}
    lic = read_model_license(model_id, cache_dir)
    if lic is None:
        msg = f"license of {model_id} not found locally (README.md front-matter)"
        return (mode != "strict"), msg
    if lic not in allowed_set:
        return False, f"license '{lic}' of {model_id} is not in allowed {sorted(allowed_set)}"
    return True, f"license {lic}"


class TransformersBackend(VLMBackend):
    """Qwen3-VL через transformers (проверено L1 на 5.20.1): AutoModelForImageTextToText + AutoProcessor,
    nf4 (LM) + bf16 (vision), картинки внутри сообщений, apply_chat_template(tokenize=True, return_dict=True)."""

    name = "transformers"
    supports_option_probs = True

    def __init__(self, model: str, quant: str = "bnb4", device: str = "cuda:0", allow_cpu: bool = False,
                 dtype: str = "bfloat16", attn_implementation: Optional[str] = "sdpa",
                 max_pixels: Optional[int] = None, min_pixels: Optional[int] = None,
                 skip_quant_modules: Optional[list[str]] = None, local_files_only: Optional[bool] = None,
                 license_guard: str = "warn", allowed_licenses: Optional[list[str]] = None,
                 min_free_vram_mb: int = 0, cache_dir: Optional[str] = None, **_: Any) -> None:
        self.model_id = model
        self.min_free_vram_mb = int(min_free_vram_mb or 0)
        self.license_guard = license_guard
        self.allowed_licenses = list(allowed_licenses or ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"])
        self.license_info = ""
        self.quant = (quant or "none").lower()
        self.device = "cuda:0" if device == "cuda" else device
        self.allow_cpu = allow_cpu
        self.dtype_name = dtype
        self.attn = attn_implementation
        self.max_pixels = int(max_pixels) if max_pixels else None   # None: не трогаем процессор (кадры ужаты сами)
        self.min_pixels = int(min_pixels) if min_pixels else None
        self.skip_modules = list(skip_quant_modules or ["visual", "lm_head"])
        self.local_files_only = _hf_offline() if local_files_only is None else bool(local_files_only)
        self.cache_dir = resolve_cache_dir(cache_dir)
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self._option_ids: dict[tuple[tuple[str, tuple[str, ...]], ...], tuple[str, dict[str, list[int]]]] = {}
        self.load_s: float = 0.0
        self.vram_after_load_mb: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle
    def available(self) -> tuple[bool, str]:
        for mod in ("torch", "transformers", "PIL"):
            if importlib.util.find_spec(mod) is None:
                return False, f"python package '{mod}' not installed"
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available() and not self.allow_cpu:
            return False, "CUDA not available (transformers.allow_cpu=false)"
        if self.quant.startswith("bnb") and importlib.util.find_spec("bitsandbytes") is None:
            return False, "bitsandbytes not installed (quant=bnb4)"
        if self.quant == "awq" and importlib.util.find_spec("awq") is None and importlib.util.find_spec("gptqmodel") is None:
            return False, "autoawq/gptqmodel not installed (quant=awq)"
        return True, ""

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _device(self) -> str:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def _hf_kw(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.cache_dir:
            kw["cache_dir"] = self.cache_dir
        return kw

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            ok, why = self.available()
            if not ok:
                raise BackendUnavailable(why)
            lic_ok, self.license_info = check_license(self.model_id, self.license_guard, self.allowed_licenses,
                                                      self.cache_dir)
            if not lic_ok:
                raise BackendUnavailable(self.license_info)
            if self.license_info.startswith("license of"):
                log.warning("%s", self.license_info)
            import torch
            import transformers
            from transformers import AutoProcessor

            t0 = time.perf_counter()
            dev = self._device()
            free_mb = _free_vram_mb() if dev.startswith("cuda") else None
            if free_mb is not None and self.min_free_vram_mb and free_mb < self.min_free_vram_mb:
                # не грузим, чтобы не выбить OOM у детектора/энкодеров: A03 может вызвать unload() у других
                raise BackendUnavailable(f"free VRAM {free_mb:.0f} MB < min_free_vram_mb {self.min_free_vram_mb}")
            dtype = getattr(torch, self.dtype_name) if dev != "cpu" else torch.float32
            kw: dict[str, Any] = {"device_map": dev, **self._hf_kw()}
            if self.attn:
                kw["attn_implementation"] = self.attn
            ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
            kw["dtype" if ver >= (4, 56) else "torch_dtype"] = dtype
            if self.quant in ("bnb4", "bnb8") and dev != "cpu":
                from transformers import BitsAndBytesConfig

                if self.quant == "bnb4":
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=self.skip_modules)
                else:
                    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                                    llm_int8_skip_modules=self.skip_modules)
            self._model = self._load_model(kw)
            self._model.eval()
            pkw = dict(self._hf_kw())
            if self.max_pixels:
                pkw["max_pixels"] = self.max_pixels
            if self.min_pixels:
                pkw["min_pixels"] = self.min_pixels
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_id, **pkw)
            except (TypeError, ValueError):
                self._processor = AutoProcessor.from_pretrained(self.model_id, **self._hf_kw())
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
                self.vram_after_load_mb = round(torch.cuda.memory_allocated() / 2**20, 1)
            self.load_s = time.perf_counter() - t0
            log.info("VLM %s loaded in %.1fs (quant=%s, device=%s)", self.model_id, self.load_s, self.quant, dev)

    def _load_model(self, kw: dict[str, Any]) -> Any:
        import transformers

        errors = []
        for cls_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration",
                         "Qwen2_5_VLForConditionalGeneration", "AutoModelForVision2Seq"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(self.model_id, **kw)
            except (ValueError, KeyError) as e:  # класс не подходит к конфигу — пробуем следующий
                errors.append(f"{cls_name}: {e}")
        raise BackendUnavailable("cannot load %s: %s" % (self.model_id, " | ".join(errors) or "no VLM class in transformers"))

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._option_ids.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ------------------------------------------------------------------ входы
    @staticmethod
    def _messages(images: list[Any], system: str, prompt: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs: list[dict[str, Any]] = []
        if system:  # строкой: так шаблоны Qwen-VL принимают system гарантированно
            msgs.append({"role": "system", "content": str(system)})
        msgs.append({"role": "user", "content": content})
        return msgs

    def _inputs(self, messages: list[dict[str, Any]]) -> Any:
        """Путь, проверенный L1 на transformers 5.20.1; запасной — текст шаблона + processor(text, images)."""
        try:
            inp = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      return_dict=True, return_tensors="pt")
        except Exception as e:  # noqa: BLE001 — старые версии / ошибки шаблона: запасной путь text + processor
            log.debug("apply_chat_template(tokenize=True) failed (%s), fallback path", type(e).__name__)
            images = [c["image"] for m in messages if isinstance(m.get("content"), list) for c in m["content"]
                      if isinstance(c, dict) and c.get("type") == "image"]
            plain = [m if not isinstance(m.get("content"), list) else
                     {**m, "content": [({"type": "image"} if isinstance(c, dict) and c.get("type") == "image" else c)
                                       for c in m["content"]]} for m in messages]
            text = self._processor.apply_chat_template(plain, tokenize=False, add_generation_prompt=True)
            inp = self._processor(text=[text], images=images or None, return_tensors="pt", padding=True)
        return inp.to(self._model.device)

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for src in (getattr(self._model, "generation_config", None), getattr(self._processor, "tokenizer", None)):
            v = getattr(src, "eos_token_id", None) if src is not None else None
            if isinstance(v, int):
                ids.add(v)
            elif isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
        return ids

    # ------------------------------------------------------------------ inference
    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        return self._run(self._messages(images, system, prompt), max_new_tokens, timeout_s, temperature)

    def chat(self, messages: list[dict[str, Any]], images: Optional[list[Any]] = None, max_new_tokens: int = 256,
             timeout_s: float = 20.0, temperature: float = 0.0) -> GenResult:
        conv: list[dict[str, Any]] = []
        imgs = list(images or [])
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        for i, m in enumerate(messages):
            role = str(m.get("role", "user"))
            text = m.get("content", "")
            if isinstance(text, list):  # уже в формате контента
                conv.append({"role": role, "content": text})
                continue
            if role != "user":  # system/assistant — строкой
                conv.append({"role": role, "content": str(text)})
                continue
            content: list[dict[str, Any]] = []
            if i == last_user and imgs:
                content += [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": str(text)})
            conv.append({"role": role, "content": content})
        return self._run(conv, max_new_tokens, timeout_s, temperature)

    def _run(self, messages: list[dict[str, Any]], max_new_tokens: int, timeout_s: float,
             temperature: float) -> GenResult:
        if self._model is None:
            self.load()
        import torch

        with self._lock:
            t0 = time.perf_counter()
            cuda = str(self._model.device).startswith("cuda")
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            inputs = self._inputs(messages)
            t1 = time.perf_counter()
            gen_kw: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "max_time": max(0.5, float(timeout_s))}
            if temperature and temperature > 0:
                gen_kw.update(do_sample=True, temperature=float(temperature))
            else:
                gen_kw.update(do_sample=False)
            try:
                with torch.inference_mode():
                    out = self._model.generate(**inputs, **gen_kw)
                if cuda:
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                del inputs
                torch.cuda.empty_cache()
                log.warning("VLM generate: CUDA OOM (уменьшите frames.max_side / max_visual_tokens_per_call)")
                return GenResult(text="", latency_s=time.perf_counter() - t0, error="cuda_oom")
            t2 = time.perf_counter()
            n_in = int(inputs["input_ids"].shape[1])
            gen_ids = out[0, n_in:]
            txt = self._processor.decode(gen_ids, skip_special_tokens=True)
            t3 = time.perf_counter()
            n_out = int(gen_ids.shape[0])
            last = int(gen_ids[-1]) if n_out else -1
            if n_out and last in self._eos_ids():
                finish = "eos"
            elif n_out >= int(max_new_tokens):
                finish = "length"
            elif (t2 - t1) >= 0.98 * gen_kw["max_time"]:
                finish = "time"
            else:
                finish = "stop"
            return GenResult(text=txt, latency_s=t3 - t0, prompt_tokens=n_in, completion_tokens=n_out,
                             timed_out=finish == "time", finish_reason=finish,
                             meta={"timings": {"prep_s": round(t1 - t0, 4), "gen_s": round(t2 - t1, 4),
                                               "decode_s": round(t3 - t2, 4)},
                                   "vram_peak_mb": _vram_peak_mb()})

    def _option_token_ids(self, options: dict[str, list[str]]) -> tuple[str, dict[str, list[int]]]:
        """Первые токены вариантов (с вариантами регистра/пробела). Совпадение первых токенов у разных
        вариантов -> режим letters (A/B/C…), иначе вероятности были бы неразличимы."""
        key = tuple((k, tuple(v)) for k, v in options.items())
        if key in self._option_ids:
            return self._option_ids[key]
        tok = self._processor.tokenizer
        ids: dict[str, list[int]] = {}
        for k, forms in options.items():
            first: list[int] = []
            for f in forms:
                for variant in {f, f.lower(), f.capitalize(), " " + f, " " + f.lower()}:
                    t = tok.encode(variant, add_special_tokens=False)
                    if t and t[0] not in first:
                        first.append(int(t[0]))
            ids[k] = first
        flat = [i for v in ids.values() for i in v]
        mode = "words"
        if len(flat) != len(set(flat)) or any(not v for v in ids.values()):
            mode = "letters"
            ids = {}
            for j, k in enumerate(options):
                letter = chr(ord("A") + j)
                ids[k] = sorted({int(t[0]) for t in (tok.encode(letter, add_special_tokens=False),
                                                     tok.encode(" " + letter, add_special_tokens=False)) if t})
        self._option_ids[key] = (mode, ids)
        return mode, ids

    def option_probs(self, images: list[Any], system: str, prompt: str,
                     options: dict[str, list[str]], timeout_s: float = 10.0) -> OptionScores:
        """Один forward (без генерации): softmax логитов первого токена ответа, сведённый к вариантам."""
        if self._model is None:
            self.load()
        import torch

        with self._lock:
            t0 = time.perf_counter()
            mode, ids = self._option_token_ids(options)
            if mode == "letters":  # варианты неразличимы по первому токену — просим букву
                legend = "; ".join(f"{chr(ord('A') + j)} — {forms[0]}" for j, (k, forms) in enumerate(options.items()))
                prompt = f"{prompt}\nВместо слова ответь одной буквой: {legend}."
            inputs = self._inputs(self._messages(images, system, prompt))
            with torch.inference_mode():
                try:
                    out = self._model(**inputs, logits_to_keep=1)
                except TypeError:
                    out = self._model(**inputs)
            logits = out.logits[0, -1].float()
            p_all = torch.softmax(logits, dim=-1)
            mass_k = {k: float(p_all[v].sum()) if v else 0.0 for k, v in ids.items()}
            total = sum(mass_k.values())
            probs = {k: (m / total if total > 0 else 1.0 / len(ids)) for k, m in mass_k.items()}
            return OptionScores(probs=probs, mass=round(total, 4), latency_s=time.perf_counter() - t0, mode=mode,
                                meta={"prompt_tokens": int(inputs["input_ids"].shape[1]),
                                      "vram_peak_mb": _vram_peak_mb()})

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id, "quant": self.quant, "loaded": self.loaded,
                "load_s": round(self.load_s, 2), "offline": self.local_files_only, "license": self.license_info,
                "cache_dir_set": bool(self.cache_dir), "vram_after_load_mb": self.vram_after_load_mb}


def _free_vram_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 2**20
    except Exception:  # noqa: BLE001
        pass
    return None


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None
