"""Локальный дамп кадров и ответов VLM для ручного разбора (только на машине команды).

Пишет в privacy.local_dump_dir (по умолчанию outputs/vlm — в .gitignore): JPEG-кадры, промпты и сырые ответы.
В stdout CLI выводит лишь путь и агрегаты — тексты ответов/описания сцен облачным агентам не показываются.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class LocalDump:
    def __init__(self, root: Optional[str], tag: str = "run") -> None:
        self.enabled = bool(root)
        self.dir: Optional[Path] = None
        self.n = 0
        if self.enabled:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.dir = Path(str(root)) / f"{tag}-{stamp}-{os.getpid()}"
            self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, rec: dict[str, Any]) -> None:
        if not self.enabled or self.dir is None:
            return
        self.n += 1
        k = self.n
        imgs = rec.pop("images", None) or []
        names = []
        for j, im in enumerate(imgs):
            name = f"{k:03d}_{rec.get('kind', 'call')}_{j + 1}.jpg"
            try:
                im.convert("RGB").save(self.dir / name, format="JPEG", quality=85)
                names.append(name)
            except Exception:  # noqa: BLE001
                pass
        rec["image_files"] = names
        with open(self.dir / "records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def write_json(self, name: str, obj: Any) -> None:
        if self.enabled and self.dir is not None:
            with open(self.dir / name, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

    @property
    def path(self) -> Optional[str]:
        return str(self.dir) if self.dir is not None else None
