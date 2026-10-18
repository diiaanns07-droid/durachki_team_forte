"""Проверка лицензий чекпойнтов (этап setup, нужен интернет или уже скачанный снапшот).

    python -m ml.detect.licenses            # офлайн: что заявлено в models.json + LICENSE-файлы в weights/hf
    python -m ml.detect.licenses --check    # онлайн: сверка с card_data.license на HF Hub

Результат — отчёт; models.json вручную переводится в license_status=verified только после просмотра человеком.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .weights import find_cache_dir

MODELS_JSON = Path(__file__).with_name("models.json")
PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd"}


def load_models() -> list[dict[str, Any]]:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def local_license_files(model_id: str) -> list[Path]:
    cache = find_cache_dir(model_id)
    if cache is None:
        return []
    snap = cache / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return sorted(p for p in snap.rglob("*") if p.is_file() and p.name.upper().startswith(("LICENSE", "README")))


def check_remote(model_id: str) -> Optional[str]:
    from huggingface_hub import model_info

    info = model_info(model_id)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = getattr(card, "license", None) if card is not None else None
    if lic is None and isinstance(card, dict):
        lic = card.get("license")
    return str(lic).lower() if lic else None


def report(remote: bool = False) -> list[dict[str, Any]]:
    rows = []
    for m in load_models():
        mid = m["hf_id_or_url"]
        row = {"id": mid, "declared": m["license"], "status": m.get("license_status", "")}
        if mid.startswith("local:"):
            row["note"] = "производная модель"
        else:
            row["local_files"] = [str(p) for p in local_license_files(mid)]
            if remote:
                try:
                    row["hub_license"] = check_remote(mid)
                    row["match"] = row["hub_license"] == str(m["license"]).lower()
                    row["permissive"] = row["hub_license"] in PERMISSIVE
                except Exception as e:  # noqa: BLE001
                    row["hub_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="сверить с HF Hub (нужен интернет)")
    a = ap.parse_args(argv)
    print(json.dumps(report(remote=a.check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
