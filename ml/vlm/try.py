"""Точка входа `python -m ml.vlm.try video.mov --t 22` (логика — в cli_try; `try` — ключевое слово Python,
поэтому импортировать этот модуль напрямую нельзя, только через -m или importlib)."""
from ml.vlm.cli_try import main

if __name__ == "__main__":
    raise SystemExit(main())
