"""python -m backend.api  — запуск uvicorn (строго 1 процесс: очередь и тёплые модели в памяти)."""
from __future__ import annotations

import argparse


def main() -> None:
    from .config import Settings
    s = Settings.from_env()
    ap = argparse.ArgumentParser(prog="python -m backend.api")
    ap.add_argument("--host", default=s.host)
    ap.add_argument("--port", type=int, default=s.port)
    ap.add_argument("--log-level", default=s.log_level.lower())
    a = ap.parse_args()
    import uvicorn
    uvicorn.run("backend.api.main:app", host=a.host, port=a.port, workers=1,
                log_level=a.log_level, access_log=False, proxy_headers=False,
                timeout_keep_alive=30)


if __name__ == "__main__":
    main()
