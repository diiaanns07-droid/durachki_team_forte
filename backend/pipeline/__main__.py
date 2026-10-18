"""python -m backend.pipeline {run,batch,profile,warmup} — см. cli.py."""
from __future__ import annotations

if __name__ == "__main__":
    from backend.pipeline.cli import main

    raise SystemExit(main())
