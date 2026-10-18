"""python -m backend.api.make_key [--append] — сгенерировать API-ключ.

Печатает ключ и строку для env. С --append дописывает его в файл ключей
(SENTINEL_API_KEYS_FILE, по умолчанию data/api_keys.txt, права 0600)."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from .config import Settings
    from .security import fingerprint, generate_key, write_private
    ap = argparse.ArgumentParser(prog="python -m backend.api.make_key")
    ap.add_argument("--append", action="store_true", help="дописать в файл ключей")
    ap.add_argument("--quiet", action="store_true", help="печатать только ключ")
    a = ap.parse_args(argv)
    key = generate_key()
    if a.append:
        s = Settings.from_env()
        assert s.api_keys_file is not None
        write_private(s.api_keys_file, key + "\n", append=True)
        if not a.quiet:
            print(f"# appended to {s.api_keys_file} (fp={fingerprint(key)})", file=sys.stderr)
    if a.quiet:
        print(key)
    else:
        print(key)
        print(f"# SENTINEL_API_KEYS={key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
