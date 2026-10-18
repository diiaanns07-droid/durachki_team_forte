"""Отдача файлов с HTTP Range (плеер UI перематывает .mov без загрузки целиком)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional

from starlette.responses import Response, StreamingResponse

_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.I)
CHUNK = 256 * 1024

MIME = {".mov": "video/quicktime", ".mp4": "video/mp4", ".m4v": "video/mp4", ".jpg": "image/jpeg"}


def parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]] | str:
    """None — отдать целиком; (start, end) включительно; 'invalid' — 416."""
    if not header:
        return None
    first = header.split(",", 1)[0]  # мульти-диапазоны не поддерживаем: отдаём первый
    m = _RANGE.match(first if "=" in first else header)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if a == "" and b == "":
        return "invalid"
    if a == "":  # последние N байт
        n = int(b)
        if n == 0:
            return "invalid"
        return max(0, size - n), size - 1
    start = int(a)
    end = int(b) if b else size - 1
    if start >= size or end < start:
        return "invalid"
    return start, min(end, size - 1)


async def _iter_file(path: Path, start: int, length: int) -> AsyncIterator[bytes]:
    import anyio

    def _read(fh, n: int) -> bytes:  # type: ignore[no-untyped-def]
        return fh.read(n)

    fh = await anyio.to_thread.run_sync(open, path, "rb")
    try:
        await anyio.to_thread.run_sync(fh.seek, start)
        left = length
        while left > 0:
            chunk = await anyio.to_thread.run_sync(_read, fh, min(CHUNK, left))
            if not chunk:
                break
            left -= len(chunk)
            yield chunk
    finally:
        fh.close()


def range_response(path: Path, range_header: Optional[str], method: str = "GET",
                   media_type: Optional[str] = None, cache: str = "private, max-age=3600") -> Response:
    size = os.path.getsize(path)
    mt = media_type or MIME.get(path.suffix.lower(), "application/octet-stream")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": cache, "Content-Type": mt}
    rng = parse_range(range_header, size)
    if rng == "invalid":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}",
                                                  "Accept-Ranges": "bytes"})
    if rng is None:
        start, end, status = 0, size - 1, 200
    else:
        start, end = rng  # type: ignore[misc]
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = max(0, end - start + 1)
    headers["Content-Length"] = str(length)
    if method == "HEAD":
        return Response(status_code=status, headers=headers)
    return StreamingResponse(_iter_file(path, start, length), status_code=status,
                             headers=headers, media_type=mt)
