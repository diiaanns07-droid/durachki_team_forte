"""Приём видео: потоковый разбор multipart прямо в файл (не в память), лимит размера,
сигнатура ISO-BMFF/QuickTime, затем проверка настоящим декодером (cv2 / PyAV / ffprobe)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

ALLOWED_EXT = (".mov", ".mp4", ".m4v")
# ftyp — современный MP4/MOV; старые QuickTime-файлы могут начинаться с wide/free/skip/mdat/moov
_QT_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot", b"uuid"}
_FLUSH = 1 << 20  # пишем на диск порциями ~1 МБ


class UploadError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class StoredUpload:
    path: Path
    filename: str
    ext: str
    size: int
    sha256: str
    content_type: str = ""
    video: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- имена файлов
_HOMOGLYPHS = str.maketrans({"Х": "X", "х": "x", "А": "A", "В": "B", "Е": "E", "К": "K",
                             "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T"})


def clean_filename(name: str) -> str:
    """Только для отображения/меты: без пути, без управляющих символов, NFC."""
    name = unicodedata.normalize("NFC", name or "")
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:200] or "upload"


def normalize_filename(name: str) -> str:
    """Х-3 (кириллица) == X-3 (латиница): для сопоставления с разметкой."""
    return clean_filename(name).translate(_HOMOGLYPHS)


def ext_of(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def check_extension(filename: str) -> str:
    ext = ext_of(filename)
    if ext and ext not in ALLOWED_EXT:
        raise UploadError(415, f"unsupported file type '{ext}': expected .mov/.mp4/.m4v")
    return ext or ".mov"


# ---------------------------------------------------------------- сигнатура
def sniff_isobmff(head: bytes) -> bool:
    """Первые атомы файла: ftyp (или классический QuickTime без ftyp)."""
    off, n = 0, 0
    while off + 8 <= len(head) and n < 16:
        size = int.from_bytes(head[off:off + 4], "big")
        typ = head[off + 4:off + 8]
        if typ not in _QT_ATOMS:
            return False
        if typ in (b"ftyp", b"moov", b"mdat"):
            return True
        if size == 1:
            if off + 16 > len(head):
                return False
            size = int.from_bytes(head[off + 8:off + 16], "big")
        elif size == 0:
            return False
        if size < 8:
            return False
        off += size
        n += 1
    return False


def probe_mp4(path: Path) -> dict[str, Any]:
    """Чистый Python: длительность из moov/mvhd, размеры из tkhd. Без декодирования."""
    out: dict[str, Any] = {}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            moov = _find_atom(fh, 0, end, b"moov")
            if not moov:
                return out
            m0, m1 = moov
            mvhd = _find_atom(fh, m0, m1, b"mvhd")
            if mvhd:
                fh.seek(mvhd[0])
                b = fh.read(32)
                if b[:1] == b"\x01":
                    ts = int.from_bytes(b[20:24], "big")
                    dur = int.from_bytes(b[24:32], "big")
                else:
                    ts = int.from_bytes(b[12:16], "big")
                    dur = int.from_bytes(b[16:20], "big")
                if ts > 0:
                    out["duration"] = round(dur / ts, 3)
            pos = m0
            while pos < m1:
                trak = _find_atom(fh, pos, m1, b"trak")
                if not trak:
                    break
                tkhd = _find_atom(fh, trak[0], trak[1], b"tkhd")
                if tkhd:
                    fh.seek(tkhd[0])
                    b = fh.read(100)
                    base = 40 if b[:1] == b"\x00" else 52  # matrix
                    if len(b) >= base + 44:
                        w = int.from_bytes(b[base + 36:base + 40], "big") / 65536
                        h = int.from_bytes(b[base + 40:base + 44], "big") / 65536
                        if w > 0 and h > 0:
                            a = int.from_bytes(b[base:base + 4], "big", signed=True)
                            bb = int.from_bytes(b[base + 4:base + 8], "big", signed=True)
                            rot = {(0, 65536): 90, (-65536, 0): 180, (0, -65536): 270}.get((a, bb), 0)
                            out.update(width=int(w), height=int(h), rotation=rot)
                            break
                pos = trak[1]
    except (OSError, ValueError):
        pass
    return out


def _find_atom(fh: Any, start: int, end: int, want: bytes) -> Optional[tuple[int, int]]:
    """(начало payload, конец атома) первого атома want в [start, end)."""
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        hdr = fh.read(8)
        if len(hdr) < 8:
            return None
        size = int.from_bytes(hdr[:4], "big")
        typ = hdr[4:8]
        hlen = 8
        if size == 1:
            size = int.from_bytes(fh.read(8), "big")
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return None
        if typ == want:
            return pos + hlen, min(pos + size, end)
        pos += size
    return None


# ---------------------------------------------------------------- декодер
def _probe_cv2(path: Path) -> Optional[dict[str, Any]]:
    try:
        import cv2  # lazy
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise UploadError(415, "file is not a decodable video (decoder could not open it)")
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise UploadError(415, "file is not a decodable video (no frames could be decoded)")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        h, w = frame.shape[:2]
        info: dict[str, Any] = {"decoder": "cv2", "fps_nominal": round(fps, 3), "frames": int(n),
                                "decoded_width": int(w), "decoded_height": int(h)}
        if fps > 0 and n > 0:  # при переменном fps — только оценка; длительность берём из контейнера
            info["duration_est"] = round(n / fps, 3)
        return info
    finally:
        cap.release()


def _probe_av(path: Path) -> Optional[dict[str, Any]]:
    try:
        import av  # lazy (PyAV, BSD)
    except ImportError:
        return None
    try:
        with av.open(str(path)) as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is None:
                raise UploadError(415, "file has no video stream")
            for fr in c.decode(vs):
                info = {"decoder": "pyav", "decoded_width": fr.width, "decoded_height": fr.height,
                        "fps_nominal": float(vs.average_rate or 0)}
                if c.duration:
                    info["duration_est"] = round(c.duration / 1_000_000, 3)
                return info
    except UploadError:
        raise
    except Exception as e:  # av.AVError и пр.
        raise UploadError(415, f"file is not a decodable video ({type(e).__name__})") from None
    raise UploadError(415, "file is not a decodable video (no frames could be decoded)")


def _probe_ffprobe(path: Path) -> Optional[dict[str, Any]]:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        # один пакет, а не -count_packets: POST должен ответить сразу после записи файла
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                            "-show_entries", "stream=width,height:packet=pts_time",
                            "-show_entries", "format=duration", "-of", "json", str(path)],
                           capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        data = json.loads(r.stdout or b"{}")
    except ValueError:
        data = {}
    streams = data.get("streams") or []
    if r.returncode != 0 or not streams or not (data.get("packets") or []):
        raise UploadError(415, "file is not a decodable video (ffprobe found no video packets)")
    st = streams[0]
    info: dict[str, Any] = {"decoder": "ffprobe", "decoded_width": st.get("width"),
                            "decoded_height": st.get("height")}
    try:
        info["duration_est"] = round(float(data.get("format", {}).get("duration")), 3)
    except (TypeError, ValueError):
        pass
    return info


DecoderProbe = Callable[[Path], Optional[dict[str, Any]]]
DEFAULT_PROBES: tuple[DecoderProbe, ...] = (_probe_cv2, _probe_av, _probe_ffprobe)


def validate_video(path: Path, mode: str = "required",
                   probes: Optional[tuple[DecoderProbe, ...]] = None) -> dict[str, Any]:
    """Открыть файл декодером и прочитать хотя бы один кадр. UploadError(415) если не видео,
    UploadError(503) если в режиме required нет ни одного декодера."""
    info = probe_mp4(path)
    if mode == "off":
        return info
    for probe in (DEFAULT_PROBES if probes is None else probes):
        r = probe(path)
        if r is not None:
            info.update({k: v for k, v in r.items() if v is not None})
            if "duration" not in info and "duration_est" in info:  # нет mvhd — хотя бы оценка
                info["duration"] = info["duration_est"]
            info["validated"] = True
            return info
    if mode == "required":
        raise UploadError(503, "no video decoder available on server (install opencv-python)")
    info["validated"] = False
    return info


# ---------------------------------------------------------------- стриминг
class _FileSink:
    """Пишет поток байт в файл: хеш, лимит, сигнатура по первым байтам."""

    def __init__(self, path: Path, max_bytes: int):
        self.path = path
        self.max_bytes = max_bytes
        self.size = 0
        self.buf = bytearray()
        self.head = b""
        self.hasher = hashlib.sha256()
        self.fh = open(path, "wb")

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self.size += len(data)
        if self.size > self.max_bytes:
            raise UploadError(413, f"file too large: limit {self.max_bytes // (1024 * 1024)} MB")
        self.buf += data
        if not self.head and len(self.buf) >= 4096:
            self._check_head()

    def _check_head(self) -> None:
        self.head = bytes(self.buf[:4096])
        if not sniff_isobmff(self.head):
            raise UploadError(415, "file is not an MP4/QuickTime video (no ftyp signature)")

    @property
    def pending(self) -> int:
        return len(self.buf)

    def flush(self) -> None:
        if self.buf:
            data = bytes(self.buf)
            self.buf.clear()
            self.hasher.update(data)
            self.fh.write(data)

    def finish(self) -> None:
        if self.size == 0:
            raise UploadError(400, "empty file")
        if not self.head:
            self._check_head()
        self.flush()
        self.fh.close()

    def abort(self) -> None:
        try:
            self.fh.close()
        except OSError:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _parse_cd(value: str) -> tuple[str, Optional[str]]:
    name, filename, filename_star = "", None, None
    for m in re.finditer(r';\s*([\w*-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^;]*)', ";" + value.split(";", 1)[-1]):
        k, v = m.group(1).lower(), m.group(2).strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"')
        if k == "name":
            name = v
        elif k == "filename":
            filename = v
        elif k == "filename*":
            from urllib.parse import unquote
            enc, _, rest = v.partition("''")
            filename_star = unquote(rest or enc, encoding="utf-8", errors="replace")
    return name, filename_star or filename


async def receive_video(stream: AsyncIterator[bytes], headers: dict[str, str], dest_dir: Path,
                        max_bytes: int, run_sync: Callable[..., Any],
                        query_filename: str = "") -> StoredUpload:
    """multipart/form-data (поле file) или «сырое» тело (video/*, application/octet-stream).
    run_sync — anyio.to_thread.run_sync: запись на диск не блокирует event loop."""
    ctype = headers.get("content-type", "")
    clen = headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > max_bytes + (1 << 20):
        raise UploadError(413, f"file too large: limit {max_bytes // (1024 * 1024)} MB")
    tmp = dest_dir / "upload.part"
    if ctype.lower().startswith("multipart/form-data"):
        return await _receive_multipart(stream, ctype, dest_dir, tmp, max_bytes, run_sync)
    if ctype.lower().startswith(("video/", "application/octet-stream")):
        filename = clean_filename(headers.get("x-filename") or query_filename or "upload.mov")
        ext = check_extension(filename)
        sink = _FileSink(tmp, max_bytes)
        try:
            async for chunk in stream:
                sink.feed(chunk)
                if sink.pending >= _FLUSH:
                    await run_sync(sink.flush)
            await run_sync(sink.finish)
        except BaseException:
            sink.abort()
            raise
        return _finalize(sink, dest_dir, filename, ext, ctype)
    raise UploadError(415, "expected multipart/form-data with field 'file'")


async def _receive_multipart(stream: AsyncIterator[bytes], ctype: str, dest_dir: Path, tmp: Path,
                             max_bytes: int, run_sync: Callable[..., Any]) -> StoredUpload:
    try:
        from python_multipart.multipart import MultipartParser, parse_options_header
    except ImportError:  # python-multipart < 0.0.13
        from multipart.multipart import MultipartParser, parse_options_header  # type: ignore

    _, params = parse_options_header(ctype)
    boundary = params.get(b"boundary")
    if not boundary:
        raise UploadError(400, "multipart boundary missing")

    st: dict[str, Any] = {"hf": b"", "hv": b"", "headers": {}, "cur": None, "sink": None,
                          "filename": "", "ext": ".mov", "part_ct": "", "done": False}

    def on_part_begin() -> None:
        st["headers"], st["hf"], st["hv"], st["cur"] = {}, b"", b"", None

    def on_header_field(data: bytes, start: int, end: int) -> None:
        st["hf"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        st["hv"] += data[start:end]

    def on_header_end() -> None:
        st["headers"][st["hf"].decode("latin-1").strip().lower()] = st["hv"].decode("utf-8", "replace")
        st["hf"], st["hv"] = b"", b""

    def on_headers_finished() -> None:
        name, filename = _parse_cd(st["headers"].get("content-disposition", ""))
        if st["sink"] is None and (name == "file" or filename is not None):
            st["filename"] = clean_filename(filename or "upload.mov")
            st["ext"] = check_extension(st["filename"])
            st["part_ct"] = st["headers"].get("content-type", "")
            st["sink"] = _FileSink(tmp, max_bytes)
            st["cur"] = "file"
        else:
            st["cur"] = "skip"

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if st["cur"] == "file":
            st["sink"].feed(data[start:end])

    def on_part_end() -> None:
        if st["cur"] == "file":
            st["done"] = True
        st["cur"] = None

    parser = MultipartParser(boundary, {
        "on_part_begin": on_part_begin, "on_header_field": on_header_field,
        "on_header_value": on_header_value, "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished, "on_part_data": on_part_data,
        "on_part_end": on_part_end})
    try:
        async for chunk in stream:
            try:
                parser.write(chunk)
            except UploadError:
                raise
            except Exception as e:
                raise UploadError(400, f"malformed multipart body ({type(e).__name__})") from None
            sink = st["sink"]
            if sink is not None and sink.pending >= _FLUSH:
                await run_sync(sink.flush)
        try:
            parser.finalize()
        except Exception:
            pass
        sink = st["sink"]
        if sink is None:
            raise UploadError(400, "multipart field 'file' is missing")
        if not st["done"]:
            raise UploadError(400, "upload incomplete (multipart part not terminated)")
        await run_sync(sink.finish)
    except BaseException:
        if st["sink"] is not None:
            st["sink"].abort()
        raise
    return _finalize(st["sink"], dest_dir, st["filename"], st["ext"], st["part_ct"])


def _finalize(sink: _FileSink, dest_dir: Path, filename: str, ext: str, ctype: str) -> StoredUpload:
    final = dest_dir / f"video{ext}"
    os.replace(sink.path, final)
    return StoredUpload(path=final, filename=filename, ext=ext, size=sink.size,
                        sha256=sink.hasher.hexdigest(), content_type=ctype)
