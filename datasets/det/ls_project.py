"""Label Studio (Apache-2.0) локально: запуск, проект с пред-разметкой, экспорт в COCO.

    python datasets/det/ls_project.py start      # сервер на http://localhost:8080 (держать окно открытым)
    python datasets/det/ls_project.py setup      # во втором окне: проект + локальное хранилище + импорт tasks_new.json
    python datasets/det/ls_project.py export     # JSON из Label Studio → export_coco.py → datasets/det/annotations.json
    python datasets/det/ls_project.py tabs       # вкладки «Val» (проверить первыми) и «Train» по split.json
    python datasets/det/ls_project.py config     # напечатать XML-конфиг разметки (для ручной настройки)

Label Studio стоит в отдельном venv datasets/det/.venv-ls (у него жёсткие пины зависимостей).
База, логин и токен — datasets/det/.ls-data/ (в git не попадает). Кадры отдаются из data/ через
Local Storage, сервер слушает только localhost.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, DATA, DET_DIR, FRAMES_DIR, LS_DIR  # noqa: E402

LS_DATA = DET_DIR / ".ls-data"
CREDS = LS_DATA / "credentials.json"
PROJECT = LS_DATA / "project.json"
TITLE = "Kassa Sentinel — детектор сверху"
HOTKEYS = "1234567890"
INSTRUCTION = """<p><b>Правим боксы OWLv2</b>: удалить лишние, поправить рамки, дорисовать пропущенные, затем <b>Submit</b>.</p>
<ul>
<li>Бокс — по видимой части объекта, плотно. Объект виден меньше чем на ~20% или не опознаётся — не размечаем.</li>
<li><b>cash</b> — купюры: каждая пачка/веер отдельно, в руке тоже. Монеты не размечаем.</li>
<li><b>cash_bag</b> — инкассаторская сумка или пломбированный мешок/пакет с деньгами (в руке, на полу, в сейфе).</li>
<li><b>safe_closed / safe_open</b> — корпус сейфа; open, если дверца приоткрыта хоть немного (бокс вместе с дверцей).</li>
<li><b>cell_phone</b> — телефон в руке или на столе. <b>cup</b> — чашка, кружка, стакан (бутылки не размечаем).</li>
<li><b>food</b> — еда и упаковки с едой. <b>money_counter</b> — счётная машинка для купюр (с лотком); плоский детектор купюр и принтер рядом — не размечаем.</li>
<li><b>door_open</b> — открытая дверь: створка + проём. Закрытую дверь не размечаем.</li>
<li><b>person</b> — каждый человек целиком, и обрезанный краем кадра, и за стеклом в соседней комнате. Не отражения в стекле и не крошечные фигуры в дальнем офисе (меньше 40 px).</li>
<li>Кадр без объектов — всё равно <b>Submit</b> (негатив). Непонятный кадр — <b>Skip</b> (в COCO не попадёт).</li>
</ul>"""


def label_config() -> str:
    labels = "\n".join(f'    <Label value="{c["name"]}" background="{c["color"]}" hotkey="{HOTKEYS[i]}"/>'
                       for i, c in enumerate(CLASSES))
    return f"""<View>
  <Header value="$video_id · t=$t_sec с"/>
  <Header value="$qa_note" size="6"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" opacityPerLabel="false">
{labels}
  </RectangleLabels>
</View>"""


def ls_exe() -> Path:
    for name in ("Scripts/label-studio.exe", "bin/label-studio"):
        p = DET_DIR / ".venv-ls" / name
        if p.is_file():
            return p
    raise SystemExit("нет datasets/det/.venv-ls — см. README, раздел «Установка»")


def creds() -> dict:
    if CREDS.is_file():
        return json.loads(CREDS.read_text(encoding="utf-8"))
    LS_DATA.mkdir(parents=True, exist_ok=True)
    # email с доменом без точки («@local») не проходит валидатор формы входа Django → «email and password don't match»;
    # пароль без похожих символов (l/I/1, O/0), чтобы его можно было набрать с экрана
    readable = "abcdefghjkmnpqrstuvwxyz23456789"
    c = {"username": "annotator@localhost", "password": "kassa-" + "".join(secrets.choice(readable) for _ in range(8)),
         "token": secrets.token_hex(20)}
    CREDS.write_text(json.dumps(c, indent=1), encoding="utf-8")
    return c


def cmd_start(args) -> int:
    c = creds()
    env = dict(os.environ,
               LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true",
               LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=str(DATA),
               LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK="true")
    print(f"Label Studio: http://localhost:{args.port}  логин {c['username']}  пароль {c['password']}  "
          f"(хранятся в {CREDS.relative_to(DET_DIR.parents[1])})", flush=True)
    return subprocess.call([str(ls_exe()), "start", "--data-dir", str(LS_DATA), "--port", str(args.port),
                            "--internal-host", "127.0.0.1", "--no-browser",
                            # c 1.2x legacy-токены выключены по умолчанию; флаг действует при создании базы
                            "--enable-legacy-api-token",
                            "--username", c["username"], "--password", c["password"], "--user-token", c["token"]],
                           env=env)


class Api:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def __call__(self, method: str, path: str, body=None, raw: bool = False):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": f"Token {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{method} {path}: HTTP {e.code} {e.read()[:500].decode('utf-8', 'replace')}") from None
        except urllib.error.URLError as e:
            raise SystemExit(f"Label Studio недоступен на {self.url} ({e.reason}) — запустите `ls_project.py start`") from None
        return data if raw else (json.loads(data) if data else None)


def api_from(args) -> Api:
    token = os.environ.get("LS_TOKEN") or creds()["token"]
    return Api(args.url, token)


def project_id(api: Api) -> int | None:
    if PROJECT.is_file():
        pid = json.loads(PROJECT.read_text(encoding="utf-8"))["id"]
        try:
            api("GET", f"/api/projects/{pid}")
            return pid
        except SystemExit:
            pass
    return None


def import_tasks(api: Api, pid: int, path: Path) -> None:
    tasks = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    if not tasks:
        print(f"нечего импортировать ({path} пуст или нет) — сначала build_tasks.py")
        return
    for i in range(0, len(tasks), 250):
        r = api("POST", f"/api/projects/{pid}/import", tasks[i:i + 250])
        print(f"  импорт {i + len(tasks[i:i + 250])}/{len(tasks)}: {r}")
    path.rename(path.with_name(f"tasks_imported_{time.strftime('%Y%m%d_%H%M%S')}.json"))


def cmd_setup(args) -> int:
    api = api_from(args)
    pid = project_id(api)
    if pid is None:
        p = api("POST", "/api/projects", {"title": TITLE, "label_config": label_config(),
                                         "expert_instruction": INSTRUCTION, "show_instruction": True,
                                         "show_skip_button": True, "enable_empty_annotation": True,
                                         "maximum_annotations": 1, "sampling": "Sequential sampling"})
        pid = p["id"]
        api("POST", "/api/storages/localfiles", {"project": pid, "title": "data/frames", "path": str(FRAMES_DIR),
                                                 "use_blob_urls": True, "regex_filter": r".*\.jpg"})
        LS_DATA.mkdir(parents=True, exist_ok=True)
        PROJECT.write_text(json.dumps({"id": pid}), encoding="utf-8")
        print(f"создан проект {pid}")
    else:
        api("PATCH", f"/api/projects/{pid}", {"label_config": label_config(), "expert_instruction": INSTRUCTION})
        print(f"проект {pid} уже есть — обновлены конфиг и инструкция")
    import_tasks(api, pid, args.tasks)
    p = api("GET", f"/api/projects/{pid}")
    print(f"задач: {p.get('task_number')}, размечено: {p.get('num_tasks_with_annotations')}  →  "
          f"{args.url}/projects/{pid}/data")
    return 0


# (заголовок, поле data, значение): «Спорные» — кадры, где второй, контрольный проход нашёл проблему.
TABS = [("0 · Спорные — проверить человеку", "qa", "check"),
        ("1 · Val — проверить первыми (честная метрика)", "split", "val"),
        ("2 · Train — для дообучения", "split", "train")]
QA_RESULTS = DATA / "frames" / "_review" / "qa" / "results"


def qa_marks() -> dict[str, tuple[str, str]]:
    """file → ("check" | "ok", подсказка над кадром) из результатов qa_sheets.py (второй проход)."""
    marks: dict[str, tuple[str, str]] = {}
    for p in sorted(QA_RESULTS.glob("*.jsonl")) if QA_RESULTS.is_dir() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            issues = r.get("issues") or []
            bad = r.get("verdict") == "fix" or r.get("confidence") == "low"
            note = "; ".join(f"#{i['box']} {i.get('detail') or i.get('problem')}" if i.get("box") else
                             str(i.get("detail") or i.get("problem")) for i in issues)
            if r.get("confidence") == "low":
                note = (note + "; " if note else "") + "трудный кадр"
            marks[r["file"]] = ("check", f"Контроль: {note}") if bad else ("ok", "Контроль: замечаний нет")
    ru = QA_RESULTS.parent / "notes_ru.json"          # перевод подсказок для разметчика (если есть)
    if ru.is_file():
        for f, note in json.loads(ru.read_text(encoding="utf-8")).items():
            if marks.get(f, ("",))[0] == "check":
                marks[f] = ("check", note)
    return marks


def all_tasks(api: Api, pid: int) -> list[dict]:
    page, tasks = 1, []
    while True:
        r = api("GET", f"/api/tasks?project={pid}&page={page}&page_size=500&fields=all")
        batch = r.get("tasks", []) if isinstance(r, dict) else r
        tasks += batch
        if len(batch) < 500:
            return tasks
        page += 1


def cmd_tabs(args) -> int:
    """split.json → data.split; результаты контроля → data.qa/qa_note; вкладки «Спорные», Val, Train."""
    api = api_from(args)
    pid = project_id(api)
    if pid is None:
        raise SystemExit("проект не найден — сначала `ls_project.py setup`")
    split_p = DET_DIR / "split.json"
    if not split_p.is_file():
        raise SystemExit("нет datasets/det/split.json — сначала export (или apply_review.py --coco)")
    split = json.loads(split_p.read_text(encoding="utf-8"))["groups"]
    marks = qa_marks()
    api("PATCH", f"/api/projects/{pid}", {"label_config": label_config()})   # заголовок с подсказкой контроля
    n = 0
    for t in all_tasks(api, pid):
        d = dict(t["data"])
        s = split.get(d.get("group") or d.get("video_id"))
        if s:
            d["split"] = s
        qa, note = marks.get(d.get("file"), ("", ""))
        d["qa"], d["qa_note"] = qa, note
        if d != t["data"]:
            api("PATCH", f"/api/tasks/{t['id']}", {"data": d})
            n += 1
    have = {v["data"].get("title") for v in api("GET", f"/api/dm/views?project={pid}")}
    for title, field, value in TABS:
        if title not in have:
            api("POST", "/api/dm/views", {"project": pid, "data": {"title": title, "filters": {
                "conjunction": "and",
                "items": [{"filter": f"filter:tasks:data.{field}", "operator": "equal", "type": "String",
                           "value": value}]}}})
    n_check = sum(m[0] == "check" for m in marks.values())
    print(f"обновлено задач: {n}; спорных по контролю: {n_check} из {len(marks)}; вкладки: "
          f"{', '.join(t for t, _, _ in TABS)}")
    return 0


def cmd_export(args) -> int:
    api = api_from(args)
    pid = project_id(api)
    if pid is None:
        raise SystemExit("проект не найден — сначала `ls_project.py setup`")
    data = api("GET", f"/api/projects/{pid}/export?exportType=JSON&download_all_tasks=false", raw=True)
    LS_DIR.mkdir(parents=True, exist_ok=True)
    out = LS_DIR / "export.json"
    out.write_bytes(data)
    (LS_DIR / f"export_{time.strftime('%Y%m%d_%H%M%S')}.json").write_bytes(data)   # резервная копия
    print(f"экспорт Label Studio: {len(json.loads(data))} задач с аннотациями → {out}")
    import export_coco
    return export_coco.main([str(out), *args.rest])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--port", type=int, default=8080)
    s = sub.add_parser("setup")
    s.add_argument("--tasks", type=Path, default=LS_DIR / "tasks_new.json")
    sub.add_parser("export", help="лишние аргументы уходят в export_coco.py (--val-frac, --val, --resplit, --out-dir)")
    sub.add_parser("config")
    sub.add_parser("tabs", help="вкладки Val/Train в Label Studio по split.json")
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd != "export":
        ap.error(f"неизвестные аргументы: {' '.join(rest)}")
    args.rest = rest
    if args.cmd == "config":
        print(label_config())
        return 0
    return {"start": cmd_start, "setup": cmd_setup, "export": cmd_export, "tabs": cmd_tabs}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
