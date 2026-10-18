# Чек-лист: последний час до сдачи (12:00)

Ответственный — L2 (одна GPU-очередь). Отмечать время. **После T−15 ничего не меняем.**

## T−60 … T−45 — заморозка
- [ ] Все нужные PR смерджены, интеграционная ветка на ноутбуке: `git status` чистый, `git log -1` записан.
- [ ] `deploy\windows\stop.ps1`.
- [ ] `.venv\Scripts\python deploy\preflight.py` → **OK**. Любой FAIL чинить по `deploy\RUNBOOK.md`; если фикса нет за 5 мин — откат (`release.py rollback`).
- [ ] `.venv\Scripts\python deploy\envcheck.py` → OK (lock L1, без обновлений).
- [ ] `.venv\Scripts\python deploy\weights.py verify` → OK (обязательные модели, точный commit).
- [ ] Питание от сети, крышка открыта, ВКЛЮЧЁН режим «не беспокоить»; посторонние GPU-процессы закрыты (`nvidia-smi`).

## T−45 … T−30 — доказательство локальности и запуск
- [ ] PowerShell **от администратора**: `deploy\windows\offline_smoke.ps1 -Firewall` → OK.
  - Сохранить `runs\deploy\offline_proof_*.md` (для питча).
  - Правило фаервола снято автоматически: проверить `firewall_python.ps1 -Action Status`.
- [ ] Обычный PowerShell: `deploy\windows\run_all.ps1` → `READY` + публичный URL.
- [ ] `.venv\Scripts\python deploy\tunnel_probe.py real --url <URL> --mb 30 --out runs\deploy\tunnel_real.json` → `ok: true` (загрузка ~30 МБ, 401 без ключа, done).

## T−30 … T−15 — приёмка
- [ ] A16: `python -m eval.jury_sim --url <URL> --key <ключ> --videos data\videos --n 6` → все `done`, время **по замеру** ≤ 300 с.
- [ ] Записать фактические цифры (max/медиана) из отчёта — **только измеренные**.
- [ ] `.venv\Scripts\python deploy\release.py mark-verified --note "A16 ok, URL ..." --report <отчёт A16>`.
- [ ] UI открывается по URL, вход по ключу работает.

## T−15 … T−0 — сдача
- [ ] В форму сдачи: URL, API-ключ (только по защищённому каналу, не в git и не в чат), commit SHA, `deploy\LICENSES.md`, лог локальности.
- [ ] `status.ps1` → `ready`, `restarts` не растёт. Ничего не трогать; ноутбук не закрывать.
- [ ] Если quick URL сменился (упал cloudflared) — сразу сообщить новый (`runs\deploy\public_url.txt`).

## После 12:00
- [ ] Не выключать сервис до конца проверки жюри; `status.ps1` раз в 15 мин.
- [ ] В конце: `stop.ps1`.
