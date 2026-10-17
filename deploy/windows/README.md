# Windows-нативный запуск (основной путь)

Рабочая venv — от L1 (`.venv`, Python 3.11, lock `tools\dataprep\requirements-l1.lock.txt`). **Ничего не обновлять вслепую.**

1. `setup.ps1`:
   - сверяет venv с lock (`envcheck`);
   - `-InstallMissing` — только недостающие fastapi/uvicorn/python-multipart под `-c lock`;
   - `gpu_probe` (сервис должен быть остановлен);
   - `weights.py adopt` + `verify` (офлайн, по кэшу `weights\hf\hub`);
   - `deploy\.env` со случайным ключом.
2. `offline_smoke.ps1 -Firewall` (PowerShell **от администратора**):
   - настоящий worker без выхода в сеть;
   - временное правило фаервола только на python.exe, снимается автоматически;
   - лог для питча — `runs\deploy\offline_proof_*.md`.
3. `run_all.ps1`: preflight → `start.ps1` (скрытый супервизор) → ожидание READY (warmup + пробная задача) → печать URL и команды приёмки A16.
4. `status.ps1` / `stop.ps1`.
   - `stop` останавливает только наши процессы (PID + время старта). Данные `data\` не трогает.
   - Повторный `start` не создаёт второй сервер.
5. `monitor.ps1` в отдельном окне — `runs\deploy\monitor.csv` (CPU, RAM, GPU, VRAM раз в 30 с).

## Электропитание
- Пока жив супервизор, система не засыпает (`SetThreadExecutionState`). Дисплей гаснуть может — это не мешает.
- Глобальный план электропитания скрипты не меняют.
- Закрытие крышки подчиняется плану: держите крышку открытой, работайте от сети.

## Другая машина
Сначала `.venv\Scripts\python deploy\gpu_probe.py`, затем `deploy\preflight.py`: VRAM и CUDA-сборка проверяются на месте.
