# VRAM-план (RTX 4060 Laptop, 8 ГБ) — анализ в приоритете

**Источник цифр.**
- **Измерено L2** на ноутбуке:
  - свободно ~5.3 ГБ VRAM (остальное занимают Windows и дисплей);
  - Qwen3-VL-4B-Instruct в nf4 с bf16 compute: веса 2.7 ГБ, пик 4.7 ГБ на 6 кадрах;
  - Qwen2.5-VL-7B не влезает.
- **Оценки A03** из `backend/pipeline/config.yaml`, не замер: `vram_gb` detect 1.0 / pose 0.5 / temporal 1.0 / vlm 5.0, `vram_reserve_gb` 0.7.
- Остальных чисел здесь нет — их даёт команда замера ниже.

## Правила
1. **Одна GPU-задача за раз.** `SENTINEL_WORKERS=1` (супервизор и `start.ps1` выставляют принудительно), `gpu_slots=1` у A02. Владелец GPU один — spawn-воркер A03.
2. **Стадии на GPU идут последовательно, не одновременно** (A03, `executor.unload_before`):
   - перед VLM выгружаются detect, pose и temporal;
   - перед detect, pose и temporal выгружается VLM.
   - При пике VLM 4.7 ГБ и 5.3 ГБ свободных запас ~0.6 ГБ, поэтому держать детекторы резидентными во время VLM **нельзя**.
3. **Если VRAM не хватает**, A03 вызывает `prepare_vram` → `unload()` модулей → перезапуск воркера: CUDA-контекст умирает вместе с процессом.
   - Не хватает и после этого — VLM-стадия пропускается с warning.
   - **Неполный анализ не превращается в `no_violation`**: обязательные стадии A03 — ingest, detect, track, scene, rules, fusion; VLM дополнительная.
4. **Копилот (A14) не берёт GPU во время мероприятия.** По умолчанию он выключен (`enabled: mock`).
   - Если включить: эмбеддинги e5-small на CPU (`embed.device: cpu`). GGUF-моделей нет, поэтому генерации нет — работает extractive backend.
   - Появится GGUF — только `gpu_layers: 0` (CPU).
   - Вариант «копилот через общий экземпляр Qwen3-VL» у A14 не реализован; это предложение (в очереди после анализа, приоритет ниже).
5. **Перед прогоном:**
   - `nvidia-smi` — закрыть всё, что держит VRAM: браузер с аппаратным ускорением, игры, другие CUDA-процессы;
   - preflight требует ≥ 5.0 ГБ свободной (`deploy/config/deploy.yaml: preflight.min_free_vram_gb`).

## Риск, вынесенный во флаг (по умолчанию ВЫКЛ)
По оценке A03 VLM нужно 5.0 + 0.7 = 5.7 ГБ, а свободно 5.3 ГБ. Тогда `prepare_vram` может решить, что памяти не хватает, и пропустить VLM, хотя измеренный пик (4.7 ГБ) помещается.

Оверлей `deploy/config/pipeline_vram_measured.yaml` выставляет vlm_* = 4.7 и reserve = 0.5. Включать его **только** если замер покажет пропуск VLM из-за нехватки VRAM:
```
SENTINEL_PIPELINE_CONFIG=deploy/config/pipeline_vram_measured.yaml
```

## Команда замера для L2 (одна GPU-очередь — сервис должен быть остановлен)
```powershell
deploy\windows\stop.ps1
# A: дефолты A03                                         B: оверлей (гипотеза)
.venv\Scripts\python deploy\offline_worker_smoke.py --video <локальный ролик> --report runs\deploy\vram_A.json
$env:SENTINEL_PIPELINE_CONFIG="deploy\config\pipeline_vram_measured.yaml"
.venv\Scripts\python deploy\offline_worker_smoke.py --video <тот же ролик> --report runs\deploy\vram_B.json
Remove-Item Env:SENTINEL_PIPELINE_CONFIG
```
- **Что сравниваем:** A — дефолты A03, B — оверлей.
- **Выход:** `runs\deploy\vram_{A,B}.json`.
- **Метрики:**
  - `result.stages.*.vram_peak_mb`, `result.vram_events`;
  - запускалась ли `vlm_verify` (`status`, `origin`);
  - `timings_s.run`, `verdict`, `n_violations`, `n_warnings`.
- **Решение:** B включаем, только если у A VLM пропущен из-за VRAM, а у B — выполнен без OOM (`api.err.log`, `vram_events`).

В отчёт и в чат идут только числа — без кадров и описаний.

Дополнительно: `python -m backend.pipeline profile <video> --json` (A03) и `python -m ml.vlm.bench` (A11).
