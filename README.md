# Gait

Генератор изображений по текстовому запросу с подключаемыми моделями (в том числе без встроенной цензуры).

## Запуск
```
pip install -r requirements.txt
uvicorn gait.server:app --reload
```
Открой http://127.0.0.1:8000.

## Модели
Описываются в `models.yaml`:
- `diffusers` — локально, любой SD 1.5 / SDXL чекпоинт (`.safetensors` из Civitai или репозиторий Hugging Face). Нужна GPU и раскомментированные зависимости в `requirements.txt`.
- `a1111` — удалённый AUTOMATIC1111 / Forge, запущенный с `--api`.
- `comfyui` — ComfyUI: либо `checkpoint` (встроенный txt2img-граф), либо свой `workflow_file` (экспорт «Save (API Format)») с плейсхолдерами `{{prompt}}`, `{{negative_prompt}}`, `{{width}}`, `{{height}}`, `{{steps}}`, `{{guidance}}`, `{{seed}}`.

## Ограничения
Только для взрослых и вымышленных персонажей. Запросы, упоминающие несовершеннолетних, блокируются (`gait/guard.py`). Не используй для изображений реальных людей без их согласия.
