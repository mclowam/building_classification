# Building Classification

Telegram-бот для определения архитектурного стиля здания по фотографии. Изображение отправляется в FastAPI, модель на PyTorch возвращает стиль и оценку уверенности.

Стили: модерн, брутализм, классицизм, готика, хай-тек и модернизм.

## Стек

Python, PyTorch, torchvision, FastAPI, aiogram, Docker.

## Запуск

Установите зависимости в виртуальное окружение:

```bash
python -m venv .venv
```

В PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

В Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Добавьте токен Telegram-бота в `.env`. Веса модели должны находиться в `worker/model/weights/best_model.pth`. Датасет для работы API не нужен.

Из корня проекта запустите API:

```bash
python -m uvicorn worker.receiver_service.main:app --port 8001
```

В другом терминале с тем же окружением:

```bash
python -m client_bot.bot
```

Отправьте боту фотографию здания. Документация API доступна по адресу [localhost:8001/docs](http://localhost:8001/docs).

API также можно запустить через Docker:

```bash
docker compose up --build
```

## Обучение

Поместите фотографии в `worker/model/data/`, распределив их по папкам с названиями стилей, затем выполните:

```bash
python -m worker.model.train
```

Используется собственная CNN с двумя свёрточными слоями. Данные делятся на обучение и валидацию в соотношении 80/20 с фиксированным seed. Лучшие веса сохраняются в `worker/model/weights/`.

## API

`GET /` — проверка работы сервиса.

`POST /predict` — загрузка изображения в поле `file`:

```bash
curl -X POST http://localhost:8001/predict -F "file=@building.jpg"
```

В PowerShell используйте `curl.exe`. Ответ содержит `style` и `confidence`.

## Структура

```text
client_bot/bot.py                 Telegram-бот
worker/receiver_service/main.py   FastAPI
worker/model/model.py            Архитектура CNN и оценка модели
worker/model/settings.py         Данные и параметры обучения
worker/model/train.py            Обучение
worker/model/inference.py        Предсказание
```

Фотографии, веса и `.env` не входят в репозиторий. Независимая тестовая оценка модели пока не проведена; `confidence` не гарантирует правильность ответа.
