import asyncio
import os
import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8001/predict")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                WORKER_URL,
                files={"file": ("image.jpg", file_bytes, "image/jpeg")},
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            await message.answer(f"Ошибка воркера: {e}")
            return

    await message.answer(
        f"Стиль: {result['style']}\n"
        f"Уверенность: {result['confidence']:.2%}"
    )


@dp.message(F.text)
async def handle_text(message: Message):
    await message.answer("Пришли мне фото здания, и я определю его архитектурный стиль.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
