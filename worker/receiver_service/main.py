from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from worker.model.inference import inference_predict_style

app = FastAPI(title="Building Classification")

@app.get('/')
async def health():
    return {"status": "OK"}


@app.post('/predict')
async def predict_style(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail='файл должен быть картинкой')

    image = await file.read()
    try:
        return await run_in_threadpool(inference_predict_style, image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение") from error
