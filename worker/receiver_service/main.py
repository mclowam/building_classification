from fastapi import FastAPI, UploadFile, File, HTTPException
from worker.model.inference import inference_predict_style

app = FastAPI()

@app.get('/')
async def health():
    return {"status": "OK"}


@app.post('/predict')
async def predict_style(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail='файл должен быть картинкой')

    image = await file.read()
    return inference_predict_style(image)




