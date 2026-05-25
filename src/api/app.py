from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path

from ..prediction.predict_disease import predict_disease

app = FastAPI(title="Ethiopian Crop Intelligence API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_disease")
async def predict_disease_endpoint(crop_type: str = Form(...), file: UploadFile = File(...)):
    try:
        # read into memory file-like object
        contents = await file.read()
        from io import BytesIO

        buf = BytesIO(contents)
        result = predict_disease(buf, crop_type)
        return JSONResponse(result)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
