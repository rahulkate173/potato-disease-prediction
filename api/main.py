from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os

app = FastAPI()

# TF Serving URL - Use environment variable for production
TF_SERVING_URL = os.environ.get(
    'TF_SERVING_URL', 
    'http://localhost:8501/v1/models/disease_models:predict'
)

CLASS_NAMES = ['Early Bright', 'Late Bright', 'Healthy']

@app. get('/')
async def root():
    return {
        "message": "Potato Disease Prediction API",
        "endpoints": {
            "health":  "/ping",
            "predict":  "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get('/ping')
async def ping():
    return "hello i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image. open(BytesIO(data)).convert("RGB")
    image = np.array(image).astype(np.float32)
    return image

@app.post('/predict')
async def predict(file: UploadFile = File(... )):
    image = read_file_as_image(await file. read())
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    
    # Send to TF Serving (Docker)
    payload = {"instances": image_batch. tolist()}
    response = requests.post(TF_SERVING_URL, json=payload)
    
    if response. status_code != 200:
        return {"error": "TF Serving Error", "details": response.text}
    
    prediction = response.json()["predictions"][0]
    
    class_id = int(np.argmax(prediction))
    label = CLASS_NAMES[class_id]
    confidence = float(np. max(prediction) * 100)
    
    return {
        "class_name": label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn. run(app, host='0.0.0.0', port=8000)