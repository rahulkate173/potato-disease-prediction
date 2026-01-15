from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# Load SavedModel format using TFSMLayer
MODEL_PATH = os.path.join(os.path. dirname(__file__), "../models/1")
MODEL = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

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

@app. post('/predict')
async def predict(file: UploadFile = File(... )):
    image = read_file_as_image(await file. read())
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    
    # TFSMLayer prediction (returns a dictionary)
    outputs = MODEL(image_batch)
    
    # Get predictions from output dictionary
    # The key might be 'output_0' or 'dense' - check your model
    output_key = list(outputs.keys())[0]
    prediction = outputs[output_key]. numpy()[0]
    
    class_id = int(np.argmax(prediction))
    label = CLASS_NAMES[class_id]
    confidence = float(np. max(prediction) * 100)
    
    return {
        "class_name": label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn. run(app, host='0.0.0.0', port=8000)