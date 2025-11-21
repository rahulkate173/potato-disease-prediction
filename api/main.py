from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import tensorflow as tf
app = FastAPI()
# TF_SERVING_URL = r"http://localhost:8501/v1/models/disease_model:predict" -> worong 
TF_SERVING_URL = r"http://localhost:8051/v1/models/disease_models:predict"


#MODEL = tf.keras.layers.TFSMLayer(r"D:/OneDrive/Desktop/disease-prediction/models/1", call_endpoint='serving_default')
CLASS_NAMES = ['Early Bright','Late Bright','Healthy']
@app.get('/ping')
async def ping():
    return "hello i am alive"

def read_file_as_image(bytes) -> np.ndarray:
    image = Image.open(BytesIO(bytes)).convert("RGB")
    # image = image.resize((224, 224))           # ⬅ match your model input size
    image = np.array(image).astype(np.float32)
    # image = image / 255.0                      # ⬅ normalize
    return image.astype('float64')

@app.post('/predict')
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,axis=0).astype(np.float32)
    payload = {"instances": image_batch.tolist()}

    response = requests.post(TF_SERVING_URL, json=payload)

    if response.status_code != 200:

        return {"error": "TF Serving Error", "details": response.text}

    prediction = response.json()["predictions"][0]
    # outputs = MODEL(image_batch)
    # scores = outputs["output_0"].numpy()[0]
    class_id = int(np.argmax(prediction))
    label = CLASS_NAMES[class_id]
    confidence = float(np.max(prediction) * 100)
    return {
        "class_name":label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)

# docker pull tensorflow/serving
## docker run -t --rm -p 8051:8051 -v D:/OneDrive/Desktop/disease-prediction:/disease-prediction tensorflow/serving --rest_api_port=8051 --model_config_file=/disease-prediction/model_config.config

#