from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
MODEL = tf.keras.layers.TFSMLayer(
    r"D:\OneDrive\Desktop\disease-prediction\models\version_1",
    call_endpoint="serving_default"   # likely correct endpoint
)
CLASS_NAMES = ['Early Bright','Late Bright','Healthy']
@app.get('/ping')
async def ping():
    return "hello i am alive"

def read_file_as_image(bytes) -> np.ndarray:
    array = np.array(Image.open(BytesIO(bytes)))
    return array
@app.post('/predict')
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,axis=0)
    predictions = MODEL.predict(image_batch)
    pass

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)