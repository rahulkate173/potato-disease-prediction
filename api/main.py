from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
MODEL = tf.keras.layers.TFSMLayer(r"D:/OneDrive/Desktop/disease-prediction/models/version_1", call_endpoint='serving_default')
CLASS_NAMES = ['Early Bright','Late Bright','Healthy']
@app.get('/ping')
async def ping():
    return "hello i am alive"

def read_file_as_image(bytes) -> np.ndarray:
    array = np.array(Image.open(BytesIO(bytes)))
    return array.astype('float')
@app.post('/predict')
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,axis=0).astype(np.float32)
    outputs = MODEL(image_batch)
    logits = outputs['output_0']          # 1×3 tensor
    logits_np = logits.numpy()            # Convert to NumPy

    class_id = int(np.argmax(logits_np[0]))
    label = CLASS_NAMES[class_id]
    confidence = np.round(float(np.max(logits_np[0])*100),2)
    return {
        "class_name":label,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)