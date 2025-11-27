from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load from container path
model = tf.keras.models.load_model("model.h5")
class_names = ["cats", "dogs", "panda"]

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    img = img.resize((128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.expand_dims(x, 0) / 255.0

    preds = model.predict(x)
    class_id = preds.argmax()

    return {"class": class_names[class_id], "confidence": float(preds[0][class_id])}
