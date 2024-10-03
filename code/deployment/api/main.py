from fastapi import FastAPI, UploadFile
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import os

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "../../../models/cifar10_model.h5 ")

model = tf.keras.models.load_model(model_path)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
def read_root():
    return {"message": "CIFAR-10 Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((32, 32))
    input_array = np.array(image) / 255.0 
    input_array = input_array.reshape((1, 32, 32, 3))
    prediction_logits = model.predict(input_array)
    prediction_probs = tf.nn.softmax(prediction_logits).numpy()
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_label = class_names[predicted_class_index]
    
    return {
        "prediction": predicted_class_label,
        "probabilities": prediction_probs.tolist()
    }

