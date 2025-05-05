from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(r'#Put model path', compile=False) #replace model path with your own
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


CLASS_NAMES = #make list of the classes you are using.

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    

    predictions = model.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'النوع': predicted_class,
        'الثقه': float(confidence)
    }



from pyngrok import ngrok
ngrok.set_auth_token("#replace with token you get from ngrok website") 
PORT = 8000
import threading
import uvicorn

def run():
    uvicorn.run(app, host='0.0.0.0', port=8000)

thread = threading.Thread(target=run)
thread.start()

# تعيين ngrok
public_url = ngrok.connect(PORT, domain = "#replace with domain you get from ngrok website") 
print(f"Public URL: {public_url}")
