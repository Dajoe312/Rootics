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

model = tf.keras.models.load_model(r'C:\Users\bedir\Desktop\Rootics\model\Gradproject.h5', compile=False) #replace model path with your own
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


CLASS_NAMES = ['بطاطس',
 'بطاطس مصابه ب لفحه مبكره',
 'بطاطس مصابه ب لفحه متأخره',
 'تفاح ',
 'تفاح مصاب بالجرب',
 'تفاح مصاب بالعفن الاسود',
 'خوخ',
 'خوخ مصاب ببقع بكتيرية',
 'ذرة',
 'ذرة  مصابه بالصدأ الشائع',
 'ذرة مصابه بتبقع أوراق رمادي',
 'ذرةمصابه لفحة ورقيه شماليه',
 'طماطم',
 'طماطم مصابه ب عث العنكبوت (الاكاروس)',
 'طماطم مصابه ببقع بكتيريه',
 'طماطم مصابه بتعفن في الأوراق',
 'طماطم مصابه بلفحه مبكره',
 'طماطم مصابه بلفحه متأخرة',
 'طماكم مصابه بفيروس تجعد واصفرار الأوراق',
 'عنب',
 'عنب مصاب بالحصبه السوداء (esca)',
 'عنب مصاب بالعفن الأسود',
 'فراولة',
 'فراولة مصابه باحتراق في الاوراق',
 'فلفل',
 'فلفل مصاب ببقع بكتيرية',
 'كرز',
 'كرز مصاب بالبياض الدقيقي']

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
ngrok.set_auth_token("2i3Z0gcH6B3GxBK7AumqIdA201Z_BA1fFCgwrRHVrwEw5m48") #replace token with the one you get from ngrok website
PORT = 8000
import threading
import uvicorn

def run():
    uvicorn.run(app, host='0.0.0.0', port=8000)

thread = threading.Thread(target=run)
thread.start()

# تعيين ngrok
public_url = ngrok.connect(PORT, domain = "gelding-vocal-singularly.ngrok-free.app") #replace domin with the one you get from ngrok website
print(f"Public URL: {public_url}")
