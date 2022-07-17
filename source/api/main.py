from fastapi import FastAPI, UploadFile,File
import io
from io import BytesIO
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.models import Model
from keras.models import load_model
import wandb
from PIL import Image
from skimage import transform
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# name of the model artifact
artifact_model_name = "emotions/model_export:latest"


# initiate the wandb project
run = wandb.init(project="emotions",job_type="api")

best_model = wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c")

#modelwb = tf.keras.models.load_model(best_model.name)
modelwb = load_model(best_model.name)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def load2(ft):
   #foto_=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
   #foto=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
   foto= cv2.cvtColor(ft, cv2.COLOR_GRAY2BGR)
   faces = face_cascade.detectMultiScale(ft, 1.3, 3)
   for (x,y,w,h) in faces:
       cv2.rectangle(foto, (x,y), (x+w, y+h), (0,0,255), 2)
       color = foto[y:y+h, x:x+w]
   color=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
   #if color[1,1,0] == 255:
   color=cv2.resize(color,(20,20))
   return color
      


def load(filename):
   np_image = Image.open(io.BytesIO(filename)) 
   #np_image = np.array(np_image).astype('float32')
   np_image = np.array(np_image).astype('np.uint8')
   #np_image = transform.resize(np_image, (600, 600, 3))
   #np_image = np.expand_dims(np_image, axis=0)
   return np_image

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


# run the model inference and use a face data structure via POST to the API.
@app.post("/face") 
async def root(file: UploadFile = File(...)):
    img = await  file.read()
    prediction = np.around(modelwb.predict(load2(load(img))), decimals=2)
    string = ','.join(str(x) for x in prediction)
    if string == "[1. 0. 0. 0. 0. 0.]":
        result = "Surprise"
    if string == "[0. 1. 0. 0. 0. 0.]":
        result = "Sad"
    if string == "[0. 0. 1. 0. 0. 0.]":
        result = "Neutral"        
    if string == "[0. 0. 0. 1. 0. 0.]":
        result = "Happy"
    if string == "[0. 0. 0. 0. 1. 0.]":
        result = "Fear"
    if string == "[0. 0. 0. 0. 0. 1.]":
        result = "Angry"        
    return result
