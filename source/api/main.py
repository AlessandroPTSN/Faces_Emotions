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

#import pathlib
#from pathlib import Path
#files = sorted(pathlib.Path('.').glob('**/haarcascade_frontalface_default.xml'))

#for i in files:
#     a = i



# name of the model artifact
artifact_model_name = "emotions/model_export:latest"


# initiate the wandb project
run = wandb.init(project="emotions",job_type="api")

best_model = wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c")

#modelwb = tf.keras.models.load_model(best_model.name)
modelwb = load_model(best_model.name)




#def read_imagefile(file) -> Image.Image:
#    image = Image.open(BytesIO(file))
#    image = np.array(image).astype('float32')
#    return image
     
     
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#face_cascade = cv2.CascadeClassifier(str(a))

def load2(ft):
#    ft = read_imagefile(ft)
     foto=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
     faces = face_cascade.detectMultiScale(foto, 1.3, 3)
     #if faces == ():
     #     color=cv2.resize(foto,(20,20))
     if 1>2:
          print("lol")
     else:
          for (x,y,w,h) in faces:
               cv2.rectangle(foto, (x,y), (x+w, y+h), (0,0,255), 2)
               color = foto[y:y+h, x:x+w]
          color=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
          if color[1,1,0] == 255:
               color=cv2.resize(color,(20,20))
     return color
      
     
     
def prepare(image):
    IMG_SIZE = 600
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,3)
     
#def load(filename):
#   np_image = Image.open(io.BytesIO(filename)) 
#   np_image = np.array(np_image).astype('float32')
#   np_image = transform.resize(np_image, (20, 20, 1))
#   np_image = np.expand_dims(np_image, axis=0)
#   return np_image

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


# run the model inference and use a face data structure via POST to the API.
@app.post("/face") 
async def root(file: UploadFile = File(...)):
    img = await file.read()
    img = np.fromstring(img, np.uint8)
    images = cv2.imdecode(img, cv2.IMREAD_COLOR).astype(np.float32)
    #images = np.fromstring(img, np.uint8)
    #images = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #images = np.array(Image.open(img))
    
    #images = np.array(Image.open(img))
    #images = np.fromstring(Image.open(img), np.uint8)
    #images = read_imagefile(img)
    
    #foto=cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    #faces = face_cascade.detectMultiScale(foto, 1.3, 3)
    #if faces == ():
    if 1>2:
        result = "Unrecognized Face"
    else:
          prediction = np.around(modelwb.predict(load2(prepare(images))), decimals=2)
          string = ','.join(str(x) for x in prediction)
          if string == "[1. 0. 0. 0. 0. 0.]":
               result = "Surprise"
          elif string == "[0. 1. 0. 0. 0. 0.]":
               result = "Sad"
          elif string == "[0. 0. 1. 0. 0. 0.]":
               result = "Neutral"        
          elif string == "[0. 0. 0. 1. 0. 0.]":
               result = "Happy"
          elif string == "[0. 0. 0. 0. 1. 0.]":
               result = "Fear"
          elif string == "[0. 0. 0. 0. 0. 1.]":
               result = "Angry"    
    return result
