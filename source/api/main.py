from fastapi import FastAPI, UploadFile,File
from io import BytesIO 
import numpy as np
from keras.models import load_model
import wandb
from PIL import Image
from skimage import transform 
import cv2
#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# name of the model artifact
#artifact_model_name = "emotions/model_export:latest"


# initiate the wandb project
#run = wandb.init(project="emotions",job_type="api")

#best_model = wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c")


#modelwb = load_model(best_model.name)

#modelwb = load_model(wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c").name)
modelwb =  wandb.restore('modell.h5', run_path="alessandroptsn/uncategorized/35r2qcuc")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

color = 0


#def load3(gray):
#   imagee = np.expand_dims(transform.resize(np.array(gray).astype('float32'), (20, 20, 1)), axis=0)
#   return imagee



#def load2(ft):
#   global color
#   foto=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
#   faces = face_cascade.detectMultiScale(foto, 1.3, 3)
#   for (x,y,w,h) in faces:
#       cv2.rectangle(foto, (x,y), (x+w, y+h), (0,0,255), 2)
#       color = foto[y:y+h, x:x+w]
#   color=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
#   color=cv2.resize(color,(20,20))
#   color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
#   return color


#def load(filename):
#   image_stream = BytesIO(filename)
#   np_image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
#   return np_image





#def load3(gray):
#   imagee = np.expand_dims(transform.resize(np.array(gray).astype('float32'), (20, 20, 1)), axis=0)
#   return imagee



#def load2(ft):
#   global color
#   foto=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
#   faces = face_cascade.detectMultiScale(foto, 1.3, 3)
#   for (x,y,w,h) in faces:
#       cv2.rectangle(foto, (x,y), (x+w, y+h), (0,0,255), 2)
#       color = foto[y:y+h, x:x+w]
#   color=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
#   color=cv2.resize(color,(20,20))
#   color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
#   return color


def load(filename):
   image_stream = BytesIO(filename)
   ft = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)


   global color
   foto=cv2.cvtColor(ft, cv2.COLOR_BGR2RGB)
   faces = face_cascade.detectMultiScale(foto, 1.3, 3)
   for (x,y,w,h) in faces:
       cv2.rectangle(foto, (x,y), (x+w, y+h), (0,0,255), 2)
       color = foto[y:y+h, x:x+w]
   color=cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
   color=cv2.resize(color,(20,20))
   color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

   imagee = np.expand_dims(transform.resize(np.array(color).astype('float32'), (20, 20, 1)), axis=0)
   return imagee


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
    #prediction = np.around(modelwb.predict(load3(load2(load(img)))), decimals=2)
    prediction = np.around(modelwb.predict(load(img)), decimals=1)
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
