from fastapi import FastAPI, UploadFile,File
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import wandb
from PIL import Image
from skimage import transform

if report.failed and not hasattr(report, "wasxfail"): 
     self.testsfailed += 1 
   
# name of the model artifact
artifact_model_name = "emotions/model_export:latest"


# initiate the wandb project
run = wandb.init(project="emotions",job_type="api")

best_model = wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c")

modelwb = tf.keras.models.load_model(best_model.name)


def load(filename):
   np_image = Image.open(io.BytesIO(filename)) 
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (20, 20, 1))
   np_image = np.expand_dims(np_image, axis=0)
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
    prediction = np.around(modelwb.predict(load(img)), decimals=2)
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
