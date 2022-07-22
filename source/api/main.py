from fastapi import FastAPI, UploadFile,File
from io import BytesIO 
import numpy as np
from keras.models import load_model
import wandb
from PIL import Image
from skimage import transform 
import cv2
from fastapi.responses import HTMLResponse
#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# name of the model artifact
#artifact_model_name = "emotions/model_export:latest"


# initiate the wandb project
#run = wandb.init(project="emotions",job_type="api")

#best_model = wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c")


#modelwb = load_model(best_model.name)

#modelwb = load_model(wandb.restore('model.h5', run_path="alessandroptsn/emotions/skt69t8c").name)

#modelwb = load_model(wandb.restore('model_.h5', run_path="alessandroptsn/uncategorized/2joxlwx7").name)
#modelwb = load_model(wandb.restore('modell.h5', run_path="alessandroptsn/uncategorized/15qco71g").name)
modelwb = load_model(wandb.restore('model_emotions.h5', run_path="alessandroptsn/uncategorized/fmpzwzvv").name)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

color = 0


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



def generate_html_response():
    html_content = """
<!DOCTYPE html>
<html>
   <meta charset="UTF-8">
   <style>
   h2 {text-align: center;}
   p {text-align: center;}
   a {text-align: center;}
   </style>
   <body>
   <section>
    <h2 >Faces Emotions</h2>
     <p><i>A API created by Alessandro Pereira</i></p>
     <p>This project consists of building a neural network model that classifies the emotion that a face shows in a given photo. The neural network consists of multiple ReLU layers and a Softmax layer to classify the emotion. For more information about the work, files and API can be found in the links below:</p>
   <p><a href="https://github.com/AlessandroPTSN/Faces_Emotions">Github</a> <a href="https://medium.com/@alessandro.pereira.700">Mediun</a> <a href="https://faces-emotions.herokuapp.com/docs">API</a> <a href="https://colab.research.google.com/drive/11px62q8u6YapccCGGHS6JgX9dDLWqrVK?usp=sharing">Colab</a> </p>
   </section>
   </body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
#@app.get("/")
#async def say_hello():
#    return {"greeting": "Hello World!"}

@app.get("/", response_class=HTMLResponse)
async def read_items():
    return generate_html_response()


# run the model inference and use a face data structure via POST to the API.
@app.post("/face") 
async def root(file: UploadFile = File(...)):
    img = await  file.read()   
    #prediction = np.around(modelwb.predict(load3(load2(load(img)))), decimals=2)
    prediction = np.around(modelwb.predict(load(img)), decimals=2)
    string = ','.join(str(x) for x in prediction)
    if string == "[1. 0. 0. 0. 0. 0.]":
        result = "Angry"
    if string == "[0. 1. 0. 0. 0. 0.]":
        result = "Fear"
    if string == "[0. 0. 1. 0. 0. 0.]":
        result = "Happy"       
    if string == "[0. 0. 0. 1. 0. 0.]":
        result = "Neutral" 
    if string == "[0. 0. 0. 0. 1. 0.]":
        result = "Sad"
    if string == "[0. 0. 0. 0. 0. 1.]":
        result = "Surprise"        
    return result
