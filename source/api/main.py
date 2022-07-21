from fastapi import FastAPI, UploadFile,File
from io import BytesIO 
import numpy as np
from keras.models import load_model
#import wandb
from PIL import Image
from skimage import transform 
import cv2
from fastapi.responses import HTMLResponse

#modelwb = load_model(wandb.restore('model_emotions.h5', run_path="alessandroptsn/uncategorized/3rm44sap").name)
modelwb = load_model("model_emotions.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

color = 0
result = ""

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
   color=cv2.resize(color,(48,48))
   color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

   imagee = np.expand_dims(transform.resize(np.array(color).astype('float32'), (48, 48, 1)), axis=0)
   color = 0
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
   <p><a href="https://github.com/AlessandroPTSN/Faces_Emotions">Github</a> <a href="https://medium.com/@alessandro.pereira.700">Mediun</a> <a href="https://faces-emotions.herokuapp.com/docs">API</a> <a href="https://colab.research.google.com/drive/1HXjcL0o-oEKmGEvturoNI07PiuZMvW0a?usp=sharing">Colab</a> </p>
   </section>
   </body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/", response_class=HTMLResponse)
async def read_items():
    return generate_html_response()

# run the model inference and use a face data structure via POST to the API.
@app.post("/face") 
async def root(file: UploadFile = File(...)):
    global result
    global color
    color = 0
    result = ""
    img = await  file.read()   
    string = ','.join(str(x) for x in np.around(modelwb.predict(load(img)), decimals=2)
    color = 0
    result = ""
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
