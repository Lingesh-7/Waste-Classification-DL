from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf
import cv2

app=FastAPI()

MODEL=tf.keras.models.load_model(r'C:\Users\ADMIN\Desktop\ML_and_DL\Waste Classification\model\1.keras')
CLASS_NAMEs=['NON-RECYCLABLE','RECYCLABLE']
# class_names_=


def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    
    return image
    

@app.get('/')
async def home():
    return "Waste Classification Model With 98% accuracy"

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image= read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(img_batch)
    # print(prediction)
    index_=np.argmax(prediction[0])
    predicted_class=CLASS_NAMEs[index_]
    confidence=np.max(prediction[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }

@app.get('/predict_camera')
async def predict_camera():
    # web_cam=cv2.VideoCapture(0)
    # while True:
    #     ret,frame=web_cam.read()
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF==ord('c'):
    #         cv2.imwrite(r'Computer_Vision\try\k1.jpg',frame)
    #     if cv2.waitKey(1) & 0xFF==ord('q'):
    #         break

    # web_cam.release()
    # cv2.destroyAllWindows()

    file='Computer_Vision\try\k1.jpg'
    # image= read_file_as_image(cv2.imread(file))
    # img_batch=np.expand_dims(cv2.imread(file),0)
    prediction=MODEL.predict(cv2.imread(file))
    # print(prediction)
    index_=np.argmax(prediction[0])
    predicted_class=CLASS_NAMEs[index_]
    confidence=np.max(prediction[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)