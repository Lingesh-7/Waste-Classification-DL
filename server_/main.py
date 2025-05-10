from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

from ultralytics import YOLO
import cv2
import tensorflow as tf
import cv2
import time 

app=FastAPI()

MODEL=tf.keras.models.load_model(r'1.keras')
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
        
    model = YOLO("yolov5s.pt")
    # Open the webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Could not access webcam"}
    
    time.sleep(5)

    ret, frame = cap.read()
    cap.release()
    

    if not ret:
        return {"error": "Failed to capture image from webcam"}

    # Run YOLOv5
    results = model(frame)
    result = results[0]
    # cv2.imshow("Object Detection", results[0].plot())
    # time.sleep(5)
    # cv2.destroyAllWindows()

    max_area = 0
    largest_box = None

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)

    if largest_box is None:
        return {"error": "No object detected"}

    x1, y1, x2, y2 = largest_box
    
    cropped_img = result.orig_img[y1:y2, x1:x2]
    cv2.imwrite(r'lar_object.jpg',cropped_img)


    # Resize and normalize for classifier
    resized_img = cv2.resize(cropped_img, (256, 256))
    # normalized = resized_img / 255.0
    cv2.imwrite(r'largest_object.jpg',resized_img)
    input_img = np.expand_dims(resized_img, axis=0)


    # cv2.imshow('Processed image',resized_img)
    # cv2.waitKey(5000)

    prediction = MODEL.predict(input_img)
    index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    predicted_class = CLASS_NAMEs[index]

    return {
        "class": predicted_class,
        "confidence": confidence
    }

    
if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)
