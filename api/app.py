import cv2
import numpy as np
import datetime
import os
import tempfile
import pyrebase
import tensorflow as tf
from flask import Flask, request
from paddleocr import PaddleOCR

app = Flask(__name__)

model = tf.saved_model.load('/home/manas.pal/capstone/final.pb')

import firebase_config

firebase = pyrebase.initialize_app(firebase_config.config)
db = firebase.database()
storage = firebase.storage()

@app.route('/ping', methods=['GET'])
def ping():
    return {"message": "pong"}, 200
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return process_and_store(img)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_license_plate(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(image, cls=True)
    predictions = []
    for line in results:
        for (_, (text, score)) in line:
            if score > 0.5:
                predictions.append((text, score))
    if predictions:
        best_text = max(predictions, key=lambda x: x[1])[0]
        return best_text.strip()
    else:
        return ""

def infer(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]  # Add batch dimension
    
    detections = model(input_tensor)

    return detections

def process_and_store(img):
    results = []
    timestamp = int(datetime.datetime.now().timestamp())
    today_date = datetime.datetime.now().strftime("%d-%m-%Y")
    
    detections = infer(model, img)
    
    if detections['num_detections'] == 0:
        return {"message": "No license plates detected"}, 200

    for i in range(detections['num_detections']):
        detection = detections['detection_boxes'][i].numpy()
        score = detections['detection_scores'][i].numpy()
        
        if score > 0.5:
            x1, y1, x2, y2 = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = img[y1:y2, x1:x2]

            if cropped is None or cropped.size == 0:
                continue

            license_plate = ocr_license_plate(cropped)
            image_name = f"{timestamp}.jpg"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, cropped)

            storage.child(f"{today_date}/{image_name}").put(temp_path)
            image_url = storage.child(f"{today_date}/{image_name}").get_url(None)

            db.child(today_date).push({
                "license_plate": license_plate,
                "date": timestamp,
                "img_Url": image_url
            })

            os.remove(temp_path)

            results.append({
                "license_plate": license_plate,
                "image_url": image_url
            })

    return {"plates_detected": results}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
