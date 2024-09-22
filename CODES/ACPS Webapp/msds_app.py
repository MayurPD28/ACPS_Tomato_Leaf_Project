import os
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
from ultralytics import YOLO
import base64
import cv2

app = Flask(__name__)

# Load the Inception model
model = tf.keras.models.load_model('MSDS_Iv3.h5')

# Define YOLOv8 model path
YOLO_MODEL_PATH = 'best.pt'

# Initialize YOLO model outside of request handler for better performance
yolo = YOLO(YOLO_MODEL_PATH)

# Define function to analyze tomato leaf using YOLOv8
def analyze_tomato_leaf(image_path):
    results = yolo(image_path)
    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return jsonify({"error": "No file selected"}), 400

        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        try:
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            predictedClass = np.argmax(preds, axis=1)[0]
            
            if predictedClass == 0:
                # Analyze using YOLOv8
                yolo_results = analyze_tomato_leaf(file_path)

                # Convert YOLO results images to base64
                yolo_images = []
                for r in yolo_results:
                    im_bgr = r.plot()
                    _, im_buffer = cv2.imencode('.jpg', im_bgr)
                    im_bytes = im_buffer.tobytes()
                    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
                    yolo_images.append(im_b64)

                result = {"predictedClass": "Diseased", "yolo_images": yolo_images}
            else:
                result = {"predictedClass": str(predictedClass)}

            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid request method"}), 405

if __name__ == '__main__':
    app.run(debug=True)
