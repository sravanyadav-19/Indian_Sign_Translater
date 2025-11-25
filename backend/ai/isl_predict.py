import sys
import numpy as np
import cv2
import base64
import json
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- CONFIGURATION ---
IMG_SIZE = 64
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'isl_model.h5')
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'label_map.json')

# --- LOAD MODEL AND LABELS ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    index_to_label = {v: k for k, v in label_map.items()}
except Exception as e:
    print(f"Error loading model or label map: {e}", file=sys.stderr)
    sys.exit(1)

# --- PREDICTION FUNCTION ---
def predict_sign(image_data):
    try:
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return ""

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.85: # High confidence threshold for better accuracy
            return index_to_label.get(predicted_index, "")
        else:
            return ""
    except Exception as e:
        # Don't print errors for every bad frame, just return empty
        return ""

if __name__ == "__main__":
    input_data = sys.stdin.read()
    if input_data:
        prediction_result = predict_sign(input_data)
        print(prediction_result, end='')