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
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
try:
    CONFIDENCE_THRESHOLD = float(
        os.getenv('ISL_CONFIDENCE_THRESHOLD', DEFAULT_CONFIDENCE_THRESHOLD)
    )
except ValueError:
    CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD
CONFIDENCE_THRESHOLD = min(1.0, max(0.0, CONFIDENCE_THRESHOLD))

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
        # The browser sends a data URL. Validate the protocol before decoding so
        # malformed requests produce a predictable empty result instead of an
        # index error or an ambiguous model failure.
        if not isinstance(image_data, str) or not image_data.startswith('data:image/'):
            return {"prediction": "", "confidence": 0.0}
        header, encoded_data = image_data.split(',', 1)
        if ';base64' not in header or not encoded_data:
            return {"prediction": "", "confidence": 0.0}

        nparr = np.frombuffer(base64.b64decode(encoded_data, validate=True), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return {"prediction": "", "confidence": 0.0}

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)

        label = index_to_label.get(predicted_index, "")
        # Return confidence even when the label is withheld. The UI can then
        # explain uncertainty instead of treating every empty result as a crash.
        return {
            "prediction": label if confidence >= CONFIDENCE_THRESHOLD else "",
            "confidence": round(float(confidence), 3),
        }
    except Exception as e:
        # Keep the process protocol stable even when one frame is malformed.
        return {"prediction": "", "confidence": 0.0}

if __name__ == "__main__":
    input_data = sys.stdin.read()
    if input_data:
        prediction_result = predict_sign(input_data)
        print(json.dumps(prediction_result), end='')