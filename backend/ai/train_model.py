import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- CONFIGURATION ---
DATA_PATH = os.path.join('dataset')
MODEL_SAVE_PATH = os.path.join('isl_model.h5')
LABEL_MAP_SAVE_PATH = os.path.join('label_map.json')
IMG_SIZE = 64

# --- 1. LOAD DATA ---
images = []
labels = []
label_map = {}
current_label = 0

print("Loading images from dataset...")
# Make sure we are in the 'ai' directory's context
full_data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)

for filename in os.listdir(full_data_path):
    if filename.endswith('.jpg'):
        label_name = filename.split('_')[0]
        
        if label_name not in label_map:
            label_map[label_name] = current_label
            current_label += 1
            
        img_path = os.path.join(full_data_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_map[label_name])

print(f"Loaded {len(images)} images for {len(label_map)} labels.")
print(f"Labels found: {list(label_map.keys())}")

with open(LABEL_MAP_SAVE_PATH, 'w') as f:
    json.dump(label_map, f)
print(f"Label map saved to {LABEL_MAP_SAVE_PATH}")

# --- 2. PREPARE DATA FOR TRAINING ---
X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(np.array(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. DEFINE AND TRAIN THE CNN MODEL ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nTraining model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# --- 4. SAVE THE TRAINED MODEL ---
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")