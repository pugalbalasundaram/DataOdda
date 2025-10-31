import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import threading
import matplotlib.pyplot as plt
from collections import deque

# ✅ Ensure TensorFlow runs on GPU if available, otherwise fall back to CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth to avoid TensorFlow consuming all memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Using GPU for processing.")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("❌ No GPU found. Running on CPU.")

# Load trained GAN generator model
MODEL_PATH = "gan_model.h5"
generator = load_model(MODEL_PATH)

# Define constants
LATENT_DIM = 100
THRESHOLD = 0.5  # Adjust threshold for anomaly detection
FRAME_SIZE = (64, 64)  # Expected GAN input size
ANOMALY_SCORES = deque(maxlen=50)  # Stores the last 50 scores for live graph

# Function to preprocess webcam frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)  # Resize to GAN input size
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(frame, axis=0)

# Function to generate an image using the GAN
def generate_synthetic_image():
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    generated_image = generator.predict(noise)[0]  # Get single image
    return np.clip(generated_image, 0, 1)  # Ensure values are in range [0,1]

# Function to compare real vs generated image and detect anomalies
def detect_anomaly(real_frame):
    synthetic_image = generate_synthetic_image()
    real_frame = np.squeeze(real_frame)  # Remove batch dimension

    # Compute pixel-wise difference
    difference = np.mean(np.abs(real_frame - synthetic_image))
    
    return difference > THRESHOLD, difference

# Function to plot the anomaly score in real-time
def plot_anomaly_score():
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    
    while True:
        if len(ANOMALY_SCORES) > 0:
            ax.clear()
            above_threshold = [score if score > THRESHOLD else None for score in ANOMALY_SCORES]
            below_threshold = [score if score <= THRESHOLD else None for score in ANOMALY_SCORES]
            
            ax.plot(above_threshold, color='red', marker='o', linestyle='dashed', label='Above Threshold')
            ax.plot(below_threshold, color='green', marker='o', linestyle='dashed', label='Below Threshold')
            
            ax.axhline(y=THRESHOLD, color='blue', linestyle='--', linewidth=1.5, label='Anomaly Threshold')
            ax.set_ylim([0.2, 0.8])
            ax.set_title("Anomaly Score Over Time")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Anomaly Score")
            ax.legend()
            plt.pause(0.1)

# Start the real-time graph in a separate thread
graph_thread = threading.Thread(target=plot_anomaly_score, daemon=True)
graph_thread.start()

# Start webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to correct format for processing but keep original for display
    processed_frame = preprocess_frame(frame)

    # Detect anomalies
    is_anomaly, score = detect_anomaly(processed_frame)
    ANOMALY_SCORES.append(score)  # Store anomaly score for graph

    # Set display text and bounding box color
    if is_anomaly:
        text = f"Anomaly Detected! Score: {score:.2f}"
        color = (0, 0, 255)  # RED for anomaly
    else:
        text = f"Normal. Score: {score:.2f}"
        color = (0, 255, 0)  # GREEN for normal

    # Draw bounding box around detected object
    height, width, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (width - 50, height - 50), color, 3)  # Bounding box
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show webcam feed
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
