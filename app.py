import io
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="Traffic Accident Severity API")

# Allow your Lovable/Vercel frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace "*" with your live Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. LOAD MODELS ON STARTUP
# ==========================================

print("Loading YOLOv11 Model...")
try:
    # IMPORTANT: Ensure your YOLO file is named exactly this in your GitHub repo
    yolo_model = YOLO('best.pt') 
except Exception as e:
    print(f"Error loading YOLO: {e}")

print("Loading EfficientNet Keras Model...")
try:
    # IMPORTANT: Ensure your Keras file is named exactly this in your GitHub repo
    keras_model = tf.keras.models.load_model('severity_model.keras', compile=False)
except Exception as e:
    print(f"Error loading Keras model: {e}")

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def preprocess_keras(image):
    """Prepares the cropped PIL Image for EfficientNet-B0 (.keras)"""
    # Resize to standard EfficientNet input size
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: If your Keras model expects inputs scaled from 0-1, uncomment the line below:
    # img_array = img_array / 255.0 
    return img_array

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "Backend is active and waiting for video frames."}

@app.post("/classify-frame/")
async def classify_frame(file: UploadFile = File(...)):
    """Receives a frame, detects a crash, and classifies severity."""
    try:
        # Read the image sent from the React frontend
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # --- Stage 1: YOLO Detection ---
        results = yolo_model(image)
        boxes = results[0].boxes
        
        # If no vehicle/crash is detected in this frame, return early
        if len(boxes) == 0:
            return {"status": "no_crash", "severity": None}
        
        # Get the highest-confidence bounding box [x1, y1, x2, y2]
        box = boxes[0].xyxy[0].tolist() 
        
        # Crop the image to just the detected crash area
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))
        
        # --- Stage 2: Keras Classification ---
        keras_input = preprocess_keras(cropped_img)
        predictions = keras_model.predict(keras_input)
        
        # Extract the highest probability class
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Map integer prediction to text labels
        labels = ["Minor", "Moderate", "Severe"]
        severity = labels[class_idx]
        
        return {
            "status": "crash_detected",
            "severity": severity,
            "confidence": round(confidence, 3),
            "box": [round(c, 2) for c in box] # Return coordinates for frontend visualization
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))