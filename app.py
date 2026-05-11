import io
import os
import cv2
import base64
import tempfile
import logging
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("severity-api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Traffic Accident Severity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABELS        = ["Minor", "Moderate", "Severe"]
IMG_SIZE      = (224, 224)          # EfficientNetB0 input size
MAX_FILE_MB   = 100                 # hard cap for uploaded files
SEVERITY_RANK = {"Minor": 0, "Moderate": 1, "Severe": 2}

ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp",
    "application/octet-stream",   # some frontends send raw bytes with this
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo",
    "application/octet-stream",
}

# ---------------------------------------------------------------------------
# Model loading  (crashes on startup if files are missing — intentional)
# ---------------------------------------------------------------------------
logger.info("Loading YOLO model …")
yolo_model = YOLO("best.pt")

logger.info("Loading Keras severity model …")
keras_model = tf.keras.models.load_model("severity_model.keras", compile=False)

logger.info("Both models loaded successfully.")


# ---------------------------------------------------------------------------
# Helper: decode image bytes → PIL Image
# Handles raw JPEG/PNG, and base64 (with or without data-URI header)
# ---------------------------------------------------------------------------
def decode_image(raw: bytes) -> Image.Image:
    # --- 1. Try direct PIL decode (JPEG, PNG, WebP, BMP, …) ---
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        logger.info("Image decoded directly via PIL.")
        return img
    except (UnidentifiedImageError, Exception):
        pass

    # --- 2. Try base64  (strip data-URI prefix if present) ---
    try:
        candidate = raw
        if b"base64," in candidate:
            candidate = candidate.split(b"base64,", 1)[1]
        # strip surrounding whitespace / quotes that some clients add
        candidate = candidate.strip().strip(b'"').strip(b"'")
        decoded = base64.b64decode(candidate)
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        logger.info("Image decoded via base64.")
        return img
    except Exception:
        pass

    raise ValueError(
        "Cannot decode the uploaded data. "
        "Accepted formats: JPEG, PNG, WebP, BMP, or a base64-encoded version of any of these."
    )


# ---------------------------------------------------------------------------
# Helper: preprocess PIL Image for EfficientNetB0
# ---------------------------------------------------------------------------
def preprocess_for_keras(pil_image: Image.Image) -> np.ndarray:
    """
    Resize → RGB → EfficientNetB0 preprocess_input (scales to [-1, 1]).
    Returns shape (1, 224, 224, 3).
    """
    img = pil_image.resize(IMG_SIZE, Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------
def run_pipeline(pil_image: Image.Image) -> dict:
    """
    1. YOLO detects crash region (returns no_crash if nothing found).
    2. Crop the highest-confidence box.
    3. EfficientNetB0 classifies severity.
    """
    results = yolo_model(pil_image, verbose=False)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return {
            "status":     "no_crash",
            "severity":   None,
            "confidence": None,
            "box":        None,
        }

    # Pick the detection with the highest YOLO confidence score
    yolo_confs = boxes.conf.cpu().numpy()
    best_idx   = int(np.argmax(yolo_confs))
    box        = boxes[best_idx].xyxy[0].cpu().numpy().tolist()   # [x1, y1, x2, y2]

    # Sanity-check: ensure the crop has positive area
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return {
            "status":     "no_crash",
            "severity":   None,
            "confidence": None,
            "box":        None,
        }

    cropped       = pil_image.crop((x1, y1, x2, y2))
    keras_input   = preprocess_for_keras(cropped)
    predictions   = keras_model.predict(keras_input, verbose=0)   # shape (1, 3)
    class_idx     = int(np.argmax(predictions[0]))
    severity_conf = float(np.max(predictions[0]))

    return {
        "status":     "crash_detected",
        "severity":   LABELS[class_idx],
        "confidence": severity_conf,
        "box":        box,
    }


# ---------------------------------------------------------------------------
# Endpoint: health check
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "Backend is active"}


# ---------------------------------------------------------------------------
# Endpoint: classify a single frame / image
# Accepts:
#   - multipart file upload (JPEG, PNG, WebP, BMP)
#   - raw bytes whose content-type is application/octet-stream
# ---------------------------------------------------------------------------
@app.post("/classify-frame/")
async def classify_frame(file: UploadFile = File(...)):
    # Content-type check (lenient — browsers/frontends are inconsistent)
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{ct}'. Send JPEG, PNG, WebP, or BMP.",
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_MB} MB limit.",
        )

    try:
        image  = decode_image(contents)
        result = run_pipeline(image)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("classify-frame failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")


# ---------------------------------------------------------------------------
# Endpoint: classify from base64 JSON body
# Body: { "image": "<base64 string or data-URI>" }
# ---------------------------------------------------------------------------
@app.post("/classify-base64/")
async def classify_base64(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    if "image" not in body:
        raise HTTPException(status_code=400, detail="Missing key 'image' in JSON body.")

    raw = body["image"]
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=400, detail="'image' must be a non-empty string.")

    try:
        image  = decode_image(raw.encode())
        result = run_pipeline(image)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("classify-base64 failed")
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}")


# ---------------------------------------------------------------------------
# Endpoint: classify an MP4 video
# Samples every `sample_every` frames (default = 30).
# Returns per-frame results + an overall worst-case severity summary.
# ---------------------------------------------------------------------------
@app.post("/classify-video/")
async def classify_video(
    file: UploadFile = File(...),
    sample_every: int = 30,          # process 1 frame every N frames
):
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{ct}'. Send MP4, MOV, or AVI.",
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_MB} MB limit.",
        )

    tmp_path = None
    try:
        # Write to a temp file so OpenCV can open it
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("OpenCV could not open the video. File may be corrupted or unsupported.")

        frame_results = []
        frame_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil   = Image.fromarray(rgb)
                res   = run_pipeline(pil)
                res["frame_index"] = frame_idx
                frame_results.append(res)

            frame_idx += 1

        cap.release()

        if not frame_results:
            return {
                "status":          "no_frames_processed",
                "overall_severity": None,
                "frames_analyzed":  0,
                "frame_results":   [],
            }

        crash_frames = [r for r in frame_results if r["status"] == "crash_detected"]

        if not crash_frames:
            return {
                "status":           "no_crash",
                "overall_severity": None,
                "frames_analyzed":  len(frame_results),
                "crash_count":      0,
                "frame_results":    frame_results,
            }

        # Overall severity = worst frame
        worst = max(crash_frames, key=lambda r: SEVERITY_RANK.get(r["severity"], -1))

        return {
            "status":              "crash_detected",
            "overall_severity":    worst["severity"],
            "overall_confidence":  worst["confidence"],
            "frames_analyzed":     len(frame_results),
            "crash_count":         len(crash_frames),
            "frame_results":       frame_results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("classify-video failed")
        raise HTTPException(status_code=500, detail=f"Video processing error: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/debug-model/")
def debug_model():
    import numpy as np
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    raw = keras_model.predict(dummy, verbose=0)
    return {
        "output_shape":     str(keras_model.output_shape),
        "last_layer":       str(keras_model.layers[-1].get_config()),
        "raw_predictions":  raw.tolist(),
        "predictions_sum":  float(raw.sum()),
    }