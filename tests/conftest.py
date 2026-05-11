"""
conftest.py — Stubs out ultralytics, tensorflow, torch, and cv2 before
app.py is imported, so the test suite runs in CI without installing any
heavy ML dependencies.

Mock behaviour:
  - YOLO returns one detection with confidence 0.92, box [10, 10, 100, 100]
  - Keras model predicts [0.05, 0.15, 0.80] → class index 2 → "Severe"
  - preprocess_input is an identity function (no-op)
"""

import sys
import numpy as np
from unittest.mock import MagicMock

import pytest


# ── 1. Build the YOLO mock ────────────────────────────────────────────────

def _build_yolo_mock() -> MagicMock:
    """
    Simulates one crash detection.

    app.py calls:
        results = yolo_model(image, verbose=False)   → [detection]
        boxes   = results[0].boxes
        len(boxes)                                    → 1
        boxes.conf.cpu().numpy()                      → np.array([0.92])
        boxes[0].xyxy[0].cpu().numpy().tolist()       → [10,10,100,100]
    """
    # Chain for boxes[0].xyxy[0].cpu().numpy().tolist()
    boxes_mock = MagicMock()
    boxes_mock.__len__.return_value = 1
    boxes_mock.conf.cpu.return_value.numpy.return_value = np.array([0.92])
    boxes_mock.__getitem__.return_value \
        .xyxy.__getitem__.return_value \
        .cpu.return_value \
        .numpy.return_value \
        .tolist.return_value = [10.0, 10.0, 100.0, 100.0]

    detection = MagicMock()
    detection.boxes = boxes_mock

    yolo_instance = MagicMock(return_value=[detection])

    ult = MagicMock()
    ult.YOLO.return_value = yolo_instance
    return ult


# ── 2. Build the TensorFlow / Keras mock ─────────────────────────────────

def _build_tf_mock() -> MagicMock:
    """
    Simulates EfficientNetB0 predicting 'Severe' (index 2).

    app.py calls:
        keras_model = tf.keras.models.load_model(...)
        keras_model.predict(keras_input, verbose=0)   → np.array([[0.05, 0.15, 0.80]])
        tf.keras.applications.efficientnet.preprocess_input(arr)
    """
    keras_model = MagicMock()
    keras_model.predict.return_value = np.array([[0.05, 0.15, 0.80]])

    tf = MagicMock()
    tf.keras.models.load_model.return_value = keras_model
    tf.keras.applications.efficientnet.preprocess_input = lambda arr: arr
    return tf


# ── 3. Inject mocks into sys.modules BEFORE app is imported ──────────────

sys.modules.setdefault("torch",       MagicMock())
sys.modules.setdefault("torchvision", MagicMock())
sys.modules.setdefault("cv2",         MagicMock())
sys.modules["ultralytics"]  = _build_yolo_mock()
sys.modules["tensorflow"]   = _build_tf_mock()


# ── 4. Now it's safe to import the app ───────────────────────────────────

from fastapi.testclient import TestClient  # noqa: E402
from app import app                        # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)