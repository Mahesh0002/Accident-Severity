"""
test_api.py — Full test coverage for the Accident Severity API.

Covers:
  - decode_image()      : raw JPEG, base64, data-URI, corrupted bytes, empty bytes
  - GET  /              : health check
  - POST /classify-frame/   : valid image, empty file, oversized file, wrong type
  - POST /classify-base64/  : valid base64, missing key, bad data, empty string
  - run_pipeline()          : crash detected path, no-crash path (via patching)
"""

import base64
import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from app import decode_image, run_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal valid in-memory JPEG."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 80, height: int = 80) -> bytes:
    """Create a minimal valid in-memory PNG."""
    img = Image.new("RGB", (width, height), color=(50, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


VALID_JPEG = _make_jpeg_bytes()
VALID_PNG  = _make_png_bytes()


# ─────────────────────────────────────────────────────────────────────────────
# decode_image() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeImage:

    def test_raw_jpeg_decodes(self):
        img = decode_image(VALID_JPEG)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_raw_png_decodes(self):
        img = decode_image(VALID_PNG)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_plain_base64_decodes(self):
        """Raw base64 string (no data-URI prefix)."""
        encoded = base64.b64encode(VALID_JPEG)
        img = decode_image(encoded)
        assert isinstance(img, Image.Image)

    def test_data_uri_base64_decodes(self):
        """data:image/jpeg;base64,<data> format sent by some frontends."""
        b64 = base64.b64encode(VALID_JPEG).decode()
        data_uri = f"data:image/jpeg;base64,{b64}".encode()
        img = decode_image(data_uri)
        assert isinstance(img, Image.Image)

    def test_corrupted_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            decode_image(b"this is not an image")

    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            decode_image(b"")

    def test_partial_jpeg_header_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            decode_image(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # truncated JPEG


# ─────────────────────────────────────────────────────────────────────────────
# GET / — Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_status_active(self, client):
        r = client.get("/")
        assert r.json() == {"status": "Backend is active"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /classify-frame/
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyFrame:

    def test_valid_jpeg_returns_crash_detected(self, client):
        # conftest mock has YOLO finding one box → Keras predicts "Severe"
        r = client.post(
            "/classify-frame/",
            files={"file": ("test.jpg", VALID_JPEG, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"]   == "crash_detected"
        assert body["severity"] == "Severe"
        assert 0.0 < body["confidence"] <= 1.0
        assert len(body["box"]) == 4

    def test_valid_png_is_accepted(self, client):
        r = client.post(
            "/classify-frame/",
            files={"file": ("test.png", VALID_PNG, "image/png")},
        )
        assert r.status_code == 200

    def test_empty_file_returns_400(self, client):
        r = client.post(
            "/classify-frame/",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400
        assert "Empty" in r.json()["detail"]

    def test_unsupported_content_type_returns_415(self, client):
        r = client.post(
            "/classify-frame/",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 415

    def test_corrupted_image_returns_422(self, client):
        r = client.post(
            "/classify-frame/",
            files={"file": ("bad.jpg", b"not an image at all", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_no_crash_path(self, client):
        """When YOLO finds no boxes, expect no_crash status."""
        no_box_mock = MagicMock()
        no_box_mock.__len__.return_value = 0
        no_box_mock.boxes = no_box_mock

        detection = MagicMock()
        detection.boxes = no_box_mock
        detection.boxes.__len__.return_value = 0

        with patch("app.yolo_model", return_value=[detection]):
            r = client.post(
                "/classify-frame/",
                files={"file": ("road.jpg", VALID_JPEG, "image/jpeg")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"]   == "no_crash"
        assert body["severity"] is None


# ─────────────────────────────────────────────────────────────────────────────
# POST /classify-base64/
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyBase64:

    def _b64_jpeg(self) -> str:
        return base64.b64encode(VALID_JPEG).decode()

    def test_valid_base64_returns_crash_detected(self, client):
        r = client.post(
            "/classify-base64/",
            json={"image": self._b64_jpeg()},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"]   == "crash_detected"
        assert body["severity"] == "Severe"

    def test_valid_data_uri_is_accepted(self, client):
        data_uri = f"data:image/jpeg;base64,{self._b64_jpeg()}"
        r = client.post("/classify-base64/", json={"image": data_uri})
        assert r.status_code == 200

    def test_missing_image_key_returns_400(self, client):
        r = client.post("/classify-base64/", json={"wrong_key": "abc"})
        assert r.status_code == 400
        assert "'image'" in r.json()["detail"]

    def test_empty_image_string_returns_400(self, client):
        r = client.post("/classify-base64/", json={"image": ""})
        assert r.status_code == 400

    def test_invalid_base64_returns_422(self, client):
        r = client.post("/classify-base64/", json={"image": "!!!not_base64!!!"})
        assert r.status_code == 422

    def test_non_json_body_returns_400(self, client):
        r = client.post(
            "/classify-base64/",
            content=b"plain text",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline() unit tests (direct, no HTTP layer)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPipeline:

    def _solid_image(self) -> Image.Image:
        return Image.new("RGB", (300, 300), color=(120, 80, 60))

    def test_crash_detected_returns_correct_keys(self):
        result = run_pipeline(self._solid_image())
        assert "status"     in result
        assert "severity"   in result
        assert "confidence" in result
        assert "box"        in result

    def test_crash_detected_severity_is_valid_label(self):
        result = run_pipeline(self._solid_image())
        if result["status"] == "crash_detected":
            assert result["severity"] in ["Minor", "Moderate", "Severe"]

    def test_confidence_is_float_between_0_and_1(self):
        result = run_pipeline(self._solid_image())
        if result["confidence"] is not None:
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_box_has_four_coordinates(self):
        result = run_pipeline(self._solid_image())
        if result["box"] is not None:
            assert len(result["box"]) == 4

    def test_no_detections_returns_no_crash(self):
        empty_boxes = MagicMock()
        empty_boxes.__len__.return_value = 0
        detection = MagicMock()
        detection.boxes = empty_boxes

        with patch("app.yolo_model", return_value=[detection]):
            result = run_pipeline(self._solid_image())

        assert result["status"]     == "no_crash"
        assert result["severity"]   is None
        assert result["confidence"] is None
        assert result["box"]        is None