# ============================================================
# app.py — Flask REST API for Fetal Image Analysis
# ============================================================
# Project : Fetal Image Analysis Using Deep Learning
# Author  : TB Solutions
# Phase   : 3 — Inference Pipeline (API Server)
#
# Endpoints:
#   POST /predict           — image file → JSON prediction
#   POST /predict/base64    — base64 JSON → JSON prediction
#   POST /predict/annotated — image file → annotated JPEG
#   GET  /health            — health check
#   GET  /classes           — list of class names
#
# Usage:
#   python app.py
#   → Server runs at http://localhost:5000
# ============================================================

import os
import io
import time
import logging
from datetime import datetime

from flask import (
    Flask, request, jsonify, send_file
)
from flask_cors import CORS
from PIL import Image

# Import our inference module
from inference import (
    load_model, get_model_info, predict,
    preprocess, annotate_image
)

# ============================================================
# CONFIGURATION
# ============================================================

# Model path — configurable via environment variable
MODEL_PATH = os.environ.get("FETAL_MODEL_PATH", "best.pt")

# Server settings
HOST = os.environ.get("FETAL_HOST", "0.0.0.0")
PORT = int(os.environ.get("FETAL_PORT", 5000))

# Limits
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Enable CORS for Antigravity frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Load model at startup (thread-safe, loaded once) ----
logger.info(f"Loading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    model_info = get_model_info(model)
    logger.info(f"Model loaded: {model_info['model_version']} | "
                f"{model_info['num_classes']} classes | "
                f"Device: {model_info['device']}")
except Exception as e:
    logger.error(f"FATAL: Failed to load model: {e}")
    model = None
    model_info = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check if file extension is in the allowed list."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_image_request():
    """
    Validate incoming image upload request.

    Returns:
        tuple: (pil_image, error_response)
            If valid: (PIL.Image, None)
            If invalid: (None, Flask response)
    """
    if "image" not in request.files:
        return None, jsonify({
            "error": "No image file provided",
            "detail": "Send a file with key 'image' in multipart/form-data"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return None, jsonify({
            "error": "Empty filename",
            "detail": "The uploaded file has no name"
        }), 400

    if not allowed_file(file.filename):
        return None, jsonify({
            "error": "Invalid file type",
            "detail": f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            "received": file.filename,
        }), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        return img, None
    except Exception as e:
        return None, jsonify({
            "error": "Could not read image",
            "detail": str(e),
        }), 400


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    Input: multipart/form-data with 'image' file
    Output: JSON prediction result
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    img, error = validate_image_request()
    if error:
        return error

    start_time = time.time()

    try:
        result = predict(img, model)
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict | "
            f"{request.files['image'].filename} | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e),
        }), 500


@app.route("/predict/base64", methods=["POST"])
def predict_base64_endpoint():
    """
    POST /predict/base64
    Input: JSON {"image": "<base64-encoded-string>"}
    Output: JSON prediction result
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({
            "error": "No image data provided",
            "detail": "Send JSON with key 'image' containing base64 string"
        }), 400

    start_time = time.time()

    try:
        b64_string = data["image"]
        result = predict(b64_string, model)
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict/base64 | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Base64 prediction error: {e}")
        return jsonify({
            "error": "Prediction failed",
            "detail": str(e),
        }), 500


@app.route("/predict/annotated", methods=["POST"])
def predict_annotated_endpoint():
    """
    POST /predict/annotated
    Input: multipart/form-data with 'image' file
    Output: Annotated JPEG image
            Prediction JSON in X-Prediction-* response headers
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    img, error = validate_image_request()
    if error:
        return error

    start_time = time.time()

    try:
        result = predict(img, model)
        annotated = annotate_image(img, result)

        # Save annotated image to bytes
        img_buffer = io.BytesIO()
        annotated.save(img_buffer, format="JPEG", quality=95)
        img_buffer.seek(0)

        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"POST /predict/annotated | "
            f"{request.files['image'].filename} | "
            f"Pred: {result['top_prediction']} "
            f"({result['confidence']*100:.1f}%) | "
            f"{elapsed:.0f}ms"
        )

        response = send_file(
            img_buffer,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="annotated_result.jpg"
        )

        # Add prediction data as custom headers
        response.headers["X-Prediction-Class"] = result["top_prediction"]
        response.headers["X-Prediction-Confidence"] = str(result["confidence"])
        response.headers["X-Prediction-Status"] = result["status"]
        response.headers["X-Processing-Time-Ms"] = str(result["processing_time_ms"])

        return response

    except Exception as e:
        logger.error(f"Annotated prediction error: {e}")
        return jsonify({
            "error": "Annotated prediction failed",
            "detail": str(e),
        }), 500


@app.route("/health", methods=["GET"])
def health_endpoint():
    """
    GET /health
    Output: Server and model health status
    """
    status = {
        "status": "ok" if model is not None else "error",
        "model": "loaded" if model is not None else "not loaded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }

    if model_info:
        status["classes"] = model_info["classes"]
        status["num_classes"] = model_info["num_classes"]
        status["device"] = model_info["device"]
        status["model_version"] = model_info["model_version"]

    return jsonify(status), 200 if model else 503


@app.route("/classes", methods=["GET"])
def classes_endpoint():
    """
    GET /classes
    Output: List of all class names the model can predict
    """
    if model_info is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({
        "classes": model_info["classes"],
        "count": model_info["num_classes"],
    }), 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(e):
    """Handle files exceeding MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": "File too large",
        "detail": f"Maximum file size is {MAX_CONTENT_LENGTH // (1024*1024)} MB",
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle unknown endpoints."""
    return jsonify({
        "error": "Endpoint not found",
        "detail": "Available endpoints: /predict, /predict/base64, "
                  "/predict/annotated, /health, /classes",
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle wrong HTTP method."""
    return jsonify({
        "error": "Method not allowed",
        "detail": str(e),
    }), 405


@app.errorhandler(500)
def internal_error(e):
    """Handle unexpected server errors."""
    logger.error(f"Internal error: {e}")
    return jsonify({
        "error": "Internal server error",
        "detail": "An unexpected error occurred. Check server logs.",
    }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🩺 Fetal Image Analysis — API Server")
    print("=" * 60)
    print(f"  Host   : {HOST}")
    print(f"  Port   : {PORT}")
    print(f"  Model  : {MODEL_PATH}")

    if model_info:
        print(f"  Task   : {model_info['task']}")
        print(f"  Classes: {model_info['num_classes']}")
        print(f"  Device : {model_info['device']}")
    else:
        print("  ⚠️  Model failed to load!")

    print("=" * 60)
    print(f"  Endpoints:")
    print(f"    POST /predict           — image → JSON")
    print(f"    POST /predict/base64    — base64 → JSON")
    print(f"    POST /predict/annotated — image → annotated JPEG")
    print(f"    GET  /health            — health check")
    print(f"    GET  /classes           — class list")
    print("=" * 60)

    app.run(host=HOST, port=PORT, debug=False)
