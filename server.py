
from flask import Flask, request, send_file, jsonify
from contextlib import contextmanager
from PIL import Image

import insightface
from insightface.app import FaceAnalysis

import numpy as np
import onnxruntime as ort

import io
import os
import sys
import threading
import warnings


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ort.set_default_logger_severity(3)

warnings.filterwarnings("ignore", category=FutureWarning)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# --------------------------------------------------
# Load models once
# --------------------------------------------------

print("Loading InsightFace models...")

with suppress_stdout():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        download=False,
        download_zip=False
    )

print("Models loaded successfully.")


# Prevent simultaneous access to the models.
model_lock = threading.Lock()


# --------------------------------------------------
# Flask application
# --------------------------------------------------

server = Flask(__name__)


def decode_image(file):
    """Decode an uploaded image into a NumPy array."""

    image = Image.open(file).convert("RGB")
    return np.array(image)


def encode_image(image):
    """Encode a NumPy image as JPEG bytes."""

    buffer = io.BytesIO()

    Image.fromarray(image).save(
        buffer,
        format="JPEG",
        quality=95
    )

    buffer.seek(0)
    return buffer


def perform_face_swap(source_image, target_image):
    """Perform the actual face swap."""

    with model_lock:

        source_faces = app.get(source_image)
        target_faces = app.get(target_image)

        if len(source_faces) == 0:
            raise ValueError("No face found in source image.")

        if len(target_faces) == 0:
            raise ValueError("No face found in target image.")

        source_face = source_faces[0]
        target_face = target_faces[0]

        result = swapper.get(
            target_image,
            target_face,
            source_face,
            paste_back=True
        )

        return result

@server.route("/swap_faces", methods=["POST"])
def swap_faces_endpoint():

    print("\n--- Incoming face swap request ---")
    print("Content type:", request.content_type)
    print("Files received:", list(request.files.keys()))

    try:
        if "source_image" not in request.files:
            print("Missing source_image")
            return jsonify({"error": "Missing source_image"}), 400

        if "target_image" not in request.files:
            print("Missing target_image")
            return jsonify({"error": "Missing target_image"}), 400

        source_image = decode_image(
            request.files["source_image"]
        )

        target_image = decode_image(
            request.files["target_image"]
        )

        print("Source dimensions:", source_image.shape)
        print("Target dimensions:", target_image.shape)

        result = perform_face_swap(
            source_image,
            target_image
        )

        buffer = encode_image(result)

        print("Face swap successful.")

        return send_file(
            buffer,
            mimetype="image/jpeg",
            download_name="swapped.jpg"
        )

    except Exception as e:
        print("ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 400

# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":

    server.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )
