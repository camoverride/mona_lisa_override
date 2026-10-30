
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

    try:
        # Validate uploaded files.
        if (
            "source_image" not in request.files
            or "target_image" not in request.files
        ):
            return jsonify({
                "error": "Both images must be provided."
            }), 400

        # Decode images.
        source_image = decode_image(
            request.files["source_image"]
        )

        target_image = decode_image(
            request.files["target_image"]
        )

        # Perform face swapping.
        result = perform_face_swap(
            source_image,
            target_image
        )

        # Return the resulting image.
        buffer = encode_image(result)

        return send_file(
            buffer,
            mimetype="image/jpeg",
            download_name="swapped.jpg"
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print(f"Server error: {e}")

        return jsonify({
            "error": "Face swapping failed."
        }), 500


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
