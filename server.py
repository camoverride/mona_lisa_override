from flask import Flask, request, send_file, jsonify
from PIL import Image

import insightface
from insightface.app import FaceAnalysis

import numpy as np
import onnxruntime as ort

import io
import os
import threading


ort.set_default_logger_severity(3)

server = Flask(__name__)

# Load models once.
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(
    "inswapper_128.onnx",
    download=False,
    download_zip=False
)

model_lock = threading.Lock()


@server.route("/swap_faces", methods=["POST"])
def swap_faces_endpoint():

    try:
        if "source_image" not in request.files:
            return jsonify({
                "error": "Missing source_image"
            }), 400

        background_path = request.form.get(
            "background_image_path"
        )

        if not background_path:
            return jsonify({
                "error": "Missing background_image_path"
            }), 400

        if not os.path.isfile(background_path):
            return jsonify({
                "error": "Background image does not exist"
            }), 400

        # Load source image.
        source_image = np.array(
            Image.open(
                request.files["source_image"]
            ).convert("RGB")
        )

        # Load background image from server filesystem.
        target_image = np.array(
            Image.open(background_path).convert("RGB")
        )

        with model_lock:

            source_faces = app.get(source_image)
            target_faces = app.get(target_image)

            if not source_faces:
                return jsonify({
                    "error": "No face in source image"
                }), 400

            if not target_faces:
                return jsonify({
                    "error": "No face in background image"
                }), 400

            result = swapper.get(
                target_image,
                target_faces[0],
                source_faces[0],
                paste_back=True
            )

        buffer = io.BytesIO()

        Image.fromarray(result).save(
            buffer,
            format="JPEG",
            quality=95
        )

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="image/jpeg"
        )

    except Exception as e:
        print("Server error:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    server.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )