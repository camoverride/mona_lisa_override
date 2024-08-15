import os
import sys
import time
from contextlib import contextmanager
import cv2
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort


# Set logging level to error to suppress warnings
ort.set_default_logger_severity(3)
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore",
    category=FutureWarning,
    module="insightface.utils.transform")
warnings.filterwarnings("ignore",
    message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
    category=UserWarning,
    module="onnxruntime.capi.onnxruntime_inference_collection"
)

# Suppress this warning too.
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Suppressing stdout during model preparation
with suppress_stdout():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=False, download_zip=False)


def swap_faces(source_image, target_image):
    """
    Takes a face from a `source_image` and applies it to the `target_image`.
    If there is more than one face in the `source_image` or `target_image,
    the first face is used.

    TODO: Apply `source_image` faces to ALL faces in `target_image`

    Parameters
    ----------
    source_image : np.array
        Image encoded as an array
    target_image : np.array
        Image encoded as an array

    Returns
    -------
    np.array
        The image with the swapped face.
    """
    # Identify Faces
    source_faces = app.get(source_image)
    target_faces = app.get(target_image)

    # Choose one face from each image
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Swap faces
    swapped_face = swapper.get(target_image, target_face, source_face, paste_back=True)

    return swapped_face


if __name__ == "__main__":
    # Load and display the initial background image
    background_image = cv2.imread("test_images/mona_lisa.jpg")
    cv2.imshow("Display Image", background_image)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Timer to track last detected face
    last_face_time = time.time()
    display_face = False

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces in the frame
        faces = app.get(frame)
        
        # If a face is detected, process the image
        if faces:
            # Perform the face swap
            new_image = swap_faces(source_image=frame, target_image=background_image)

            # Display the new image
            cv2.imshow("Display Image", new_image)

            # Update the last face detection time
            last_face_time = time.time()
            display_face = True
        else:
            # If no face is detected for 10 seconds, switch back to background
            if display_face and (time.time() - last_face_time > 10):
                cv2.imshow("Display Image", background_image)
                display_face = False

        # Check for key presses
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
