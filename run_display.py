import cv2
import os
import time
import yaml
from swap_utils import swap_faces
from utils import get_os_name, rotate_screen



def set_up_display(operating_system : str) -> None:
    """
    Sets the OpenCV display canvas to be fullscreen by creating a
    cv2.namedWindow object addressable as "Display Image".

    Parameters
    ----------
    operating_system : str
        The name of the OS. Current options are:
            - "raspbian"
            - "ubuntu"
            - "macos"

    Returns
    -------
    None
        Creates a fullscreen canvas for displaying images.
    """
    # Raspbian.
    if operating_system == "raspbian":
        # Access the display. TODO: check if still necessary!
        os.environ["DISPLAY"] = ':0'

        # Hide the mouse.
        os.system("unclutter -idle 0 &")

        # Set up the display.
        if operating_system == "raspbian":
            cv2.namedWindow("Display Image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Display Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Ubuntu.
    elif operating_system == "ubuntu":
        # Access the display. TODO: check if still necessary!
        os.environ["DISPLAY"] = ':0'

        time.sleep(5)

        # Hide the mouse
        os.system("unclutter -idle 0 &")

        # Workaround for Wayland: Manually resize the window to fill the screen
        cv2.namedWindow("Display Image", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Display Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # MacOS.
    elif operating_system == "macos":
        cv2.namedWindow("Display Image", cv2.WND_PROP_FULLSCREEN)



if __name__ == "__main__":
    # Open the config to get the rotation and default image.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Get the name of the OS. Should be either "raspbian", "ubuntu", or "macos".
    os_name = get_os_name()

    # Rotate the screen.
    rotate_screen(operating_system=os_name,
                  rotation=config["rotation"])

    # Set up the display to show images.
    set_up_display(operating_system=os_name)


    # Start the camera stream.
    # Raspbian: NOTE: assumes we are using the picam, NOT a webcam!
    if os_name == "raspbian":
        from picamera2 import Picamera2  # type: ignore

        # Initialize the picamera.
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",}))
                                                                    # "size": (WIDTH, HEIGHT)}))
        picam2.start()

    # Ubuntu or MacOS. 
    elif os_name in ["ubuntu", "macos"]:
        # Initialize the cv2 camera.
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly.
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()


    # Set the background image.
    background_image = cv2.imread(f"images/{config['base_image_path']}")

    # Display the default background image.
    cv2.imshow("Display Image", background_image)

    # Timer to track last detected face.
    last_face_time = time.time()
    display_face = False

    # Catch errors to clean up camera resources.
    try:
        # Main event loop.
        while True:
            # Wait time
            time.sleep(0.1)

            # Picam image capture.
            if os_name == "raspbian":
                frame = picam2.capture_array() # type: ignore

            # Webcam image capture.
            if os_name in ["ubuntu", "macos"]:
                ret, frame = cap.read()

                # Check if the frame was captured successfully.
                if not ret:
                    print("Error: Could not capture frame.")
                    exit()

            # Swap faces
            swapped_face = swap_faces(source_image=frame,
                                      target_image=background_image)

            if swapped_face is not None:
                # Display the new image.
                cv2.imshow("Display Image", swapped_face)

                # Update the last face detection time.
                last_face_time = time.time()
                display_face = True

            else:
                # If no face is detected for 10 seconds, switch back to background.
                if display_face and (time.time() - last_face_time > 10):
                    cv2.imshow("Display Image", background_image)
                    display_face = False

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC key
                break

        # Release the camera and close windows
        cv2.destroyAllWindows()
        cap.release()


    # Clean up camera resources.
    finally:
        print("Cleaning up resources...")
        
        # Force destroy all windows first
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process window destruction events
        except:
            pass
        
        # Then cleanup cameras
        if os_name in ["ubuntu", "macos"]:
            try:
                cap.release()
            except:
                pass
                
        if os_name == "raspbian":
            try:
                picam2.stop()
            except:
                pass
        
        # Extra cleanup for good measure
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        print("Cleanup complete")
