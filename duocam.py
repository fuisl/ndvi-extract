import cv2
import numpy as np
import os
from libcamera import controls
from picamera2 import Picamera2, Preview
from time import sleep
from datetime import datetime


def open_camera(camera_id):
    picam2 = Picamera2(camera_id)
    config = picam2.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2.configure(config)

    picam2.set_controls({"AwbEnable": False})
    picam2.set_controls({"AeEnable": True})
    picam2.set_controls({"AfMetering": controls.AfMeteringEnum.Auto})
    picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
    picam2.set_controls({"AfRange": controls.AfRangeEnum.Macro})

    picam2.start()
    return picam2


def get_frame(picam2):
    frame = picam2.capture_array()
    return frame


def save_images(frame1, frame2):
    now = datetime.now()
    timestamp = now.strftime("%d%m%y_%H%M")

    filename_rgb = f"RGB/{timestamp}_RGB.jpg"
    filename_nir = f"NIR/{timestamp}_NIR.jpg"

    cv2.imwrite(filename_rgb, frame1)
    cv2.imwrite(filename_nir, frame2)

    print(f"Saved images:\n  RGB -> {filename_rgb}\n  NIR -> {filename_nir}")


def main():
    os.makedirs("NIR", exist_ok=True)
    os.makedirs("RGB", exist_ok=True)

    try:
        # Open two cameras
        picam2_1 = open_camera(0)
        picam2_2 = open_camera(1)

        while True:
            # Capture frames from both cameras
            frame1 = get_frame(picam2_1)
            frame2 = get_frame(picam2_2)

            # Combine the two frames side by side
            combined_frame = cv2.hconcat([frame1, frame2])

            # Display the combined frame
            cv2.imshow("Dual Camera Feed", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space key pressed
                save_images(frame1, frame2)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the cameras and close the window
        picam2_1.stop()
        picam2_2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
