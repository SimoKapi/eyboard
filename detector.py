import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2 as cv
import numpy as np
import time

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks

    for i in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[i]
        for landmark in hand_landmarks:
            cv.circle(annotated_image, (int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])), 10, (0, 255, 0), -1)

    return annotated_image

latest_result = None
latest_frame = None
def write_results(landmarker_result, frame, timestamp):
    global latest_result
    latest_result = landmarker_result


model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = write_results,
    num_hands = 2
)

running = False
def start_capture():
    global running
    global latest_frame
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    running = True

    landmarker = HandLandmarker.create_from_options(options)
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        latest_frame = frame.copy()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        time.sleep(0.01)

    landmarker.close()
    cap.release()
    cv.destroyAllWindows()

def stop_capture():
    global running
    running = False

def get_result():
    return latest_result

def get_frame():
    return latest_frame