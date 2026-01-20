import detector
import threading
import cv2

def main():
    capture_thread = threading.Thread(target=detector.start_capture)
    capture_thread.start()

    try:
        while True:
            frame = detector.get_frame()
            result = detector.get_result()
            if frame is not None:
                if result is not None:
                    frame = detector.draw_landmarks_on_image(frame, result)
                cv2.imshow("Frame", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")

    detector.stop_capture()
    capture_thread.join()
    
if __name__ == "__main__":
    main()