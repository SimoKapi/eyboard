import detector
import threading
import cv2

KEYBOARD_OFFSET = (-35, 50)
KEYBOARD_DIMS = (400, 170)

class Key:
    pressed = False
    height, width = None, None
    color = (0, 0, 255)

    def __init__(self, letter: str):
        self.letter = letter

    def press(self):
        if not self.pressed:
            print(self.letter)
        self.pressed = True

    def release(self):
        self.pressed = False

class Keyboard:
    template = [
        "QWERTYUIOP",
        "ASDFGHJKL",
        "ZXCVBNM"
    ]
    key_gap = 5

    last_pressed = None

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._generate()

    def _generate(self):
        self.keys = [
            [Key(letter) for letter in row] for row in self.template
        ]
        key_height = round((self.height - (len(self.keys) + 1) * self.key_gap) / len(self.keys))
        for row in range(len(self.keys)):
            key_width = round((self.width - (len(self.keys[row]) + 1) * self.key_gap) / len(self.keys[row]))
            for key in self.keys[row]:
                key.height = key_height
                key.width = key_width

    def draw(self, image, x, y):
        cv2.rectangle(image, (x, y), (x + self.width, y - self.height), (255, 0, 0), -1)
        
        for row in range(len(self.keys)):
            for col in range(len(self.keys[row])):
                key = self.keys[row][col]
                position = (x + self.key_gap + col * (key.width + self.key_gap),
                            y - self.height + self.key_gap + row * (key.height + self.key_gap))
                cv2.rectangle(image, position, (position[0] + key.width, position[1] + key.height), key.color, -1)
                cv2.putText(image, key.letter, (position[0], position[1] + key.height - self.key_gap), 2, 1, (255, 255, 0), 2)

    def press_key(self, rel_x, rel_y):
        if abs(rel_x) > 1 or abs(rel_y) > 1: return
        row = round((1-rel_y) * (len(self.keys) - 1))
        col = round(rel_x * (len(self.keys[row]) - 1))
        # print(rel_x, rel_y)
        # print(col, row)
        # print("*****")
        key = self.keys[row][col]
        key.color = (0, 255, 255)
        # self.last_pressed = key
        # key.press()

default_z = None
def get_keyboard_origin(landmarker_result):
    global default_z
    frame = detector.get_frame()
    origin_index = 17 # Pinky base
    try:
        keyboard_origin = landmarker_result.hand_landmarks[0][origin_index]
    except IndexError as e:
        return

    default_z = keyboard_origin.z
    return (round(keyboard_origin.x * frame.shape[1] + KEYBOARD_OFFSET[0]),
            round(keyboard_origin.y * frame.shape[0] + KEYBOARD_OFFSET[1]))

def get_finger_keyboard_position(finger_position, keyboard_origin):
    x = (finger_position[0] - keyboard_origin[0])/KEYBOARD_DIMS[0]
    y = (keyboard_origin[1] - finger_position[1])/KEYBOARD_DIMS[1]
    return (x, y)

keyboard = Keyboard(KEYBOARD_DIMS[0], KEYBOARD_DIMS[1])
def handle_keyboard(landmarker_result):
    origin = get_keyboard_origin(landmarker_result)
    frame = detector.get_frame()
    if origin:
        keyboard.draw(frame, *origin)

        #right
        index_r = landmarker_result.hand_landmarks[0][8]
        index_r_x = round(index_r.x * frame.shape[1])
        index_r_y = round(index_r.y * frame.shape[0])
        index_r_position = get_finger_keyboard_position((index_r_x, index_r_y), origin)
        keyboard.press_key(*index_r_position)
        if abs(index_r.z - default_z) > 0.033:
            color = (255, 255, 255)
        else:
            color = (0, 255, 0)
        cv2.circle(frame, (index_r_x, index_r_y), 10, color, -1)

        if len(landmarker_result.hand_landmarks) >= 2:
            index_l = landmarker_result.hand_landmarks[1][8]
            index_l_x = round(index_l.x * frame.shape[1])
            index_l_y = round(index_l.y * frame.shape[0])
            index_l_position = get_finger_keyboard_position((index_l_x, index_l_y), origin)
            print(index_l_position)
            keyboard.press_key(*index_l_position)
            if abs(index_l.z - default_z) > 0.033:
                color = (255, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.circle(frame, (index_l_x, index_l_y), 10, color, -1)

    # frame = detector.draw_landmarks_on_image(frame, landmarker_result)
    cv2.imshow("Frame", frame)


def main():
    capture_thread = threading.Thread(target=detector.start_capture)
    capture_thread.start()

    while True:
        frame = detector.get_frame()
        result = detector.get_result()
        if frame is not None:
            if result is not None:
                handle_keyboard(result)

        if cv2.waitKey(1) == ord('q'):
            break

    detector.stop_capture()
    capture_thread.join()
    
if __name__ == "__main__":
    main()