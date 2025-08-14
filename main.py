import threading

import configargparse
import cv2 as cv
from djitellopy import Tello

from handler import *
from utils import FrameCalculator


def get_args():
    print("# Reading configuration #")
    parser = configargparse.ArgParser(default_config_files=["config.txt"])

    parser.add_argument(
        "-c",
        "--my-config",
        required=False,
        is_config_file=True,
        help="config file path",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--width", help="cap width", type=int)
    parser.add_argument("--height", help="cap height", type=int)
    parser.add_argument("--is_keyboard", help="To use Keyboard control by default", type=bool)
    parser.add_argument(
        "--use_static_image_mode", action="store_true", help="True if running on photos"
    )
    parser.add_argument(
        "--min_detection_confidence", help="min_detection_confidence", type=float
    )
    parser.add_argument("--min_tracking_confidence", help="min_tracking_confidence", type=float)
    parser.add_argument("--buffer_len", help="Length of gesture buffer", type=int)

    args = parser.parse_args()

    return args


def main():
    global gestureBuffer
    global gestureID
    global batteryStatus

    # Argument parsing
    args = get_args()
    keyboard_control = args.is_keyboard
    write_control = False
    flying = False

    host = ""
    tello = Tello(host=host)
    tello.connect(wait_for_state=False)
    tello.streamon()

    cap = tello.get_frame_read()

    gesture_controller = GestureController(tello)
    keyboard_controller = TelloKeyboardController(tello)

    _gestureDetector = GestureRecognition(
        args.use_static_image_mode,
        args.min_detection_confidence,
        args.min_tracking_confidence,
    )

    gestureBuffer = GestureBuffer(buffer_len=args.buffer_len)

    def tello_control(_key, _keyboardController, _gestureController):
        global gestureBuffer

        if keyboard_control:
            _keyboardController.control(_key)
        else:
            _gestureController.gesture_control(gestureBuffer)

    def tello_battery(tello):
        _batteryStatus: int
        try:
            _batteryStatus = tello.get_battery()
        except Exception as exception:
            print(f"Error: {exception}")

    frameCalculator = FrameCalculator(buffer_len=10)

    mode = 0
    number = -1
    batteryStatus = -1

    tello.send_command_without_return("command")

    while True:
        fps = frameCalculator.get()

        key = cv.waitKey(1) & 0xFF

        if key == 27:  # Esc button.
            break
        elif key == ord("t"):
            tello.takeoff()
            flying = True
        elif key == ord("l"):
            tello.land()
            flying = False
        elif key == ord("k"):  # Move to keyboard mode.
            mode = 0
            keyboard_control = True
            write_control = False
            tello.send_rc_control(0, 0, 0, 0)  # Stop moving
        elif key == ord("g"):  # Move to gesture mode.
            keyboard_control = False
        elif key == ord("n"):
            mode = 1
            write_control = True
            keyboard_control = True

        if write_control:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        image = cap.frame

        debugInfo, gestureID = _gestureDetector.recognize(image, number, mode)
        gestureBuffer.add_gesture(gestureID)

        threading.Thread(
            target=tello_control,
            args=(
                key,
                keyboard_controller,
                gesture_controller,
            ),
        ).start()

        threading.Thread(target=tello_battery, args=(tello,)).start()

        debugInfo = _gestureDetector._drawInfo(debugInfo, fps, mode, number)

        cv.putText(
            debugInfo,
            f"Battery: {batteryStatus}",
            (780, 30),
            cv.QT_FONT_BOLD,
            1,
            (0, 0, 255),
            2,
        )
        cv.imshow("ITI MEDI - TELLO MAIN", debugInfo)

    if flying:
        tello.land()

    cv.destroyAllWindows()
    tello.streamoff()


if __name__ == "__main__":
    main()
