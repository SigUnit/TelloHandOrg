import cv2
import mediapipe as mp
from djitellopy import Tello

def telloTest():
    mpDrawing = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    minimumConfidence = 0.25
    recommendedConfidence = 0.5

    hands = mpHands.Hands(
        min_detection_confidence=minimumConfidence,
        min_tracking_confidence=recommendedConfidence,
    )

    tello = Tello()

    try:
        tello.connect()

        tello.streamon()
        frame_read = tello.get_frame_read()

        while True:
            image = frame_read.frame

            flipCode = 1
            image = cv2.cvtColor(
                cv2.flip(
                    src=image,
                    flipCode=flipCode,
                ), cv2.COLOR_BGR2RGB
            )

            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            if results.multiHandLandmarks:
                for handLandmarks in results.multiHandLandmarks:
                    mpDrawing.draw_landmarks(
                        image,
                        handLandmarks,
                        mpHands.HAND_CONNECTIONS
                    )

            name = "ITI MEDI - Tello Hands"
            cv2.imshow(
                name,
                image
            )

            key = cv2.waitKey(1) & 0xff

            print("For takeoff, touch T")

            match key:
                case 27:  # ESC
                    break
                case ord('t'):
                    tello.takeoff()
                case ord('l'):
                    tello.land()
                case ord('w'):
                    tello.move_forward(30)
                case ord('s'):
                    tello.move_back(30)
                case ord('a'):
                    tello.move_left(30)
                case ord('d'):
                    tello.move_right(30)
                case ord('e'):
                    tello.rotate_clockwise(30)
                case ord('q'):
                    tello.rotate_counter_clockwise(30)
                case ord('r'):
                    tello.move_up(30)
                case ord('f'):
                    tello.move_down(30)
    except Exception as exception:
        print(f"Error: {exception}")

    hands.close()

    tello.land()
