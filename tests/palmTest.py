import cv2 as calc
import mediapipe as mp

def palmTest():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    minimumConfidence = 0.5
    recommendedConfidence = 0.5

    hands = mp_hands.Hands(
        min_detection_confidence=minimumConfidence,
        min_tracking_confidence=recommendedConfidence
    )

    try:
        cap = calc.VideoCapture()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            flipCode = 1
            image = calc.cvtColor(
                calc.flip(
                    src=image,
                    flipCode=flipCode
                ),
                calc.COLOR_BGR2RGB
            )

            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = calc.cvtColor(image, calc.COLOR_RGB2BGR)
            if results.multiHandLandmarks:
                for hand_landmarks in results.multiHandLandmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            calc.imshow('MP Hands Test', image)
            if calc.waitKey(5) & 0xFF == 27:
                break

            hands.close()

            cap.release()
    except Exception as exception:
        print(f"Error: {exception}")



if __name__ == '__main__':
    palmTest()