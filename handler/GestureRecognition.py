import copy
import csv
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as media
import numpy as num

from model import KeyPointClassifier
from model import PointHistoryClassifier


class GestureRecognition:
    def __init__(
            self,
            use_static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            history_length=16,
    ):
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.history_length = history_length

        # Load models
        (
            self.hands,
            self.keypoint_classifier,
            self.keypoint_classifier_labels,
            self.point_history_classifier,
            self.point_history_classifier_labels,
        ) = self.loadModel()

        # Finger gesture history
        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)

    def loadModel(self):
        # Model load
        mediaHands = media.solutions.hands
        hands = mediaHands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # Read labels
        with open(
                "model/keypoint_classifier/keypointClassifierLabel.csv",
                encoding="utf-8-sig",
        ) as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
        with open(
                "model/point_history_classifier/pointHistoryClassifierLabel.csv",
                encoding="utf-8-sig",
        ) as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        return (
            hands,
            keypoint_classifier,
            keypoint_classifier_labels,
            point_history_classifier,
            point_history_classifier_labels,
        )

    def recognize(self, image, number=-1, mode=0):

        # TODO: Move constants to other place
        USE_BRECT = True

        image = cv.flip(image, 1)  # Mirror display
        debugImage = copy.deepcopy(image)

        # Saving gesture id for drone controlling
        gestureID = -1

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
            ):
                # Bounding box calculation
                boundRect = self._calculateBoundRect(debugImage, hand_landmarks)
                # Landmark calculation
                landmarkList = self._calculateLandmarkList(debugImage, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                preProcessedLandmarkList = self._preProcessLandmark(landmarkList)
                preProcessedPointHistoryList = self._preProcessPointHistory(
                    debugImage, self.point_history
                )

                # Write to the dataset file
                self._loggingCSV(
                    number, mode, preProcessedLandmarkList, preProcessedPointHistoryList
                )

                # Hand sign classification
                handSignID = self.keypoint_classifier(preProcessedLandmarkList)
                if handSignID == 2:  # Point gesture
                    self.point_history.append(landmarkList[8])
                else:
                    self.point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(preProcessedPointHistoryList)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        preProcessedPointHistoryList
                    )

                # Calculates the gesture IDs in the latest detection
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                # Drawing part
                debugImage = self._drawBoundRect(USE_BRECT, debugImage, boundRect)
                debugImage = self._drawLandmarks(debugImage, landmarkList)
                debugImage = self._drawInfoTest(
                    debugImage,
                    boundRect,
                    handedness,
                    self.keypoint_classifier_labels[handSignID],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                # Saving gesture
                gestureID = handSignID
        else:
            self.point_history.append([0, 0])

        debugImage = self.drawPointHistory(debugImage, self.point_history)

        return debugImage, gestureID

    def drawPointHistory(self, image, pointHistory):
        for index, point in enumerate(pointHistory):
            if point[0] != 0 and point[1] != 0:
                cv.circle(
                    image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
                )

        return image

    def _drawInfo(self, image, fps, mode, number):
        cv.putText(
            image,
            "FPS:" + str(fps),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
            )

        cv.putText(
            image,
            "FPS:" + str(fps),
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
            )

        modeString = ["Logging Key Point", "Logging Point History"]

        if 1 <= mode <= 2:
            cv.putText(
                image,
                "MODE:" + modeString[mode - 1],
                (10, 90),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
                )

            if 0 <= number <= 9:
                cv.putText(
                    image,
                    "NUM:" + str(number),
                    (10, 110),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                    )
        return image

    def _loggingCSV(self, number, mode, landmarkList, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            print("WRITE")
            csvPath = "model/keypoint_classifier/keypoint.csv"
            with open(csvPath, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmarkList])
        if mode == 2 and (0 <= number <= 9):
            csvPath = "model/point_history_classifier/pointHistory.csv"
            with open(csvPath, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def _calculateBoundRect(self, image, landmarks):
        imageWidth, imageHeight = image.shape[1], image.shape[0]

        landmarkArray = num.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmarkX = min(int(landmark.x * imageWidth), imageWidth - 1)
            landmarkY = min(int(landmark.y * imageHeight), imageHeight - 1)

            landmarkPoint = [num.array((landmarkX, landmarkY))]

            landmarkArray = num.append(landmarkArray, landmarkPoint, axis=0)

        x, y, w, h = cv.boundingRect(landmarkArray)

        return [x, y, x + w, y + h]

    def _calculateLandmarkList(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmarkPoint = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmarkPoint.append([landmark_x, landmark_y])

        return landmarkPoint

    def _preProcessLandmark(self, landmarkList):
        tempLandmarkList = copy.deepcopy(landmarkList)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(tempLandmarkList):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            tempLandmarkList[index][0] = tempLandmarkList[index][0] - base_x
            tempLandmarkList[index][1] = tempLandmarkList[index][1] - base_y

        # Convert to a one-dimensional list
        tempLandmarkList = list(itertools.chain.from_iterable(tempLandmarkList))

        # Normalization
        max_value = max(list(map(abs, tempLandmarkList)))

        def normalize_(n):
            return n / max_value

        tempLandmarkList = list(map(normalize_, tempLandmarkList))

        return tempLandmarkList

    def _preProcessPointHistory(self, image, pointHistory):
        image_width, image_height = image.shape[1], image.shape[0]

        tempPointHistory = copy.deepcopy(pointHistory)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(tempPointHistory):
            if index == 0:
                base_x, base_y = point[0], point[1]

            tempPointHistory[index][0] = (
                                                 tempPointHistory[index][0] - base_x
                                           ) / image_width
            tempPointHistory[index][1] = (
                                                 tempPointHistory[index][1] - base_y
                                           ) / image_height

        # Convert to a one-dimensional list
        tempPointHistory = list(itertools.chain.from_iterable(tempPointHistory))

        return tempPointHistory

    def _drawLandmarks(self, image, landmarkPoint):
        if len(landmarkPoint) > 0:
            # Thumb
            cv.line(
                image, tuple(landmarkPoint[2]), tuple(landmarkPoint[3]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[2]),
                tuple(landmarkPoint[3]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[3]), tuple(landmarkPoint[4]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[3]),
                tuple(landmarkPoint[4]),
                (255, 255, 255),
                2,
            )

            # Index finger
            cv.line(
                image, tuple(landmarkPoint[5]), tuple(landmarkPoint[6]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[5]),
                tuple(landmarkPoint[6]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[6]), tuple(landmarkPoint[7]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[6]),
                tuple(landmarkPoint[7]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[7]), tuple(landmarkPoint[8]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[7]),
                tuple(landmarkPoint[8]),
                (255, 255, 255),
                2,
            )

            # Middle finger
            cv.line(
                image, tuple(landmarkPoint[9]), tuple(landmarkPoint[10]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[9]),
                tuple(landmarkPoint[10]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmarkPoint[10]),
                tuple(landmarkPoint[11]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmarkPoint[10]),
                tuple(landmarkPoint[11]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmarkPoint[11]),
                tuple(landmarkPoint[12]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmarkPoint[11]),
                tuple(landmarkPoint[12]),
                (255, 255, 255),
                2,
            )

            # Ring finger
            cv.line(
                image,
                tuple(landmarkPoint[13]),
                tuple(landmarkPoint[14]),
                (0, 0, 0),
                6,
            )

            cv.line(
                image,
                tuple(landmarkPoint[13]),
                tuple(landmarkPoint[14]),
                (255, 255, 255),
                2,
            )

            cv.line(
                image,
                tuple(landmarkPoint[14]),
                tuple(landmarkPoint[15]),
                (0, 0, 0),
                6,
            )

            cv.line(
                image,
                tuple(landmarkPoint[14]),
                tuple(landmarkPoint[15]),
                (255, 255, 255),
                2,
            )

            cv.line(
                image,
                tuple(landmarkPoint[15]),
                tuple(landmarkPoint[16]),
                (0, 0, 0),
                6,
            )

            cv.line(
                image,
                tuple(landmarkPoint[15]),
                tuple(landmarkPoint[16]),
                (255, 255, 255),
                2,
            )

            # Little finger
            cv.line(
                image,
                tuple(landmarkPoint[17]),
                tuple(landmarkPoint[18]),
                (0, 0, 0),
                6,
            )

            cv.line(
                image,
                tuple(landmarkPoint[17]),
                tuple(landmarkPoint[18]),
                (255, 255, 255),
                2,
            )

            cv.line(
                image,
                tuple(landmarkPoint[18]),
                tuple(landmarkPoint[19]),
                (0, 0, 0),
                6,
            )

            cv.line(
                image,
                tuple(landmarkPoint[18]),
                tuple(landmarkPoint[19]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmarkPoint[19]),
                tuple(landmarkPoint[20]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmarkPoint[19]),
                tuple(landmarkPoint[20]),
                (255, 255, 255),
                2,
            )

            # Palm
            cv.line(
                image, tuple(landmarkPoint[0]), tuple(landmarkPoint[1]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[0]),
                tuple(landmarkPoint[1]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[1]), tuple(landmarkPoint[2]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[1]),
                tuple(landmarkPoint[2]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[2]), tuple(landmarkPoint[5]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[2]),
                tuple(landmarkPoint[5]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[5]), tuple(landmarkPoint[9]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[5]),
                tuple(landmarkPoint[9]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[9]), tuple(landmarkPoint[13]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[9]),
                tuple(landmarkPoint[13]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image,
                tuple(landmarkPoint[13]),
                tuple(landmarkPoint[17]),
                (0, 0, 0),
                6,
            )
            cv.line(
                image,
                tuple(landmarkPoint[13]),
                tuple(landmarkPoint[17]),
                (255, 255, 255),
                2,
            )
            cv.line(
                image, tuple(landmarkPoint[17]), tuple(landmarkPoint[0]), (0, 0, 0), 6
            )
            cv.line(
                image,
                tuple(landmarkPoint[17]),
                tuple(landmarkPoint[0]),
                (255, 255, 255),
                2,
            )

        # Key Points
        for index, landmark in enumerate(landmarkPoint):
            if index == 0:  # Wrist 1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # Wrist 2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # Index finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # Middle finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: point first
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # Ring finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # Little finger: base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: point first
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def _drawInfoTest(
            self, image, rect, handedness, handSignText, fingerGestureText
    ):
        cv.rectangle(
            image, (rect[0], rect[1]), (rect[2], rect[1] - 22), (0, 0, 0), -1
        )

        info_text = handedness.classification[0].label[0:]
        if handSignText != "":
            info_text = info_text + ":" + handSignText
        cv.putText(
            image,
            info_text,
            (rect[0] + 5, rect[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        return image

    def _drawBoundRect(self, useRect, image, rect):
        if useRect:
            # Outer rectangle
            cv.rectangle(
                image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), 1
            )

        return image


class GestureBuffer:
    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len
        self._buffer = deque(maxlen=buffer_len)

    def add_gesture(self, gesture_id):
        self._buffer.append(gesture_id)

    def get_gesture(self):
        counter = Counter(self._buffer).most_common()
        if counter[0][1] >= (self.buffer_len - 1):
            self._buffer.clear()
            return counter[0][0]
        else:
            return
