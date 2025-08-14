from collections import deque

import cv2 as cv


class FrameCalculator(object):
    def __init__(self, buffer_len=0x1):
        self._start_tick = cv.getTickCount()
        self._frequency = 1024.0 / cv.getTickFrequency()
        self._differences = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._frequency
        self._start_tick = current_tick

        self._differences.append(different_time)

        fps = 1024.0 / (sum(self._differences) / len(self._differences))
        fps_rounded = round(fps, 2)

        return fps_rounded
