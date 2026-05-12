from __future__ import annotations

import cv2
import numpy as np


class FacePrivacyFilter:
    """
    Optional face blurring for privacy-safe demos.

    This detects face-like regions and blurs them. It does not recognize identity,
    compare faces, store biometrics, or label a person by name.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._cascade = None

    def _load(self) -> None:
        if self._cascade is not None:
            return
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)

    def detect_face_like_regions(self, frame: np.ndarray):
        self._load()
        if self._cascade is None or self._cascade.empty():
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    def has_face_like_region(self, frame: np.ndarray) -> bool:
        return len(self.detect_face_like_regions(frame)) > 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        faces = self.detect_face_like_regions(frame)
        for x, y, w, h in faces:
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            k = max(15, (min(w, h) // 2) | 1)
            frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame
