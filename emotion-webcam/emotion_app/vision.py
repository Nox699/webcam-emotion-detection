"""Haar face/eye/smile detectors, quality checks, and actual image features."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import HOG_SIZE


@dataclass
class FaceObservation:
    box: tuple[int, int, int, int]
    basic: np.ndarray
    personal: np.ndarray | None
    brightness: float
    sharpness: float
    issue: str | None


class FeatureExtractor:
    def __init__(self):
        def cascade(filename):
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
            if detector.empty():
                raise ValueError(f"Missing OpenCV detector: {filename}. Reinstall opencv-python.")
            return detector
        self.face = cascade("haarcascade_frontalface_default.xml")
        self.eye = cascade("haarcascade_eye_tree_eyeglasses.xml")
        self.smile = cascade("haarcascade_smile.xml")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.hog = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
        if self.hog.getDescriptorSize() != HOG_SIZE:
            raise ValueError("Unexpected OpenCV HOG feature size.")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = self.gray(frame)
        factor = min(1.0, 640 / gray.shape[1])
        resized = cv2.resize(gray, None, fx=factor, fy=factor) if factor < 1 else gray
        boxes = self.face.detectMultiScale(cv2.equalizeHist(resized), scaleFactor=1.15,
            minNeighbors=5, minSize=(max(30, round(80 * factor)),) * 2)
        h, w = gray.shape
        output = []
        for x, y, bw, bh in boxes:
            x, y, bw, bh = [round(int(v) / factor) for v in (x, y, bw, bh)]
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw >= 48 and bh >= 48:
                output.append((x, y, bw, bh))
        return sorted(output, key=lambda box: box[2] * box[3], reverse=True)

    @staticmethod
    def gray(frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.size == 0 or frame.ndim not in (2, 3)):
            raise ValueError("Expected a nonempty uint8 camera image.")
        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("Expected a grayscale or BGR image.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def extract(self, frame, box, personal: bool = False) -> FaceObservation:
        gray = self.gray(frame)
        x, y, w, h = [int(v) for v in box]
        if x < 0 or y < 0 or w < 48 or h < 48 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("Invalid face region.")
        crop = gray[y:y+h, x:x+w]
        brightness = float(crop.mean())
        contrast = float(crop.std())
        # Measure on a consistent scale, independently of the camera's resolution.
        normalized = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
        issue = ("Too dark - add light" if brightness < 35 else
                 "Too bright - reduce glare" if brightness > 225 else
                 "Low contrast - improve lighting" if contrast < 12 else
                 "Blurred - hold still / clean lens" if sharpness < 25 else None)
        equalized = cv2.equalizeHist(normalized)
        eyes = self.eye.detectMultiScale(equalized[:58], scaleFactor=1.1,
            minNeighbors=4, minSize=(12, 12))
        # Rectangle proportions are only a proxy; this is not eyelid landmark geometry.
        openness = np.mean([min(1.0, eh / ew) for _, _, ew, eh in eyes[:2]]) if len(eyes) else 0.35
        smiles = self.smile.detectMultiScale(equalized[48:], scaleFactor=1.15,
            minNeighbors=18, minSize=(22, 10))
        intensity = min(1.0, max((sw for _, _, sw, _ in smiles), default=0) / 76.8)
        basic = np.array([openness, intensity], dtype=float)
        extended = None
        if personal:
            appearance = self.clahe.apply(cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA))
            descriptor = self.hog.compute(appearance).ravel().astype(float)
            extended = np.concatenate((basic, np.clip(descriptor, 0, 1)))
        return FaceObservation((x, y, w, h), basic, extended, brightness, sharpness, issue)


def overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw) - max(ax, bx)) * max(0, min(ay+ah, by+bh) - max(ay, by))
    union = aw*ah + bw*bh - intersection
    return intersection / union if union else 0.0


class ProbabilitySmoother:
    """Reset on lost/changed face; elapsed-time EMA avoids frame-rate dependence."""
    def __init__(self, time_constant: float = 0.35):
        self.time_constant = time_constant
        self.clear()

    def clear(self):
        self.values = None
        self.box = None
        self.timestamp = None

    def update(self, probabilities, box, timestamp):
        values = np.asarray(probabilities, dtype=float)
        if self.values is None or overlap(box, self.box) < 0.3 or timestamp - self.timestamp > 1.0:
            self.values = values.copy()
        else:
            weight = 1 - np.exp(-max(0, timestamp - self.timestamp) / self.time_constant)
            self.values = (1 - weight) * self.values + weight * values
        self.box, self.timestamp = box, timestamp
        return self.values.copy()
