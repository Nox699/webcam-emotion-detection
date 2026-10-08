"""Testable recording controller; keep all classes in each independent session."""
import random
import uuid
from datetime import datetime, timezone

from .config import LABELS, PERSONAL_FEATURES


class CollectionController:
    def __init__(self, samples_per_class=25, interval=0.4):
        if not 10 <= samples_per_class <= 200:
            raise ValueError("Samples per class must be between 10 and 200.")
        self.samples_per_class = samples_per_class
        self.interval = interval
        self.session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.labels = list(LABELS)
        random.Random(self.session).shuffle(self.labels)
        self.phase = 0
        self.count = 0
        self.started_at = None
        self.last_sample = None
        self.rows = []

    @property
    def complete(self):
        return self.phase == len(self.labels)

    @property
    def label(self):
        return self.labels[min(self.phase, len(self.labels)-1)]

    def start(self, timestamp):
        if self.started_at is None and not self.complete:
            self.started_at = timestamp + 2.0
            self.last_sample = None

    def add(self, features, timestamp):
        if self.complete or self.started_at is None or timestamp < self.started_at:
            return False
        if self.last_sample is not None and timestamp - self.last_sample < self.interval:
            return False
        if len(features) != len(PERSONAL_FEATURES):
            raise ValueError("Wrong recording feature dimension.")
        row = dict(zip(PERSONAL_FEATURES, map(float, features)))
        row.update(label=self.label, session=self.session, sample=len(self.rows),
                   captured_at=datetime.now(timezone.utc).isoformat())
        self.rows.append(row)
        self.last_sample = timestamp
        self.count += 1
        if self.count == self.samples_per_class:
            self.phase += 1
            self.count = 0
            self.started_at = None
        return True

    def instruction(self, timestamp):
        if self.complete:
            return "Complete session saved."
        if self.started_at is None:
            return f"Pose a {self.label} expression. Press SPACE when ready."
        if timestamp < self.started_at:
            return f"Hold your expression. Starting in {self.started_at-timestamp:.1f} seconds."
        return "Recording features. Vary your pose slightly; keep facing the camera."
