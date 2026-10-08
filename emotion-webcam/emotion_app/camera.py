"""Camera and video lifecycles, inference, collection, and hardware diagnostics."""
import logging
import platform
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .collection import CollectionController
from .config import WINDOW
from .data import save_session
from .display import render
from .storage import atomic_json
from .vision import FeatureExtractor, ProbabilitySmoother

LOG = logging.getLogger(__name__)


def open_capture(camera=0, video=None):
    if video is not None:
        path = Path(video)
        if not path.is_file():
            raise ValueError(f"Video does not exist: {path}")
        backends = [cv2.CAP_ANY]
        source = str(path)
    else:
        if camera < 0:
            raise ValueError("Camera index must be zero or positive.")
        source = camera
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(source, backend)
        if capture.isOpened():
            if video is None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 30)
            ok, first = capture.read()
            if ok and first is not None and first.size:
                LOG.info("Opened %s, backend %s, %sx%s", "video" if video else f"camera {camera}",
                         capture.getBackendName(), first.shape[1], first.shape[0])
                return capture, first
        capture.release()
    raise ValueError("Cannot read the camera/video. Close Teams/Zoom/Camera, enable Windows "
                     "Settings > Privacy & security > Camera > desktop app access, "
                     "and try another camera index (0, 1, or 2).")


def analyze(frame, extractor, model, smoother, timestamp, threshold=0.6):
    boxes = extractor.detect(frame)
    result = {"boxes": boxes, "label": "Looking for a face", "probabilities": None,
              "status": "Face the camera in good light.", "brightness": None}
    if len(boxes) != 1:
        smoother.clear()
        if len(boxes) > 1:
            result.update(label="Multiple faces", status="Use one face at a time for a reliable recording and prediction.")
        return result
    observation = extractor.extract(frame, boxes[0], personal=model.payload["source"] == "personal")
    result["brightness"] = observation.brightness
    if observation.issue:
        smoother.clear()
        result.update(label="Poor image quality", status=observation.issue)
        return result
    features = observation.personal if model.payload["source"] == "personal" else observation.basic
    scores = smoother.update(model.probabilities(features)[0], boxes[0], timestamp)
    best = int(np.argmax(scores))
    result["probabilities"] = scores
    result["label"] = model.classes[best] if scores[best] >= threshold else "Uncertain"
    result["status"] = ("Synthetic demo: webcam accuracy is not validated. Record your own data to train."
                        if model.payload["source"] == "simulated" else
                        "Personal model. Expression labels do not measure your internal emotional state.")
    return result


def run_webcam(model, camera=0, video=None, headless=False, max_frames=None,
               threshold=0.6, stats_path=None, output=None):
    if headless and video is None and max_frames is None:
        raise ValueError("Headless webcam runs need --max-frames so the camera is released.")
    if output and video and Path(output).resolve() == Path(video).resolve():
        raise ValueError("The output video must have a different filename from the input video.")
    if output and Path(output).suffix.lower() != ".avi":
        raise ValueError("Use an .avi filename for the annotated MJPEG output.")
    extractor = FeatureExtractor()
    smoother = ProbabilitySmoother()
    capture, frame = open_capture(camera, video)
    writer = None
    counts = {}
    timings = deque(maxlen=10_000)
    total_analysis_ms = 0.0
    analyzed_frames = 0
    frames = 0
    failures = 0
    mirror = True
    started = time.perf_counter()
    fps = 0.0
    previous = started
    try:
        if not headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1180, 680)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), 20, (1180, 680))
            if not writer.isOpened():
                raise ValueError("Cannot create the output video. Use an .avi filename and a writable folder.")
        while True:
            now = time.perf_counter()
            analysis_start = now
            result = analyze(frame, extractor, model, smoother, now, threshold)
            analysis_ms = (time.perf_counter() - analysis_start)*1000
            timings.append(analysis_ms)
            total_analysis_ms += analysis_ms
            analyzed_frames += 1
            counts[result["label"]] = counts.get(result["label"], 0) + 1
            current_fps = 1/max(1e-6, now-previous)
            fps = current_fps if frames == 1 else fps*0.9+current_fps*0.1 if frames else 0
            previous = now
            screen = render(frame, **result, fps=fps, mirror=mirror,
                            mode="SIMULATED DEMO" if model.payload["source"] == "simulated" else "PERSONAL MODEL")
            if writer:
                writer.write(screen)
            if not headless:
                cv2.imshow(WINDOW, screen)
                key = cv2.waitKey(1) & 0xff
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("m"), ord("M")):
                    mirror = not mirror
                if key in (ord("r"), ord("R")):
                    smoother.clear()
            frames += 1
            if max_frames and frames >= max_frames:
                break
            while True:
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    failures = 0
                    break
                if video is not None:
                    frame = None
                    break
                failures += 1
                if failures >= 10:
                    raise ValueError("The webcam stopped returning frames. Reconnect it and restart the app.")
                time.sleep(0.05)
            if frame is None:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    stats = {"frames": analyzed_frames, "elapsed_seconds": elapsed,
             "throughput_fps": analyzed_frames/max(elapsed, 1e-6),
             "mean_analysis_ms": total_analysis_ms/analyzed_frames if analyzed_frames else 0,
             "p95_analysis_ms": float(np.percentile(timings, 95)) if timings else 0,
             "p95_window_frames": len(timings),
             "display_counts": counts,
             "input": "video file" if video is not None else "webcam",
             "model_source": model.payload["source"],
             "note": "Processing speed is environment-specific. P95 uses at most the last 10,000 frames. "
                     "This is not an emotion accuracy test."}
    if stats_path:
        atomic_json(Path(stats_path), stats)
    return stats


def collect_session(data_path, camera=0, samples_per_class=25):
    controller = CollectionController(samples_per_class)
    extractor = FeatureExtractor()
    capture, frame = open_capture(camera)
    mirror = True
    completed = False
    failures = 0
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1180, 680)
        while not controller.complete:
            now = time.perf_counter()
            boxes = extractor.detect(frame)
            issue = None
            brightness = None
            if len(boxes) == 1:
                observation = extractor.extract(frame, boxes[0], personal=True)
                issue, brightness = observation.issue, observation.brightness
                if issue is None:
                    controller.add(observation.personal, now)
            else:
                issue = "Exactly one face must be visible."
            if controller.complete:
                completed = True
                break
            screen = render(frame, boxes, label=controller.label,
                status=issue or controller.instruction(now), mirror=mirror, mode="COLLECT YOUR DATA",
                collection=(controller.count, samples_per_class, controller.phase+1, 3), brightness=brightness)
            cv2.imshow(WINDOW, screen)
            key = cv2.waitKey(1) & 0xff
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                controller.start(time.perf_counter())
            if key in (ord("m"), ord("M")):
                mirror = not mirror
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 10:
                    raise ValueError("Camera disconnected during recording. This incomplete session was not saved.")
                time.sleep(0.05)
                while failures < 10:
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        break
                    failures += 1
                    time.sleep(0.05)
                if not ok or frame is None:
                    raise ValueError("Camera disconnected during recording. This incomplete session was not saved.")
            failures = 0
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if completed:
        save_session(Path(data_path), controller.rows)
        return len(controller.rows)
    LOG.info("Incomplete session discarded; existing data is unchanged.")
    return 0


def check_camera(camera=0):
    capture, first = open_capture(camera)
    try:
        extractor = FeatureExtractor()
        result = {"camera_index": camera, "width": first.shape[1], "height": first.shape[0],
                  "backend": capture.getBackendName(), "faces_in_first_frame": len(extractor.detect(first)),
                  "opencv": cv2.__version__, "platform": platform.system(), "success": True}
        return result
    finally:
        capture.release()
