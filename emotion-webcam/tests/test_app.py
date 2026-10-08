import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

from emotion_app.camera import analyze, check_camera, open_capture, run_webcam
from emotion_app.collection import CollectionController
from emotion_app.config import BASIC_FEATURES, DEMO_MODEL, LABELS, PERSONAL_FEATURES, ROOT
from emotion_app.data import read_personal, save_session, simulated_dataset, split_dataset, validate_dataset
from emotion_app.display import render
from emotion_app.model import EmotionModel, save_model, train
from emotion_app.reporting import write_report
from emotion_app.storage import atomic_json, atomic_text
from emotion_app.vision import FaceObservation, FeatureExtractor, ProbabilitySmoother, overlap

FIXTURE = Path(__file__).parent / "fixtures" / "face.png"


def fake_personal(sessions=5):
    rng = np.random.default_rng(901)
    rows = []
    for session in range(sessions):
        for index, label in enumerate(LABELS):
            for sample in range(10):
                values = rng.uniform(0, 0.12, len(PERSONAL_FEATURES))
                values[index+2] += 0.7
                row = dict(zip(PERSONAL_FEATURES, values))
                row.update(label=label, session=f"test-session-{session}", sample=sample, captured_at="test")
                rows.append(row)
    return pd.DataFrame(rows)


def completed_controller(samples=10):
    controller = CollectionController(samples, interval=0.4)
    now = 0.0
    features = np.full(len(PERSONAL_FEATURES), 0.2)
    while not controller.complete:
        controller.start(now)
        now += 2.01
        for _ in range(samples):
            controller.add(features, now)
            now += 0.41
    return controller


class DataTests(unittest.TestCase):
    def test_simulation_is_balanced_bounded_and_reproducible(self):
        a, b = simulated_dataset(), simulated_dataset()
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 2400)
        self.assertEqual(a.groupby("label").size().tolist(), [800, 800, 800])
        self.assertTrue(a[list(BASIC_FEATURES)].ge(0).all().all())
        self.assertTrue(a[list(BASIC_FEATURES)].le(1).all().all())

    def test_synthetic_split_is_stratified_80_20(self):
        frame = simulated_dataset()
        train_i, test_i = split_dataset(frame)
        self.assertEqual((len(train_i), len(test_i)), (1920, 480))
        self.assertFalse(set(train_i) & set(test_i))
        self.assertEqual(frame.iloc[test_i].groupby("label").size().tolist(), [160]*3)

    def test_rejects_nan_and_invalid_labels_and_bounds(self):
        for field, value in [(BASIC_FEATURES[0], np.nan), (BASIC_FEATURES[0], 1.1),
                             ("label", "Angry")]:
            with self.subTest(field=field, value=value):
                frame = simulated_dataset(30)
                frame.loc[0, field] = value
                with self.assertRaises(ValueError):
                    validate_dataset(frame, BASIC_FEATURES)

    def test_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            validate_dataset(simulated_dataset(30).drop(columns=BASIC_FEATURES[0]), BASIC_FEATURES)
        with self.assertRaises(ValueError):
            simulated_dataset(1)

    def test_requires_five_complete_sessions(self):
        with self.assertRaisesRegex(ValueError, "5 separate"):
            split_dataset(fake_personal(4), True)
        incomplete = fake_personal()
        incomplete = incomplete.loc[~((incomplete.session == "test-session-0") & (incomplete.label == "Sad"))]
        with self.assertRaisesRegex(ValueError, "Each session"):
            split_dataset(incomplete, True)

    def test_personal_holdout_has_no_session_leakage(self):
        frame = fake_personal()
        train_i, test_i = split_dataset(frame, True)
        self.assertEqual((len(train_i), len(test_i)), (120, 30))
        self.assertFalse(set(frame.iloc[train_i].session) & set(frame.iloc[test_i].session))
        self.assertEqual(set(frame.iloc[test_i].label), set(LABELS))

    def test_missing_personal_file_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Record"):
                read_personal(Path(directory) / "missing.csv")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = simulated_dataset()
        cls.model, cls.metrics, cls.pipeline = train(cls.frame)

    def test_scaler_uses_only_training_subset(self):
        values = self.frame[list(BASIC_FEATURES)].to_numpy()
        expected = values[self.metrics["train_indices"]].mean(axis=0)
        np.testing.assert_allclose(self.model.mean, expected, atol=1e-12)

    def test_json_inference_matches_sklearn(self):
        values = self.frame[list(BASIC_FEATURES)].to_numpy()
        np.testing.assert_allclose(self.model.probabilities(values), self.pipeline.predict_proba(values),
                                   atol=1e-12, rtol=1e-12)

    def test_roundtrip_json_is_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(self.model, path)
            loaded = EmotionModel.load(path)
            np.testing.assert_allclose(loaded.probabilities([0.7, 0.8]), self.model.probabilities([0.7, 0.8]))

    def test_metrics_match_predictions(self):
        test_i = self.metrics["test_indices"]
        scores = self.model.probabilities(self.frame.iloc[test_i][list(BASIC_FEATURES)].to_numpy())
        predicted = np.array(LABELS)[scores.argmax(axis=1)]
        actual = self.frame.iloc[test_i].label.to_numpy()
        self.assertAlmostEqual(self.metrics["accuracy"], np.mean(predicted == actual))
        self.assertEqual(sum(map(sum, self.metrics["confusion_matrix"])), 480)
        self.assertGreater(self.metrics["accuracy"], 0.80)
        self.assertLess(self.metrics["accuracy"], 1)

    def test_rejects_corrupt_dimensions_scale_and_schema(self):
        for field, value in [("scale", [0, 1]), ("coef", [[1, 2]]), ("mean", [float("nan"), 1]),
                             ("feature_names", ["smile", "eye"]), ("extractor_version", "bad"),
                             ("classes", ["Happy", "Sad", "Neutral"])]:
            with self.subTest(field=field):
                payload = copy.deepcopy(self.model.payload)
                payload[field] = value
                with self.assertRaises(ValueError):
                    EmotionModel(payload)

    def test_rejects_invalid_json_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaises(ValueError):
                EmotionModel.load(path)
            for content in ["not json", "[]", "{}"]:
                atomic_text(path, content)
                with self.assertRaises(ValueError):
                    EmotionModel.load(path)

    def test_rejects_bad_prediction_inputs(self):
        for values in [[1], [np.nan, 0.2], [[[0.3, 0.4]]]]:
            with self.assertRaises(ValueError):
                self.model.probabilities(values)

    def test_probabilities_are_normalized(self):
        scores = self.model.probabilities([[0, 0], [1, 1], [0.5, 0.5]])
        np.testing.assert_allclose(scores.sum(axis=1), 1)
        self.assertTrue((scores >= 0).all())

    def test_personal_training_and_holdout_model(self):
        frame = fake_personal()
        model, metrics, pipeline = train(frame, True)
        self.assertEqual(len(model.names), 902)
        self.assertFalse(set(metrics["train_sessions"]) & set(metrics["test_sessions"]))
        values = frame[list(PERSONAL_FEATURES)].to_numpy()
        np.testing.assert_allclose(model.probabilities(values), pipeline.predict_proba(values), atol=1e-12)


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = FeatureExtractor()
        cls.frame = cv2.imread(str(FIXTURE))
        if cls.frame is None:
            raise RuntimeError("Missing bundled public-domain face test fixture.")

    def test_real_image_face_and_hog_features(self):
        boxes = self.extractor.detect(self.frame)
        self.assertEqual(len(boxes), 1)
        a = self.extractor.extract(self.frame, boxes[0], True)
        b = self.extractor.extract(self.frame, boxes[0], True)
        self.assertIsNone(a.issue)
        self.assertEqual(a.personal.shape, (902,))
        np.testing.assert_array_equal(a.personal, b.personal)
        self.assertTrue(np.isfinite(a.personal).all())
        self.assertTrue(((a.personal >= 0) & (a.personal <= 1)).all())

    def test_blank_image_has_no_face(self):
        self.assertEqual(self.extractor.detect(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_rejects_invalid_image_and_box(self):
        for image in [np.zeros((0, 0), dtype=np.uint8), np.zeros((20, 20)),
                      np.zeros((20, 20, 4), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                self.extractor.detect(image)
        for box in [(-1, 0, 100, 100), (0, 0, 10, 10), (500, 500, 100, 100)]:
            with self.assertRaises(ValueError):
                self.extractor.extract(self.frame, box)

    def test_dark_and_blurry_quality_gates(self):
        dark = self.extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 100, 100))
        self.assertIn("dark", dark.issue)
        flat = self.extractor.extract(np.full((100, 100, 3), 100, dtype=np.uint8), (0, 0, 100, 100))
        self.assertIsNotNone(flat.issue)

    def test_smoother_resets_on_face_change_and_loss(self):
        smoother = ProbabilitySmoother()
        a = (0, 0, 100, 100)
        b = (200, 0, 100, 100)
        smoother.update([0.8, 0.1, 0.1], a, 0)
        mixed = smoother.update([0.1, 0.8, 0.1], a, 0.1)
        self.assertGreater(mixed[0], 0.1)
        np.testing.assert_array_equal(smoother.update([0.1, 0.8, 0.1], b, 0.2), [0.1, 0.8, 0.1])
        smoother.clear()
        self.assertIsNone(smoother.values)

    def test_overlap(self):
        self.assertEqual(overlap((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(overlap((0, 0, 10, 10), (20, 20, 10, 10)), 0)

    def test_analysis_suppresses_no_multiple_and_poor_faces(self):
        model = EmotionModel.load(DEMO_MODEL)
        smoother = ProbabilitySmoother()
        fake = MagicMock()
        for boxes, expected in [([], "Looking for a face"), ([(0, 0, 100, 100)]*2, "Multiple faces")]:
            fake.detect.return_value = boxes
            smoother.values = np.array([0.9, 0.05, 0.05])
            result = analyze(self.frame, fake, model, smoother, 1)
            self.assertEqual(result["label"], expected)
            self.assertIsNone(result["probabilities"])
            self.assertIsNone(smoother.values)
        fake.detect.return_value = [(0, 0, 100, 100)]
        fake.extract.return_value = FaceObservation((0, 0, 100, 100), np.array([0.7, 0.8]), None, 0, 0, "Too dark")
        self.assertEqual(analyze(self.frame, fake, model, smoother, 2)["label"], "Poor image quality")

    def test_threshold_yields_uncertain(self):
        fake = MagicMock()
        fake.detect.return_value = [(0, 0, 100, 100)]
        fake.extract.return_value = FaceObservation((0, 0, 100, 100), np.array([0.5, 0.5]), None, 100, 100, None)
        result = analyze(self.frame, fake, EmotionModel.load(DEMO_MODEL), ProbabilitySmoother(), 0, threshold=1)
        self.assertEqual(result["label"], "Uncertain")

    def test_renderer_layout_and_mirror(self):
        a = render(self.frame, [(170, 60, 110, 110)], label="Happy", probabilities=[0.8, 0.1, 0.1])
        b = render(self.frame, [(170, 60, 110, 110)], mirror=False)
        self.assertEqual(a.shape, (680, 1180, 3))
        self.assertFalse(np.array_equal(a, b))


class CollectionTests(unittest.TestCase):
    def test_countdown_spacing_and_completion(self):
        controller = CollectionController(10)
        features = np.full(902, 0.1)
        self.assertFalse(controller.add(features, 0))
        controller.start(0)
        self.assertFalse(controller.add(features, 1))
        self.assertTrue(controller.add(features, 2))
        self.assertFalse(controller.add(features, 2.1))
        complete = completed_controller()
        self.assertTrue(complete.complete)
        self.assertEqual(len(complete.rows), 30)
        self.assertEqual(pd.DataFrame(complete.rows).groupby("label").size().tolist(), [10]*3)

    def test_atomic_append_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "personal.csv"
            a, b = completed_controller(), completed_controller()
            save_session(path, a.rows)
            save_session(path, b.rows)
            frame = read_personal(path)
            self.assertEqual((len(frame), frame.session.nunique()), (60, 2))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                save_session(path, a.rows)
            self.assertEqual(path.read_bytes(), before)

    def test_incomplete_session_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            rows = completed_controller().rows[:10]
            with self.assertRaises(ValueError):
                save_session(path, rows)
            self.assertFalse(path.exists())

    def test_invalid_collection_size(self):
        for size in [1, 201]:
            with self.assertRaises(ValueError):
                CollectionController(size)


class CameraTests(unittest.TestCase):
    def test_failed_open_releases_camera(self):
        fake = MagicMock()
        fake.isOpened.return_value = False
        with patch("emotion_app.camera.cv2.VideoCapture", return_value=fake), patch("emotion_app.camera.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                open_capture()
        fake.release.assert_called_once()

    def test_windows_backend_fallback_releases_failed_device(self):
        bad, good = MagicMock(), MagicMock()
        bad.isOpened.return_value = True
        bad.read.return_value = (False, None)
        good.isOpened.return_value = True
        good.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        with patch("emotion_app.camera.cv2.VideoCapture", side_effect=[bad, good]), patch("emotion_app.camera.platform.system", return_value="Windows"):
            result, frame = open_capture()
        bad.release.assert_called_once()
        self.assertIs(result, good)
        self.assertEqual(frame.shape, (100, 100, 3))

    def test_bad_camera_index_and_missing_video(self):
        with self.assertRaises(ValueError):
            open_capture(-1)
        with self.assertRaises(ValueError):
            open_capture(video=ROOT / "missing-video.avi")

    def test_exception_during_processing_releases_camera(self):
        capture = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("emotion_app.camera.open_capture", return_value=(capture, frame)), patch("emotion_app.camera.analyze", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                run_webcam(EmotionModel.load(DEMO_MODEL), headless=True, max_frames=1)
        capture.release.assert_called_once()

    def test_q_key_releases_camera_and_window(self):
        capture = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("emotion_app.camera.open_capture", return_value=(capture, frame)), \
             patch("emotion_app.camera.cv2.namedWindow"), patch("emotion_app.camera.cv2.resizeWindow"), \
             patch("emotion_app.camera.cv2.imshow"), patch("emotion_app.camera.cv2.waitKey", return_value=ord("q")), \
             patch("emotion_app.camera.cv2.destroyAllWindows") as destroy:
            stats = run_webcam(EmotionModel.load(DEMO_MODEL))
        capture.release.assert_called_once()
        destroy.assert_called_once()
        self.assertEqual(stats["frames"], 1)

    def test_headless_webcam_requires_frame_limit(self):
        with self.assertRaisesRegex(ValueError, "max-frames"):
            run_webcam(EmotionModel.load(DEMO_MODEL), headless=True)

    def test_output_cannot_overwrite_input_and_requires_avi(self):
        with self.assertRaisesRegex(ValueError, "different filename"):
            run_webcam(EmotionModel.load(DEMO_MODEL), video=FIXTURE, output=FIXTURE, headless=True)
        with self.assertRaisesRegex(ValueError, ".avi"):
            run_webcam(EmotionModel.load(DEMO_MODEL), output="output.mp4", headless=True, max_frames=1)

    def test_doctor_releases_camera(self):
        capture = MagicMock()
        frame = cv2.imread(str(FIXTURE))
        with patch("emotion_app.camera.open_capture", return_value=(capture, frame)):
            result = check_camera(0)
        self.assertTrue(result["success"])
        capture.release.assert_called_once()

    def test_actual_video_end_to_end_and_max_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "input.avi", root / "output.avi"
            photo = cv2.imread(str(FIXTURE))
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 15, (512, 512))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros_like(photo))
            for _ in range(7):
                writer.write(photo)
            writer.release()
            stats = run_webcam(EmotionModel.load(DEMO_MODEL), video=source, headless=True,
                               stats_path=root / "stats.json", output=destination)
            self.assertEqual(stats["frames"], 10)
            self.assertEqual(stats["display_counts"].get("Looking for a face"), 3)
            self.assertEqual(sum(stats["display_counts"].values()), 10)
            saved = json.loads((root / "stats.json").read_text())
            self.assertEqual(saved["frames"], 10)
            recorded = cv2.VideoCapture(str(destination))
            self.assertTrue(recorded.isOpened())
            self.assertEqual(int(recorded.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            ok, frame = recorded.read()
            recorded.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (680, 1180, 3))
            limited = run_webcam(EmotionModel.load(DEMO_MODEL), video=source, headless=True, max_frames=4)
            self.assertEqual(limited["frames"], 4)


class ReportAndCliTests(unittest.TestCase):
    def test_report_is_self_contained_and_matches_metrics(self):
        _, metrics, _ = train(simulated_dataset(30))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = write_report(metrics, root)
            content = report.read_text()
            self.assertIn("data:image/png;base64,", content)
            self.assertIn("do not measure webcam accuracy", content)
            self.assertIn(f"{metrics['accuracy']:.1%}", content)
            saved = json.loads((root / "metrics.json").read_text())
            self.assertEqual(saved, metrics)
            self.assertIsNotNone(cv2.imread(str(root / "confusion_matrix.png")))

    def test_atomic_writes_do_not_leave_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "sample.json", {"a": 1})
            atomic_text(root / "sample.json", "replacement")
            self.assertEqual((root / "sample.json").read_text(), "replacement")
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_cli_training_and_doctor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-m", "emotion_app", "train", "--samples-per-class", "30",
                       "--model", str(root / "test.json"), "--report-dir", str(root / "report")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "test.json").exists())
            self.assertIn("simulated-data accuracy", result.stdout)
        result = subprocess.run([sys.executable, "-m", "emotion_app", "doctor"], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["detectors"], "OK")

    def test_cli_rejects_invalid_confidence(self):
        result = subprocess.run([sys.executable, "-m", "emotion_app", "run", "--confidence", "2"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("threshold", result.stderr)


if __name__ == "__main__":
    unittest.main()
