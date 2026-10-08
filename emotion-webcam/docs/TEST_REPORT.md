# Test report — Expression Lab 1.0.1

## Executed environment

- Remote Linux, x86-64, Python 3.12.14
- OpenCV 4.12.0.88 (cv2 reports 4.12.0)
- NumPy 2.2.6, Pandas 2.2.3
- scikit-learn 1.7.2, SciPy 1.15.3, Matplotlib 3.10.8
- Exact pinned requirements; tests use Python's unittest, no extra test package

## Completed checks

| Check | Result | Evidence |
| --- | --- | --- |
| Automated unit/integration suite | 48 tests passed | reports/software-tests.txt |
| Simulated training, seed 42 | 1,920 training / 480 test samples; 87.71% accuracy | reports/demo/metrics.json |
| JSON deployment inference | Matches sklearn probabilities to numerical precision | ModelTests.test_json_inference_matches_sklearn |
| Leakage checks | Scaler fits training only; personal train/test session IDs disjoint | ModelTests and DataTests |
| Personal-mode training path | Trains, evaluates, serializes, predicts using a generated test-only 902-feature fixture | ModelTests.test_personal_training_and_holdout_model |
| Actual face-image extraction | Haar finds one face; real 902-element feature vector is deterministic and finite | VisionTests.test_real_image_face_and_hog_features |
| Camera error/cleanup paths | Failures and Q exit release resources; mocked Windows fallback succeeds | CameraTests |
| Offline decode → detect → predict → annotate → encode | Valid annotated AVI with expected frame dimensions/count | CameraTests.test_actual_video_end_to_end_and_max_frames |
| 100-frame offline video benchmark | Approximately 44 FPS; approximately 14 ms mean analysis | reports/video_integration.json |
| CLI train and doctor | Exit 0 and expected artifacts/diagnostics | ReportAndCliTests |
| Python compilation / shell syntax | Passed | compileall and bash -n |
| Pinned Windows x64 / Python 3.12 dependencies | Binary-wheel resolution succeeded for all 19 pinned packages | reports/windows_dependency_check.json (pip dry-run resolver) |
| Preview layout and confusion chart | Rendered and visually inspected | docs/camera_preview.png, docs/collection_preview.png, reports/demo/confusion_matrix.png |

The generated personal test fixture validates code paths only. It is not real
labeled human-expression training data, and its classifier score is not
reported as webcam accuracy. The NASA face fixture has no expression label.

## Windows launcher fix in 1.0.1

The Python version/bitness check previously passed inline code through Windows
PowerShell. The reported Windows run stripped its string quotes and raised
`NameError: name P is not defined`. Version 1.0.1 runs that check from
`scripts/check_runtime.py`, and uses the same file for Python discovery.
Six new regression tests cover supported versions, incorrect bitness,
nonzero failure exits, a script filename with spaces, and elimination of
inline Python code from the PowerShell launcher. These tests run here; actual
Windows execution remains a local acceptance check.

## Validation still requiring the laptop

- Actual Windows batch/PowerShell execution and Tkinter GUI interactions
- Physical webcam index, backend, permissions, camera contention, and reconnect
- Real labeled data collection and personalized expression performance
- Real-time display FPS, CPU use, and long-run behavior on the target hardware

The Windows bootstrap and fallback branches are inspected/tested where
possible; cross-platform wheel resolution is not equivalent to executing
the app on Windows. Follow LOCAL_ACCEPTANCE.md and run TEST_WINDOWS.bat before
describing the product as fully validated for that laptop.
