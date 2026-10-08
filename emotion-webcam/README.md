# Expression Lab — Real-Time Emotion Detection

A complete local Python implementation of the supervised-learning assignment,
prioritized for a Windows laptop and its built-in webcam. Classifies **posed
facial expressions** as Happy, Sad, or Neutral. It does not measure a person's
internal emotional state.

## Start on Windows

1. Install **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/windows/).
   Python 3.11 and 3.13 are also supported targets. Include the launcher, pip,
   and Tcl/Tk. These pinned dependencies target Python 3.11–3.13, not 3.14.
2. Extract the entire ZIP to a writable folder, for example `Documents\emotion-webcam`.
   Open the extracted folder; do not run the launcher inside the ZIP.
3. Double-click **START_WINDOWS.bat**. The first run installs dependencies into
   `.venv` and needs internet access. Later operation runs locally and offline.
4. Click **Open assignment demo**. Allow desktop camera access in Windows
   Settings if necessary. Click the camera window, then press **Q** to quit.

Open **START_HERE.html** for the illustrated quick-start page.
Run **TEST_WINDOWS.bat** to run the software tests on your laptop.

## Two model modes

| Mode | Training data | Features | How to interpret it |
| --- | --- | --- | --- |
| Assignment demo | 2,400 simulated labeled samples | Eye-openness and smile-intensity proxies | Demonstrates the assignment; simulated accuracy does not establish webcam accuracy. |
| Personal webcam | Your explicitly labeled recording sessions | Same two proxies + 900 image-derived HOG features | Evaluated on held-out sessions; performance is specific to the person, camera, expressions, and conditions recorded. |

The bundled assignment model achieved **87.71%** accuracy on 480 synthetic test
samples (1,920 training samples; seed 42). The report contains per-class
precision, recall, F1, and a confusion matrix. Real webcam accuracy has not
been measured in this environment.

## Train a model for your own webcam

1. Click **Record labeled session**. One session includes all three expressions
   in a shuffled order. Pose the displayed expression, press **SPACE**, hold it
   through the two-second countdown, and vary your head position slightly
   while keeping your face frontal. Default: 25 samples per expression.
2. Follow the prompts for all three labels. A complete session saves 75 rows.
   Pressing Q early discards that incomplete session and preserves earlier data.
3. Repeat recording **at least five times**. Between sessions, take a break and
   vary position or lighting a little. Ideally record sessions at different times.
   Each session must contain all three labels. Keep one person in frame.
4. Click **Train my webcam model**, then **Open evaluation report**. The split
   holds out 20% of complete sessions, preventing adjacent frames from entering
   both the training and test sets. Five equal-size sessions give 300 training
   and 75 test rows. The model deployed is exactly the model evaluated.
5. Click **Run my webcam model**. Collect additional fresh sessions if predictions
   are unreliable. Avoid repeatedly changing settings to optimize the same test
   session; reserve new sessions for a final independent check.

Only numerical features, labels, and session metadata are saved during normal
collection. No photos, video, audio, or cloud transmission are used. HOG
features are still face-derived data: protect or delete them when no longer needed.

## Controls and output

| Control or label | Meaning |
| --- | --- |
| Q / Esc / camera-window close | Quit and release the camera |
| M | Toggle mirrored display; inference always uses the original image orientation |
| R | Reset temporal probability smoothing |
| SPACE, during collection | Start the displayed labeled expression after a two-second countdown |
| Uncertain | Highest smoothed model score is below the threshold (default 60%) |
| Poor image quality | Low light, excessive brightness, blur, or low contrast suppresses prediction |
| Multiple faces | Predictions pause until exactly one face is visible |

The live bars show classifier scores, not calibrated probabilities of emotions.
Smoothing resets when a face disappears, changes position substantially, or
image quality is insufficient. It reduces flicker but adds a short response delay.

## Command-line use

From the project folder after Windows setup:

```powershell
.venv\Scripts\python -m emotion_app doctor --camera 0
.venv\Scripts\python -m emotion_app train
.venv\Scripts\python -m emotion_app run --demo --camera 0
.venv\Scripts\python -m emotion_app collect --camera 0
.venv\Scripts\python -m emotion_app train --personal
.venv\Scripts\python -m emotion_app run --camera 0 --confidence 0.65
.venv\Scripts\python -m unittest discover -s tests -v
```

Without `--demo` or an explicit `--model`, `run` uses the personal model when one
exists and otherwise the simulated demo. The launcher's personal-model button
requires a trained personal model and never silently falls back to the demo.

Explicit offline-video testing and optional annotated recording:

```powershell
.venv\Scripts\python -m emotion_app run --demo --video sample.avi --headless --stats reports\video.json
.venv\Scripts\python -m emotion_app run --model models\personal.json --output reports\recording.avi
```

`--output` explicitly saves annotated video; it is never enabled by the launcher.
Recorded output uses MJPEG AVI at 20 FPS. It is a diagnostic export, not an audio
recording or a synchronized recreation of capture timing.

For macOS/Linux: install Python 3.11–3.13 with Tk, then run `bash start.sh`.
On Linux, distro packages such as `python3-tk`, `python3-venv`, `libgl1`, and
`libglib2.0-0` may be necessary. Cross-platform hardware validation remains local.

## Project map

| Location | Purpose |
| --- | --- |
| `emotion_app/data.py` | Pandas simulation, validation, stratified and session-based splits |
| `emotion_app/model.py` | Scaler + Logistic Regression training, metrics, safe JSON inference |
| `emotion_app/vision.py` | Haar detection, quality checks, image features, probability smoothing |
| `emotion_app/camera.py` | Webcam/video capture, inference, recording, cleanup, diagnostics |
| `emotion_app/collection.py` | Labeled-session recording controller |
| `emotion_app/display.py` | Live camera interface |
| `emotion_app/launcher.py` | Native desktop launcher |
| `emotion_app/reporting.py` | Offline HTML evaluation, JSON metrics, PNG confusion chart |
| `tests/` | Automated tests and a public-domain image fixture |
| `models/demo.json` | Pretrained simulated assignment model |
| `reports/demo/` | Reproducible synthetic dataset and measured evaluation |
| `docs/ASSIGNMENT_REPORT.md` | Assignment explanation, results, limitations, discussion |
| `docs/TEST_REPORT.md` | Executed checks and explicit validation gaps |
| `docs/LOCAL_ACCEPTANCE.md` | Physical webcam and Windows checks to perform locally |

## Engineering and limitations

Pinned dependencies, an isolated environment, reproducible seeded training,
validated JSON models, atomic dataset/model writes, bounded rotating logs,
camera backend fallback, quality/uncertainty gates, and automated video
integration tests are included. Models contain numeric parameters; loading
does not execute Python object serialization. No extra landmark-model download
is required; Haar detectors come with the OpenCV wheel.

This is a finished course application with production-minded engineering, not
a scientifically validated general-purpose emotion product. Sad/Neutral can
be visually ambiguous. Haar eye rectangles do **not** measure eyelid openness.
The default simulation is not calibrated to actual webcam feature distributions.
Personal HOG models can learn identity, lighting, or pose; independent people
and larger real datasets would be needed for general deployment. Windows GUI,
permissions, and your physical camera must pass the supplied local checklist
before claiming readiness for that laptop.

See `docs/TROUBLESHOOTING.md`, `docs/MODEL_CARD.md`, and `THIRD_PARTY_NOTICES.md`.
