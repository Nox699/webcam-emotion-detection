# Real-Time Emotion Detection Using Supervised Learning

**Course topic:** Machine Learning — Supervised Learning  
**Project:** Expression Lab, version 1.0.1  
**Target:** Windows laptop with built-in webcam  
**Student name / class / submission date:** Complete before submission

## 1. Introduction and objective

The assignment requires a supervised classifier for Happy, Sad, and Neutral,
connected to a real-time camera stream. This implementation uses Python,
OpenCV, Pandas, NumPy, scikit-learn, and Matplotlib. The Raspberry Pi camera
scenario is adapted to the requested laptop webcam through OpenCV
VideoCapture. A native desktop launcher provides access to demonstration,
collection, training, evaluation, and camera diagnostics.

The system predicts a visible facial-expression label. It cannot reliably
infer a person's internal emotional state from a face. This distinction is
especially important when interpreting synthetic-data accuracy.

## 2. Supervised-learning theory

Each labeled example consists of input features **X** and a target class **y**.
The classifier learns parameters that associate the features with the class.
Unseen test samples estimate how well that learned association generalizes
within the evaluated data distribution.

Multiclass logistic regression computes a score for each class from scaled
features, converts the scores to probabilities with softmax, and predicts
the largest score. StandardScaler subtracts the training mean and divides
by the training standard deviation. It is fit only on training data to avoid
information leakage. Regularization limits excessively large coefficients.

## 3. Assignment-task implementation

| Task | Implementation | Evidence |
| --- | --- | --- |
| 1. Import libraries | Pandas/NumPy for data; sklearn for training and metrics; Matplotlib for charts; cv2 for camera and Haar detection | Source modules and requirements.txt |
| 2. Simulated dataset | 800 labeled samples per class, 2,400 total, in a Pandas DataFrame; overlapping normalized eye/smile feature distributions | reports/demo/simulated_dataset.csv |
| 2. Training/testing split | Seeded stratified 80/20 split: 1,920 training and 480 test rows | reports/demo/metrics.json with exact indices |
| 3. Supervised model | StandardScaler + multiclass LogisticRegression | emotion_app/model.py; models/demo.json |
| 3. Evaluation | Accuracy, class precision/recall/F1, macro/weighted metrics, confusion matrix | reports/demo/evaluation.html |
| 4. Webcam capture | OpenCV VideoCapture, Windows backend fallback, 640×480 requested capture | emotion_app/camera.py |
| 4. Face and features | Frontal-face Haar cascade, face ROI, Haar eye/smile proxies; optional HOG appearance features | emotion_app/vision.py |
| 4. Prediction and display | Model inference, smoothed scores, face rectangle, label and live FPS | Camera interface and offline integration output |
| 4. Exit | Q, Esc, or camera-window close; camera and writer cleanup in finally blocks | Automated Q/exception cleanup tests |
| 5. Discussion | Measured synthetic results, measured offline processing speed, limits, local checklist, future improvements | This report; MODEL_CARD.md and TEST_REPORT.md |

## 4. Dataset preparation

The simulated dataset has two normalized features: eye-openness proxy and
smile-intensity proxy. Gaussian means are Happy=(0.72, 0.82),
Neutral=(0.62, 0.36), and Sad=(0.34, 0.15), with standard deviations
(0.14, 0.16). Values are clipped to [0, 1]. The intentional distribution
overlap makes the problem nontrivial. The seed is 42, allowing reproduction.

These values are illustrative assumptions, not measured facial landmarks.
The synthetic feature distribution is not calibrated to the Haar estimates
from a real webcam. Therefore even a high synthetic test accuracy would not
demonstrate correct real-world expression recognition.

The optional personal workflow replaces simulation with explicitly labeled
webcam features. Each session records all three posed expressions in shuffled
order, with a two-second preparation countdown and default 0.4-second sample
spacing. Twenty-five rows per class gives 75 rows per session. Collection saves
only complete sessions, and stores numerical features rather than images.

At least five sessions are required for personal training. An 80/20 session
split keeps neighboring frames together. Five equal-size default sessions
give 300 training and 75 test rows. This reduces frame-level leakage but does
not eliminate correlations or establish person-independent generalization.

## 5. Algorithm and feature extraction

Logistic regression was selected because it directly meets Option A,
is understandable for a supervised-learning course, supports multiclass
scores, and runs efficiently on a laptop. The model uses lbfgs, balanced
class weights, max_iter=2000, and C=1 for simulated data. The personal model
uses stronger regularization, C=0.2, because it has many more features.

The camera workflow detects a frontal face using an OpenCV Haar cascade,
extracts its region, and equalizes a 96×96 grayscale representation for eye
and smile detection. Detected eye rectangle proportions and smile width
produce two rough feature proxies. Eye rectangles are not true eyelid
measurements, and undetected eyes use a fixed fallback value of 0.35.

The personal model adds 900 Histogram of Oriented Gradients features from
a 48×48 CLAHE-preprocessed face. These features describe image gradients
and allow real labeled appearance data to influence the classifier. Training
and inference share exactly the same extractor version and feature order.

The deployed model uses validated numeric JSON parameters, so loading does
not execute pickle data or depend on sklearn object-serialization compatibility.
JSON probabilities are tested against sklearn.predict_proba. The exact
held-out evaluated model is deployed without refitting on the test data.

## 6. Measured model performance

**Synthetic held-out test accuracy: 87.71% (421 / 480 correct).**

| Class | Precision | Recall | F1-score | Test support |
| --- | ---: | ---: | ---: | ---: |
| Happy | 0.916 | 0.881 | 0.898 | 160 |
| Neutral | 0.810 | 0.825 | 0.817 | 160 |
| Sad | 0.908 | 0.925 | 0.916 | 160 |
| Macro average | 0.878 | 0.877 | 0.877 | 480 |
| Weighted average | 0.878 | 0.877 | 0.877 | 480 |

Confusion matrix (rows are true labels; columns are predictions):

| True / Predicted | Happy | Neutral | Sad |
| --- | ---: | ---: | ---: |
| Happy | 141 | 19 | 0 |
| Neutral | 13 | 132 | 15 |
| Sad | 0 | 12 | 148 |

Neutral has the weakest performance because its assumed distributions overlap
with both other classes. Precision measures the proportion of predictions of a
class that are correct. Recall measures the proportion of actual examples of
a class that are recovered. F1 balances precision and recall. Because the
test set is class-balanced, macro and weighted averages coincide here.

Inference on a fixed input is deterministic. Temporal smoothing reduces
changes between neighboring webcam frames, but consistency and correctness
on the requested physical webcam have not yet been measured. The synthetic
metric must not be substituted for a live result.

## 7. Real-time workflow and measured software performance

The app reads a frame, detects faces, and classifies only when exactly one
face is visible with acceptable image quality. A smoothed maximum model
score below 60% displays Uncertain. Low light, excessive brightness, low
contrast, or blur pauses prediction. Scores reset on face loss or substantial
position changes. The preview can be mirrored without changing inference
orientation. Q releases the camera and closes the preview.

An offline 100-frame MJPEG integration video, containing 10 blank frames and
90 repeated public-domain face-image frames, ran at approximately **44 FPS**
in the remote Linux environment. Average face-analysis time was approximately
**14 ms** and the 95th percentile approximately **18 ms**. This timing includes
a specific input and environment; it does not establish a Windows webcam
frame rate. The image has no ground-truth expression label, so the model's
displayed output is not an emotion-accuracy test. Exact figures are in
reports/video_integration.json.

The live quality/smoothing pipeline is not part of the raw classifier test
metrics. To assess it, separately measure response delay, rejection rate,
stable-pose consistency, and correctness on independent live labeled trials.

## 8. Testing and challenges

Automated checks cover generated-data reproducibility, class balance,
train/test separation, training-only scaling, personal session holdout,
JSON inference parity, invalid files, real image feature extraction,
quality gates, temporal resets, complete-session writes, interrupted sessions,
camera-open failure, Windows backend fallback, resource cleanup, Q exit,
offline video decoding/annotation, and CLI training/evaluation.

Important challenges are weak Haar feature proxies, lighting sensitivity,
pose/face detection errors, glasses and occlusion, correlated frames, and
limited real labeled data. HOG improves access to image appearance but can
also learn irrelevant lighting or identity cues. Sad and Neutral are often
hard to distinguish visually. Temporal smoothing improves stability at the
cost of some delay.

Windows-compatible package wheels were resolved for the pinned dependency
set. Native Windows launcher, GUI, camera-permission, and physical-webcam
acceptance remain local checks; their status is documented rather than
presented as completed.

## 9. Future improvements

1. Collect a larger consented real dataset spanning people, cameras, lighting,
   glasses, and pose; use person-disjoint final evaluation.
2. Replace detector rectangle proxies with facial landmarks and better
   alignment; use measured eyelid/mouth geometry.
3. Compare logistic regression, SVM, and random forest on identical held-out
   sets. Tune settings with training-only cross-validation.
4. Evaluate a CNN or a real dataset such as FER2013, with explicit license,
   preprocessing, class mapping, and person/data-leakage checks.
5. Calibrate classifier scores and select rejection thresholds using separate
   validation data. Evaluate rejection coverage and error jointly.
6. Add Angry, Fearful, Surprised, or Disgusted only after collecting suitable
   labels and validating those classes.

## 10. Live laptop evaluation — complete after local testing

| Item | Result to enter |
| --- | --- |
| Windows version / laptop / webcam | Not yet measured |
| Webcam backend / resolution | Run doctor --camera 0 |
| Real training sessions / people | Not yet collected |
| Personal held-out accuracy and F1 | Copy from reports/personal/evaluation.html after training |
| Independent live trial count / accuracy | Not yet measured |
| Uncertain / quality rejection rate | Not yet measured |
| Stable-pose consistency / transition delay | Not yet measured |
| Typical FPS and poor-light behavior | Not yet measured |

## 11. References

- OpenCV cascade-classifier tutorial: https://docs.opencv.org/4.12.0/db/d28/tutorial_cascade_classifier.html
- OpenCV HOGDescriptor: https://docs.opencv.org/4.0.0/d5/d33/structcv_1_1HOGDescriptor.html
- scikit-learn LogisticRegression: https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
- scikit-learn classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- Python Windows setup: https://docs.python.org/3/using/windows.html
- Public-domain image fixture: https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut
