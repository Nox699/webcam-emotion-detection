# Model card

**Intended use:** course demonstration of supervised learning and local
classification of consented, posed facial expressions for one laptop user.

**Classes:** Happy, Neutral, Sad. Alphabetical order is used consistently in
training, JSON inference, bars, reports, and confusion matrices.

**Architecture:** StandardScaler followed by multiclass logistic regression
(lbfgs, max_iter=2000, balanced class weights; C=1 for simulation and C=0.2
for personal HOG models). The scaler is fit on training samples only. The
held-out evaluation model is deployed without a full-dataset refit.

**Demo data:** 800 samples per class, generated from overlapping Gaussian
distributions for two normalized feature proxies, clipped to [0, 1]. Seed 42.
80/20 stratified split. Demo distributions are an illustrative assumption,
not empirically estimated facial-expression measurements.

**Personal data:** explicitly labeled features from complete recording
sessions. All three classes must have at least 10 samples in each session.
At least five sessions are required. GroupShuffleSplit holds out 20% of
session IDs (rounded upward); this is 20% of sessions, not necessarily 20%
of rows when session sizes differ. No face images are stored by collection.

**Feature extraction:** OpenCV frontal-face Haar cascade; eye and smile Haar
cascades on a 96×96 equalized face; rough eye rectangle ratio and smile width
proxies. Personal model adds HOG on a 48×48 CLAHE-preprocessed face (16×16
blocks, 8×8 stride/cells, nine bins, 900 coefficients). HOG captures image
gradients; it is not a facial landmark estimator. Normal operation uses one
frontal face at a time.

**Live postprocessing:** elapsed-time exponential smoothing (0.35-second
time constant), reset on face loss, poor quality, IoU below 0.3, or a gap
above one second. Highest score below 0.6 displays Uncertain by default.
Brightness <35 or >225, contrast <12, or normalized Laplacian variance <25
suppresses prediction. Thresholds are practical defaults, not calibrated
clinical criteria. IoU continuity does not verify person identity: switching
people at the same location can retain smoothing briefly. Press R to clear it.

**Evaluation:** accuracy, per-class precision/recall/F1, macro/weighted
averages, confusion matrix, exact split indices and session IDs. Metrics
evaluate raw held-out classifier predictions, without live smoothing or
quality rejection. They do not measure temporal response or rejection-adjusted
live accuracy. The report states its data source explicitly.

**Known risks to validity:** recording frames are correlated; session
holdout reduces leakage but does not solve small-sample uncertainty.
Training on one person is not evidence for performance on other people.
Posed expressions are not spontaneous emotion. Sad/Neutral ambiguity, face
alignment, pose, glasses, blur, lighting, and detector errors can dominate.
The model can learn nuisance cues rather than expression. High scores can
be wrong and are not calibrated emotion probabilities. No demographic,
clinical, hiring, surveillance, or safety-critical validation was performed.

**Privacy:** numerical HOG features remain face-derived information.
Keep the project local and protect personal.csv. Collection saves no raw
images. Optional CLI `--output` is an explicit annotated-video export.

**Release gate:** complete LOCAL_ACCEPTANCE.md on the intended Windows
laptop. For deployment beyond this assignment, obtain a larger consented
real dataset, person-disjoint evaluations, appropriate intended-use review,
and independent real-world validation.
