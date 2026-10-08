# Local Windows webcam acceptance checklist

These checks need the actual laptop. They have not been represented as completed
by tests run on the remote Linux environment.

| Check | Expected result | Local result |
| --- | --- | --- |
| Extract ZIP to a folder with spaces in its name | START_WINDOWS.bat resolves the project and builds `.venv` | Pending |
| First launch with supported Python installed | Dependencies install and launcher opens | Pending |
| Second launch without internet | Existing launcher/model operate locally | Pending |
| TEST_WINDOWS.bat | All bundled software tests pass | Pending |
| Check camera, index 0 (or correct alternate) | Diagnostic prints resolution, backend, and success | Pending |
| Assignment demo | Video, face rectangle, scores, and SYNTHETIC DEMO badge appear | Pending |
| Q, Esc, and close camera window | App returns to launcher; other apps can reopen camera | Pending |
| M | Preview mirrors; inference orientation remains consistent | Pending |
| Face leaves frame | Label returns to Looking for a face; previous scores clear | Pending |
| Two faces enter frame | Multiple faces; prediction pauses | Pending |
| Very poor light | Quality warning and suppressed scores | Pending |
| Cancel partial collection with Q | Existing data is unchanged; incomplete session is discarded | Pending |
| Complete five labeled sessions | 375 default rows, five unique sessions, all three classes in each | Pending |
| Train personal model | Report and personal.json created; train/test sessions are disjoint | Pending |
| New sixth session / independent recording | Document measured expression accuracy without assuming the synthetic result applies | Pending |
| Camera unplug/disable or already occupied | Helpful failure; app exits and releases resources | Pending |
| Typical lighting, 10-minute run | Record mean FPS, label consistency, CPU use, and any drops | Pending |

For a small live labeled evaluation: independently pose each expression 10
times, wait about one second after each pose to let smoothing settle, and
record the displayed prediction (including Uncertain). Keep the model and
threshold fixed. Report overall accuracy, rejection rate, per-class errors,
lighting, resolution, person, and number of trials. Thirty poses provide only
a small preliminary check, not broad generalization evidence.

Fill in the live evaluation section of ASSIGNMENT_REPORT.md after this check.
