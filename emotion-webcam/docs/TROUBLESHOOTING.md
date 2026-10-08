# Troubleshooting

| Symptom | Action |
| --- | --- |
| Python not found or wrong version | Install 64-bit Python 3.12 with pip, launcher, and Tcl/Tk. Close and reopen the folder, then run START_WINDOWS.bat. Python 3.14 is outside this dependency set. |
| Camera cannot open | Close Camera, Teams, Zoom, and other camera apps. Enable Settings → Privacy & security → Camera → Camera access and Let desktop apps access your camera. Try camera 1 or 2 in the launcher. |
| First run cannot download packages | Check internet access. Retry START_WINDOWS.bat. On a restricted school laptop, ask the administrator for ordinary package-install or camera access. |
| School policy blocks the PowerShell launcher | Use a terminal with the manual setup below; follow your school's execution policy. |
| Tkinter is missing | Modify the Python installation to include Tcl/Tk. The direct CLI `python -m emotion_app run --demo` can still run without Tk if OpenCV GUI support works. |
| OpenCV GUI error | Use opencv-python, not opencv-python-headless. Remove conflicting OpenCV variants from the environment, or recreate `.venv`. |
| Too dark / too bright / blurred | Put light in front of your face, reduce backlighting/glare, clean the camera, sit closer, and keep your face frontal. |
| No face detected | Face the camera, avoid extreme angles or obstruction, and ensure the face occupies at least roughly 80 pixels across. Haar detection can miss faces. |
| Multiple faces | Keep only the person being classified in frame. This app intentionally pauses classification with multiple people. |
| Prediction flickers or gets Sad/Neutral wrong | Train a personal model from at least five varied sessions. Add fresh labeled sessions. A synthetic classifier cannot establish real webcam accuracy. |
| Training says five sessions are required | Finish all three labels before exiting collection. Repeat the recording button five separate times; 375 rows is the default total. |
| Personal model or CSV is corrupt | Back up the file. Rename `data/personal.csv` to restart data collection, or retrain `models/personal.json`. The app never silently overwrites an incompatible dataset. |
| Launcher seems busy | Finish the current camera task with Q. Only one task runs at a time. If you close the launcher during a task, it waits for that task to finish so writes and camera cleanup complete normally. |
| Dependencies become broken | Delete only the project's `.venv` folder, then rerun START_WINDOWS.bat. Keep `data`, `models`, and `reports` to preserve your work. |

Manual Windows setup, from the extracted project folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --only-binary=:all: -r requirements.txt
.venv\Scripts\python -m emotion_app doctor
.venv\Scripts\python -m emotion_app gui
```

Logs are in `logs/app.log` with two rotating backups. They record operational
errors and timings, not camera frames. When requesting help, provide the error
and Python/OpenCV versions, not your personal dataset or recorded video.

To delete personal training information, exit all tasks and delete
`data/personal.csv`, `models/personal.json`, and `reports/personal/`.
Delete any explicit video exports you created separately.
