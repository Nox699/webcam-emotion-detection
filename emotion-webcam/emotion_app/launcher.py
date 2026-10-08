"""Small Windows-friendly Tk desktop launcher with asynchronous child tasks."""
import queue
import subprocess
import sys
import threading
import webbrowser

from .config import PERSONAL_DATA, PERSONAL_MODEL, ROOT


def launch():
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as error:
        raise ValueError("Tkinter is missing. On Windows reinstall Python with Tcl/Tk enabled. "
                         "Alternatively run: python -m emotion_app run --demo") from error
    root = tk.Tk()
    root.title("Expression Lab")
    root.geometry("800x730")
    root.minsize(730, 660)
    root.configure(bg="#101820")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Lab.TButton", font=("Segoe UI", 11), padding=(14, 10))
    camera = tk.IntVar(value=0)
    events = queue.Queue()
    child = None
    buttons = []
    closing = False

    def label(value, size=11, color="#aeb9c5", parent=root):
        item = tk.Label(parent, text=value, bg="#101820", fg=color,
                        font=("Segoe UI", size), anchor="w", justify="left")
        return item

    top = tk.Frame(root, bg="#101820")
    top.pack(fill="x", padx=28, pady=(24, 10))
    label("EXPRESSION LAB", 25, "#ecf5fa", top).pack(anchor="w")
    label("Happy  /  Neutral  /  Sad    |    Local supervised learning", 11, parent=top).pack(anchor="w", pady=8)
    label("The demo uses simulated data. Train your own model for your webcam.", 11, "#f2c97c", top).pack(anchor="w")
    settings = tk.Frame(root, bg="#101820")
    settings.pack(fill="x", padx=28, pady=8)
    label("Camera index", parent=settings).pack(side="left")
    spin = ttk.Spinbox(settings, from_=0, to=9, textvariable=camera, width=4)
    spin.pack(side="left", padx=12)
    label("Try 1 or 2 if your built-in camera is not index 0.", 10, parent=settings).pack(side="left")
    state = label("", 11, "#8ee3ca")
    state.pack(fill="x", padx=28, pady=10)

    def refresh():
        sessions = 0
        try:
            if PERSONAL_DATA.exists():
                from .data import read_personal
                sessions = read_personal(PERSONAL_DATA).session.nunique()
        except Exception:
            state.configure(text="Dataset needs attention. Check the log or back up data/personal.csv.")
            return
        state.configure(text=f"Complete recording sessions: {sessions} / 5 minimum     "
                             f"Personal model: {'ready' if PERSONAL_MODEL.exists() else 'not trained'}")

    def run(arguments, uses_camera=False):
        nonlocal child
        if child is not None:
            return
        try:
            index = camera.get()
            if not 0 <= index <= 9:
                raise ValueError()
        except (ValueError, tk.TclError):
            messagebox.showerror("Camera index", "Enter a camera index between 0 and 9.")
            return
        command = [sys.executable, "-u", "-m", "emotion_app", *arguments]
        if uses_camera:
            command.extend(["--camera", str(index)])
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except OSError as error:
            messagebox.showerror("Could not start", str(error))
            child = None
            return
        for button in buttons:
            button.configure(state="disabled")
        spin.configure(state="disabled")
        output.insert("end", "\nStarting " + " ".join(arguments) + " ...\n")
        output.see("end")
        process = child
        def read_output():
            for line in process.stdout:
                events.put(("line", line))
            process.stdout.close()
            events.put(("done", process.wait()))
        threading.Thread(target=read_output, daemon=True).start()

    actions = tk.Frame(root, bg="#101820")
    actions.pack(fill="x", padx=28, pady=4)
    def button(title, command, row, column):
        item = ttk.Button(actions, text=title, command=command, style="Lab.TButton")
        item.grid(row=row, column=column, sticky="ew", padx=(0, 8), pady=5)
        buttons.append(item)
    button("1. Open assignment demo", lambda: run(["run", "--demo"], True), 0, 0)
    button("2. Record labeled session", lambda: run(["collect"], True), 0, 1)
    button("3. Train my webcam model", lambda: run(["train", "--personal"]), 1, 0)
    button("4. Run my webcam model", lambda: run(["run", "--model", str(PERSONAL_MODEL)], True), 1, 1)
    button("Check camera", lambda: run(["doctor"], True), 2, 0)
    button("Rebuild assignment model", lambda: run(["train"]), 2, 1)
    actions.columnconfigure(0, weight=1)
    actions.columnconfigure(1, weight=1)
    links = tk.Frame(root, bg="#101820")
    links.pack(fill="x", padx=28, pady=10)
    def open_report():
        path = ROOT / "reports" / ("personal" if PERSONAL_MODEL.exists() else "demo") / "evaluation.html"
        if path.exists():
            webbrowser.open(path.as_uri())
        else:
            messagebox.showinfo("Report", "Train a model first to generate its evaluation report.")
    ttk.Button(links, text="Open evaluation report", command=open_report).pack(side="left")
    ttk.Button(links, text="Quick start guide", command=lambda: webbrowser.open((ROOT / "START_HERE.html").as_uri())).pack(side="left", padx=12)
    label("Repeat recording five times. Between sessions, change position or lighting slightly.", 10).pack(fill="x", padx=28)
    label("In the camera window: SPACE records a posed label. Q exits. No photos are saved.", 10).pack(fill="x", padx=28, pady=(4, 12))
    output = tk.Text(root, bg="#1c2935", fg="#d1dee8", insertbackground="white", font=("Consolas", 10),
                     height=10, wrap="word", relief="flat", padx=12, pady=12)
    output.pack(fill="both", expand=True, padx=28, pady=(0, 24))
    output.insert("end", "Ready. Start with the assignment demo or check your camera.\n"
                  "Record labeled sessions to build a personal model. Scores are expression estimates.\n")

    def pump():
        nonlocal child
        try:
            while True:
                event, value = events.get_nowait()
                if event == "line":
                    output.insert("end", value)
                    output.see("end")
                else:
                    child = None
                    output.insert("end", f"Finished (exit code {value}).\n")
                    for item in buttons:
                        item.configure(state="normal")
                    spin.configure(state="normal")
                    refresh()
                    if closing:
                        root.destroy()
                        return
        except queue.Empty:
            pass
        root.after(100, pump)

    def close():
        nonlocal closing
        if child is not None and child.poll() is None:
            closing = True
            state.configure(text="Press Q in the camera window to finish; the launcher will close afterward.")
            # Let the child release its camera and preserve atomic writes normally.
        else:
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    pump()
    root.mainloop()
