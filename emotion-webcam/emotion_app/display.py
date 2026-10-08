"""OpenCV desktop presentation; no network service or browser camera involved."""
import cv2
import numpy as np

from .config import LABELS

BACKGROUND = (23, 19, 16)
PANEL = (39, 32, 26)
WHITE = (239, 242, 244)
MUTED = (168, 156, 143)
TEAL = (178, 207, 74)
AMBER = (95, 190, 244)
COLORS = {"Happy": (137, 213, 95), "Neutral": (237, 188, 102), "Sad": (218, 146, 177)}


def text(image, value, position, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def wrap_text(image, value, x, y, width=290, color=MUTED, line_height=23):
    line = ""
    for word in str(value).split():
        candidate = (line + " " + word).strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] > width and line:
            text(image, line, (x, y), 0.48, color)
            y += line_height
            line = word
        else:
            line = candidate
    text(image, line, (x, y), 0.48, color)
    return y + line_height


def render(frame, boxes, label="Looking for a face", probabilities=None, fps=0,
           mode="SIMULATED DEMO", status="Face the camera in good light.", mirror=True,
           collection=None, brightness=None):
    canvas = np.full((680, 1180, 3), BACKGROUND, dtype=np.uint8)
    canvas[88:620, 830:1160] = PANEL
    text(canvas, "EXPRESSION LAB", (28, 38), 0.82, WHITE, 2)
    text(canvas, "Local webcam / Supervised learning", (28, 66), 0.48, MUTED)
    text(canvas, mode, (850, 40), 0.49, AMBER if "DEMO" in mode else TEAL)
    text(canvas, f"{fps:.1f} FPS", (850, 65), 0.48, MUTED)
    camera = cv2.flip(frame, 1) if mirror else frame.copy()
    h, w = camera.shape[:2]
    factor = min(790/w, 530/h)
    dw, dh = round(w*factor), round(h*factor)
    ox, oy = 28 + (790-dw)//2, 88 + (530-dh)//2
    preview = cv2.resize(camera, (dw, dh), interpolation=cv2.INTER_AREA)
    canvas[oy:oy+dh, ox:ox+dw] = preview
    for box in boxes:
        x, y, bw, bh = box
        if mirror:
            x = w - x - bw
        start = (ox + round(x*factor), oy + round(y*factor))
        end = (ox + round((x+bw)*factor), oy + round((y+bh)*factor))
        cv2.rectangle(canvas, start, end, TEAL if len(boxes) == 1 else AMBER, 2, cv2.LINE_AA)
    text(canvas, "PREDICTED EXPRESSION" if collection is None else "LABELED RECORDING", (850, 124), 0.45, MUTED)
    wrap_text(canvas, label, 850, 166, color=COLORS.get(label, WHITE), line_height=30)
    if probabilities is not None:
        for i, (name, score) in enumerate(zip(LABELS, probabilities)):
            y = 225 + 69*i
            text(canvas, name, (850, y), 0.52, WHITE)
            text(canvas, f"{score:.0%}", (1090, y), 0.52, COLORS[name])
            cv2.rectangle(canvas, (850, y+13), (1138, y+25), (64, 55, 46), -1)
            cv2.rectangle(canvas, (850, y+13), (850+round(288*float(score)), y+25), COLORS[name], -1)
        text(canvas, "Model scores; not calibrated", (850, 432), 0.43, MUTED)
    if collection is not None:
        count, total, phase, phases = collection
        text(canvas, f"Expression {phase}/{phases}", (850, 228), 0.58, TEAL)
        text(canvas, f"Samples {count}/{total}", (850, 265), 0.65, WHITE)
        cv2.rectangle(canvas, (850, 286), (1138, 302), (64, 55, 46), -1)
        cv2.rectangle(canvas, (850, 286), (850+round(288*count/total), 302), TEAL, -1)
        wrap_text(canvas, "Only numerical features are saved. Camera frames are not recorded.", 850, 352)
    wrap_text(canvas, status, 850, 474, color=AMBER if label in ("Uncertain", "Poor image quality") else MUTED)
    if brightness is not None:
        text(canvas, f"Face brightness: {brightness:.0f}/255", (850, 593), 0.43, MUTED)
    text(canvas, "Q  Quit     M  Mirror     R  Reset smoothing", (28, 651), 0.49, MUTED)
    text(canvas, "SPACE  Start capture" if collection else "Happy / Neutral / Sad", (850, 651), 0.46, TEAL)
    return canvas
