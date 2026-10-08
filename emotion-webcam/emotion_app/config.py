from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ("Happy", "Neutral", "Sad")
BASIC_FEATURES = ("eye_openness_proxy", "smile_intensity_proxy")
HOG_SIZE = 900
PERSONAL_FEATURES = BASIC_FEATURES + tuple(f"hog_{i:04d}" for i in range(HOG_SIZE))
EXTRACTOR_VERSION = "haar-hog48-clahe-v1"
DEMO_MODEL = ROOT / "models" / "demo.json"
PERSONAL_MODEL = ROOT / "models" / "personal.json"
PERSONAL_DATA = ROOT / "data" / "personal.csv"
WINDOW = "Expression Lab | Q to quit"
