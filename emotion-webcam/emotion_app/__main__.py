import argparse
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__
from .config import DEMO_MODEL, PERSONAL_DATA, PERSONAL_MODEL, ROOT


def setup_logging():
    directory = ROOT / "logs"
    directory.mkdir(exist_ok=True)
    handler = RotatingFileHandler(directory / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()],
                        format="%(asctime)s %(levelname)s %(message)s")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def confidence(value):
    number = float(value)
    if not 0.34 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Model-score threshold must be between 0.34 and 1.0.")
    return number


def parser():
    result = argparse.ArgumentParser(description="Expression Lab: local webcam expression classification")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command")
    commands.add_parser("gui", help="Open the desktop launcher (default)")
    training = commands.add_parser("train", help="Train and evaluate an assignment or personal model")
    training.add_argument("--personal", action="store_true")
    training.add_argument("--data", type=Path)
    training.add_argument("--model", type=Path)
    training.add_argument("--report-dir", type=Path)
    training.add_argument("--samples-per-class", type=positive, default=800)
    training.add_argument("--seed", type=int, default=42)
    live = commands.add_parser("run", help="Run the webcam or a local video")
    live.add_argument("--model", type=Path)
    live.add_argument("--demo", action="store_true", help="Use the simulated model even if personal.json exists")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--video", type=Path)
    live.add_argument("--headless", action="store_true")
    live.add_argument("--max-frames", type=positive)
    live.add_argument("--confidence", type=confidence, default=0.6)
    live.add_argument("--stats", type=Path)
    live.add_argument("--output", type=Path, help="Optional annotated .avi file; explicit recording only")
    collection = commands.add_parser("collect", help="Record one complete labeled webcam session")
    collection.add_argument("--camera", type=int, default=0)
    collection.add_argument("--data", type=Path, default=PERSONAL_DATA)
    collection.add_argument("--samples-per-class", type=positive, default=25)
    check = commands.add_parser("doctor", help="Check imports, detectors, model, and optionally webcam")
    check.add_argument("--camera", type=int, help="Explicitly open and release this webcam for a check")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    setup_logging()
    try:
        if arguments.command in (None, "gui"):
            from .launcher import launch
            launch()
        elif arguments.command == "train":
            from .data import read_personal, simulated_dataset
            from .model import save_model, train
            from .reporting import write_report
            from .storage import atomic_text
            if arguments.personal:
                frame = read_personal(arguments.data or PERSONAL_DATA)
            else:
                if arguments.data:
                    raise ValueError("--data is for --personal training; simulation generates its own data.")
                frame = simulated_dataset(arguments.samples_per_class, arguments.seed)
            model, metrics, _ = train(frame, arguments.personal, arguments.seed)
            model_path = arguments.model or (PERSONAL_MODEL if arguments.personal else DEMO_MODEL)
            report_dir = arguments.report_dir or ROOT / "reports" / ("personal" if arguments.personal else "demo")
            report_path = write_report(metrics, report_dir)
            if not arguments.personal:
                atomic_text(report_dir / "simulated_dataset.csv", frame.to_csv(index=False))
            save_model(model, model_path)
            print(f"Model saved: {model_path}\nReport: {report_path}\n"
                  f"Held-out accuracy: {metrics['accuracy']:.2%}\n"
                  f"Train/test samples: {metrics['train_samples']}/{metrics['test_samples']}")
            if not arguments.personal:
                print("This is simulated-data accuracy, not measured webcam accuracy.")
        elif arguments.command == "run":
            from .camera import run_webcam
            from .model import EmotionModel
            model_path = arguments.model or (DEMO_MODEL if arguments.demo or not PERSONAL_MODEL.exists() else PERSONAL_MODEL)
            stats = run_webcam(EmotionModel.load(model_path), camera=arguments.camera, video=arguments.video,
                headless=arguments.headless, max_frames=arguments.max_frames, threshold=arguments.confidence,
                stats_path=arguments.stats, output=arguments.output)
            print(json.dumps(stats, indent=2))
        elif arguments.command == "collect":
            from .camera import collect_session
            from .data import read_personal
            count = collect_session(arguments.data, arguments.camera, arguments.samples_per_class)
            if count:
                sessions = read_personal(arguments.data).session.nunique()
                print(f"Saved {count} numerical samples. Complete sessions: {sessions}/5 minimum.")
                print("Close and reopen collection for a new session; vary light and position between sessions.")
            else:
                print("Recording canceled. Incomplete session discarded; previous sessions preserved.")
        elif arguments.command == "doctor":
            import cv2
            import matplotlib
            import numpy
            import pandas
            import sklearn
            from .model import EmotionModel
            from .vision import FeatureExtractor
            FeatureExtractor()
            EmotionModel.load(DEMO_MODEL)
            result = {"app": __version__, "python": sys.version.split()[0], "opencv": cv2.__version__,
                      "numpy": numpy.__version__, "pandas": pandas.__version__,
                      "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                      "detectors": "OK", "demo_model": "OK", "camera": "not requested"}
            if arguments.camera is not None:
                from .camera import check_camera
                result["camera"] = check_camera(arguments.camera)
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Stopped. Camera resources released.")
        return 130
    except Exception as error:
        logging.getLogger(__name__).exception("Operation failed")
        print(f"\nERROR: {error}\nSee logs/app.log and docs/TROUBLESHOOTING.md.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
