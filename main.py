from __future__ import annotations

import argparse
import logging
import multiprocessing
import signal
from pathlib import Path
from typing import List, Sequence

import uvicorn
from watchfiles import run_process

from app.object_detection import WebcamObjectDetector

LOG_FILE_PATH = Path("logs/object_detections.txt")
MODEL_DIR = Path("models")
WATCH_PATHS: Sequence[str] = ("app", "main.py")


def run_object_detection() -> None:
    detector = WebcamObjectDetector(
        model_dir=MODEL_DIR,
        log_path=LOG_FILE_PATH,
        confidence_threshold=0.25,
        log_interval_seconds=1.0,
        model_variant="yolox",
    )
    detector.run()


def run_api() -> None:
    config = uvicorn.Config("app.api:app", host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()


def _configure_interrupt_handling(children: List[multiprocessing.Process]) -> None:
    def handle_signal(signum, frame):  # type: ignore[override]
        logging.info("Received signal %s; terminating child processes", signum)
        for child in children:
            if child.is_alive():
                child.terminate()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)


def _launch_children() -> None:
    object_detection_process = multiprocessing.Process(
        target=run_object_detection, name="ObjectDetection"
    )
    api_process = multiprocessing.Process(target=run_api, name="API")

    children = [object_detection_process, api_process]
    _configure_interrupt_handling(children)

    try:
        for process in children:
            process.start()

        for process in children:
            process.join()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received; shutting down processes")
    finally:
        for process in children:
            if process.is_alive():
                process.terminate()
        for process in children:
            process.join(timeout=5)


def _run_once() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        force=True,
    )
    multiprocessing.freeze_support()
    _launch_children()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the object detection loop and API server."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload. Restarts processes on code changes.",
    )
    args = parser.parse_args()

    if args.reload:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [Main] %(levelname)s: %(message)s",
            force=True,
        )
        logging.info(
            "Starting with hot reload enabled; watching paths: %s", ", ".join(WATCH_PATHS)
        )
        run_process(*WATCH_PATHS, target=_run_once)
    else:
        _run_once()


if __name__ == "__main__":
    main()
