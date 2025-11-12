import io
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from openai import OpenAI


import cv2
import numpy as np
import onnxruntime as ort

from .class_names import COCO_CLASS_NAMES

MODEL_FILENAME = "object_detection_yolox_2022nov.onnx"
MODEL_URL = (
    "https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
    "object_detection_yolox_2022nov.onnx?download=1"
)
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.75
OBJ_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
DEFAULT_INTERVAL_SECONDS = 1.0


def ensure_file(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url) as response, path.open("wb") as out_file:
        out_file.write(response.read())

def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[idx] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class YOLOXDetector:
    def __init__(
        self,
        *,
        capture_index: int = 0,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        base_dir: Optional[Path] = None,
        log_filename: str = "detections.txt",
    ) -> None:
        self.capture_index = capture_index
        self.interval_seconds = interval_seconds
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.model_path = self.base_dir / "models" / MODEL_FILENAME
        self.log_path = self.base_dir / "logs" / log_filename
        self.session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[List[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stride_grid, self._stride_steps = self._build_grids()
        self._last_logged: float = 0.0
        self.caption_client = OpenAI()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        ensure_file(self.model_path, MODEL_URL)
        self.session = self._create_session()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="YOLOXWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def _create_session(self) -> ort.InferenceSession:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = [p for p in providers if p in ort.get_available_providers()]
        session = ort.InferenceSession(
            str(self.model_path),
            providers=available or ["CPUExecutionProvider"],
        )
        self._input_name = session.get_inputs()[0].name
        self._output_names = [output.name for output in session.get_outputs()]
        return session

    def _build_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        strides = [8, 16, 32]
        grid_list = []
        stride_list = []
        input_h, input_w = INPUT_SIZE
        for stride in strides:
            hs = input_h // stride
            ws = input_w // stride
            ys, xs = np.meshgrid(np.arange(hs), np.arange(ws))
            grid = np.stack((xs, ys), axis=-1).reshape(-1, 2)
            grid_list.append(grid)
            stride_list.append(np.full((grid.shape[0], 1), stride))

        grid_concat = np.concatenate(grid_list, axis=0).astype(np.float32)
        stride_concat = np.concatenate(stride_list, axis=0).astype(np.float32)
        return grid_concat, stride_concat

    def _run_loop(self) -> None:
        """
        Continuously capture frames, run YOLOX inference, render the visualization,
        and log detections with context when behavior changes.
        """
        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._write_log("camera_unavailable\n")
            return

        cv2.namedWindow("YOLOX Live", cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                # Store latest frame for context captioning
                self._last_frame = frame.copy()

                # Run object detection
                detections = self._infer(frame)

                # Render detections to live window
                self._render_frame(frame.copy(), detections)

                # Stop gracefully if user closes window
                if cv2.getWindowProperty("YOLOX Live", cv2.WND_PROP_VISIBLE) < 1:
                    self._stop_event.set()
                    break

                # Log periodically (and only when needed)
                now = time.time()
                if now - self._last_logged >= self.interval_seconds:
                    self._log_detections(now, detections)
                    self._last_logged = now

            cap.release()

        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


    def _infer(self, frame: np.ndarray) -> List[Detection]:
        if self.session is None:
            raise RuntimeError("ONNX runtime session is not initialized")

        blob, ratio = self._preprocess(frame)
        if self._input_name is None or self._output_names is None:
            raise RuntimeError("ONNX runtime session is not configured correctly")

        outputs = self.session.run(self._output_names, {self._input_name: blob})[0]
        detections = self._postprocess(outputs, ratio, frame.shape[:2])
        return detections

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        padded = np.ones((INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32) * 114.0

        ratio = min(INPUT_SIZE[0] / rgb.shape[0], INPUT_SIZE[1] / rgb.shape[1])
        resized = cv2.resize(
            rgb,
            (int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        padded[: resized.shape[0], : resized.shape[1]] = resized

        blob = np.transpose(padded, (2, 0, 1))[np.newaxis, :, :, :]
        return blob, ratio

    def _postprocess(
        self,
        outputs: np.ndarray,
        ratio: float,
        original_shape: Tuple[int, int],
    ) -> List[Detection]:
        output = outputs[0] if outputs.ndim == 3 else outputs
        num_classes = output.shape[-1] - 5
        predictions = output.reshape(-1, num_classes + 5)

        predictions[:, 0:2] = (predictions[:, 0:2] + self._stride_grid) * self._stride_steps
        predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * self._stride_steps

        boxes = predictions[:, 0:4].copy()
        boxes[:, 0] -= boxes[:, 2] * 0.5
        boxes[:, 1] -= boxes[:, 3] * 0.5
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        boxes = boxes / ratio

        h, w = original_shape
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

        obj_scores = predictions[:, 4]
        class_scores = predictions[:, 5:]

        obj_mask = obj_scores > OBJ_THRESHOLD
        if not np.any(obj_mask):
            return []

        boxes = boxes[obj_mask]
        class_scores = class_scores[obj_mask]
        obj_scores = obj_scores[obj_mask]

        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        scores = obj_scores * class_confidences

        conf_mask = scores > CONFIDENCE_THRESHOLD
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        class_ids = class_ids[conf_mask]

        if boxes.size == 0:
            return []

        keep = nms(boxes, scores, NMS_THRESHOLD)

        class_names = COCO_CLASS_NAMES[:num_classes]

        detections = [
            Detection(
                label=class_names[class_ids[idx]]
                if class_ids[idx] < len(class_names)
                else f"class_{class_ids[idx]}",
                confidence=float(scores[idx]),
                bbox=tuple(map(float, boxes[idx])),
            )
            for idx in keep
        ]
        return detections

    # def _log_detections(self, timestamp: float, detections: Sequence[Detection]) -> None:
    #     if not detections:
    #         message = "none"
    #     else:
    #         grouped: Dict[str, List[float]] = {}
    #         message_parts = []

    #         for det in detections:
    #             grouped.setdefault(det.label, []).append(det.confidence)
    #             # Generate caption text for each detected object
    #             caption_text = self._describe_context(
    #                 frame=self._last_frame,  # store last frame for captioning
    #                 detection=det
    #             )
    #             message_parts.append(
    #                 f"{det.label} ({det.confidence:.2f}): {caption_text}"
    #             )
            
    #         for label, scores in sorted(grouped.items()):
    #             max_score = max(scores)
    #             message_parts.append(f"{label} max={max_score:.2f}")
    #         message = "; ".join(message_parts)

    #     timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
    #     line = f"{timestamp_str} | {message}\n"
    #     self._write_log(line)

    def _log_detections(self, timestamp: float, detections: Sequence[Detection]) -> None:
        """
        Log detections and caption only when new or meaningfully changed objects appear.
        """
        MOTION_TOLERANCE = 30  # pixels of allowed movement before re-captioning

        # Convert detections to a compact summary: label + bbox center + size
        current_summary = {}
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # center
            w, h = x2 - x1, y2 - y1
            current_summary[det.label] = (round(cx, 1), round(cy, 1), round(w, 1), round(h, 1))

        # Compare with last detections to detect changes
        caption_needed = False
        last_summary = getattr(self, "_last_summary", {})

        # 1️⃣ Check for new or missing labels
        if set(current_summary.keys()) != set(last_summary.keys()):
            caption_needed = True
        else:
            # 2️⃣ Check for significant motion
            for label, (cx, cy, w, h) in current_summary.items():
                prev_cx, prev_cy, prev_w, prev_h = last_summary[label]
                dx = abs(cx - prev_cx)
                dy = abs(cy - prev_cy)
                dw = abs(w - prev_w)
                dh = abs(h - prev_h)

                if dx > MOTION_TOLERANCE or dy > MOTION_TOLERANCE or dw > 0.15 * w or dh > 0.15 * h:
                    caption_needed = True
                    break

        # Store for next loop
        self._last_summary = current_summary

        if not detections:
            message = "none"
        else:
            grouped: Dict[str, List[float]] = {}
            message_parts = []

            for det in detections:
                grouped.setdefault(det.label, []).append(det.confidence)

                if caption_needed:
                    caption_text = self._describe_context(
                        frame=self._last_frame,
                        detection=det
                    )
                    message_parts.append(
                        f"{det.label} ({det.confidence:.2f}): {caption_text}"
                    )
                else:
                    message_parts.append(f"{det.label} ({det.confidence:.2f})")

            for label, scores in sorted(grouped.items()):
                max_score = max(scores)
                message_parts.append(f"{label} max={max_score:.2f}")

            message = "; ".join(message_parts)

        # Log with local timezone

        # Log with local timezone
        from datetime import datetime
        timestamp_str = datetime.fromtimestamp(timestamp).astimezone().isoformat()

        line = f"{timestamp_str} | {message}\n"
        self._write_log(line)


    def _write_log(self, content: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(content)

    def _describe_context(self, frame: np.ndarray, detection) -> str:
        """Generate a short caption for a single detection using GPT-4o."""
        if frame is None or detection is None:
            return "none"

        try:
            import io, base64, cv2

            # Encode the cropped region or the full frame if desired
            x1, y1, x2, y2 = map(int, detection.bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return "(empty crop)"

            success, buffer = cv2.imencode(".jpg", crop)
            if not success:
                return "encoding_failed"

            # Convert to base64 so it’s JSON serializable
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # Describe what’s happening
            prompt = f"You are viewing a camera frame. Describe what the {detection.label} is doing."

            response = self.caption_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            #{"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }
                ],
                max_tokens=50,
            )

            caption = response.choices[0].message.content.strip()
            return caption or "none"

        except Exception as e:
            return f"caption service failed: {e}"




    def _invoke_caption_service(
        self,
        prompt: str,
        image_bytes: bytes,
        detection: Detection,
        full_frame_path: Path,  # pyright: ignore[reportUnusedParameter]
    ) -> str:
        # Use GPT-4o to generate a context description for an image crop.
        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4o" for full version
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image_bytes},
                        ],
                    }
                ],
                max_tokens=150,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"caption service failed: {e}"


    def _render_frame(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{det.label} {det.confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            top_left = (x1, max(y1 - label_height - baseline, 0))
            bottom_right = (x1 + label_width, top_left[1] + label_height + baseline)
            cv2.rectangle(frame, top_left, bottom_right, (0, 200, 0), cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (x1, top_left[1] + label_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("YOLOX Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._stop_event.set()

