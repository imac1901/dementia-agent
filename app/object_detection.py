from __future__ import annotations

import logging
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

from app.class_labels import CLASS_LABELS

LOGGER = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


MODEL_FILES: Dict[str, str] = {
    "object_detection_nanodet_2022nov.onnx": (
        "https://huggingface.co/opencv/object_detection_nanodet/resolve/main/"
        "object_detection_nanodet_2022nov.onnx?download=1"
    ),
    "object_detection_yolox_2022nov.onnx": (
        "https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
        "object_detection_yolox_2022nov.onnx?download=1"
    ),
    "object_detection_yolox_2022nov_int8bq.onnx": (
        "https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
        "object_detection_yolox_2022nov_int8bq.onnx?download=1"
    ),
}

MODEL_VARIANTS = {
    "nanodet": "object_detection_nanodet_2022nov.onnx",
    "yolox": "object_detection_yolox_2022nov.onnx",
    "yolox-int8": "object_detection_yolox_2022nov_int8bq.onnx",
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]

class NanoDetModel:
    """Lightweight ONNX object detector backed by OpenCV DNN."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.strides = (8, 16, 32, 64)
        self.input_size = (416, 416)
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(
            1, 1, 3
        )

        self.net = cv2.dnn.readNet(str(model_path))
        if self.net.empty():
            raise RuntimeError(f"Failed to load NanoDet model from {model_path}")
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()
        self.anchors_per_level = self._build_anchors()

    def _build_anchors(self) -> List[np.ndarray]:
        anchors: List[np.ndarray] = []
        for stride in self.strides:
            feat_h = int(self.input_size[0] / stride)
            feat_w = int(self.input_size[1] / stride)
            shift_x = np.arange(feat_w) * stride
            shift_y = np.arange(feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            cx = (xv + 0.5 * (stride - 1)).reshape(-1)
            cy = (yv + 0.5 * (stride - 1)).reshape(-1)
            anchors.append(np.column_stack((cx, cy)).astype(np.float32))
        return anchors

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        img = image.copy()
        top, left, new_h, new_w = 0, 0, self.input_size[0], self.input_size[1]
        if img.shape[0] != img.shape[1]:
            hw_ratio = img.shape[0] / img.shape[1]
            if hw_ratio > 1:
                new_h, new_w = self.input_size[0], int(self.input_size[1] / hw_ratio)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                left = int((self.input_size[1] - new_w) * 0.5)
                right = self.input_size[1] - new_w - left
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                new_h, new_w = int(self.input_size[0] * hw_ratio), self.input_size[1]
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                top = int((self.input_size[0] - new_h) * 0.5)
                bottom = self.input_size[0] - new_h - top
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    bottom,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        return img, (top, left, new_h, new_w)

    def _pre_process(self, image: np.ndarray) -> np.ndarray:
        processed = (image.astype(np.float32) - self.mean) / self.std
        return cv2.dnn.blobFromImage(processed)

    def _post_process(self, outputs: Sequence[np.ndarray]) -> np.ndarray:
        cls_scores = outputs[::2]
        bbox_preds = outputs[1::2]

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        for stride, cls_score, bbox_pred, anchors in zip(
            self.strides, cls_scores, bbox_preds, self.anchors_per_level
        ):
            cls_score = np.squeeze(cls_score, axis=0)
            bbox_pred = np.squeeze(bbox_pred, axis=0)

            # apply softmax on bbox distances
            bbox_pred = bbox_pred.reshape(-1, self.reg_max + 1)
            bbox_pred = np.exp(bbox_pred)
            bbox_pred /= np.sum(bbox_pred, axis=1, keepdims=True)
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            max_candidates = 1000
            if max_candidates > 0 and cls_score.shape[0] > max_candidates:
                topk_indices = np.argsort(cls_score.max(axis=1))[::-1][:max_candidates]
                anchors = anchors[topk_indices, :]
                bbox_pred = bbox_pred[topk_indices, :]
                cls_score = cls_score[topk_indices, :]

            x1 = anchors[:, 0] - bbox_pred[:, 0]
            y1 = anchors[:, 1] - bbox_pred[:, 1]
            x2 = anchors[:, 0] + bbox_pred[:, 2]
            y2 = anchors[:, 1] + bbox_pred[:, 3]

            boxes = np.column_stack((x1, y1, x2, y2))
            all_boxes.append(boxes)
            all_scores.append(cls_score)

        if not all_boxes:
            return np.empty((0, 6), dtype=np.float32)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        widths_heights = boxes.copy()
        widths_heights[:, 2:4] -= widths_heights[:, 0:2]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        indices = cv2.dnn.NMSBoxes(
            widths_heights.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        flat_indices = np.array(indices).reshape(-1)
        selected_boxes = boxes[flat_indices]
        selected_confidences = confidences[flat_indices]
        selected_classes = class_ids[flat_indices]

        return np.column_stack(
            (selected_boxes, selected_confidences, selected_classes)
        ).astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        letterboxed, (top, left, new_h, new_w) = self._letterbox(rgb_frame)
        blob = self._pre_process(letterboxed)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)
        detections = self._post_process(outputs)
        if detections.size == 0:
            return []

        scale_x = frame_width / new_w if new_w else 1.0
        scale_y = frame_height / new_h if new_h else 1.0

        results: List[Detection] = []
        for x1, y1, x2, y2, confidence, class_id in detections:
            if confidence < self.confidence_threshold:
                continue
            if not (0 <= class_id < len(CLASS_LABELS)):
                continue

            x1_adj = (x1 - left) * scale_x
            x2_adj = (x2 - left) * scale_x
            y1_adj = (y1 - top) * scale_y
            y2_adj = (y2 - top) * scale_y

            x1_int = max(0, min(frame_width - 1, int(round(x1_adj))))
            x2_int = max(0, min(frame_width - 1, int(round(x2_adj))))
            y1_int = max(0, min(frame_height - 1, int(round(y1_adj))))
            y2_int = max(0, min(frame_height - 1, int(round(y2_adj))))

            if x2_int <= x1_int or y2_int <= y1_int:
                continue

            results.append(
                Detection(
                    label=CLASS_LABELS[int(class_id)],
                    confidence=float(confidence),
                    box=(x1_int, y1_int, x2_int, y2_int),
                )
            )
        return results


class YOLOXModel:
    """YOLOX ONNX detector with improved multi-class performance."""

    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self.net = cv2.dnn.readNet(str(model_path))
        if self.net.empty():
            raise RuntimeError(f"Failed to load YOLOX model from {model_path}")
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

    def _letterbox(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        target_w, target_h = self.input_size[1], self.input_size[0]

        scale = min(target_w / width, target_h / height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = int(np.floor(pad_w / 2))
        pad_right = int(np.ceil(pad_w / 2))
        pad_top = int(np.floor(pad_h / 2))
        pad_bottom = int(np.ceil(pad_h / 2))

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_left, pad_top

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        letterboxed, scale, pad_left, pad_top = self._letterbox(frame_bgr)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1 / 255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)

        if isinstance(outputs, (list, tuple)):
            predictions = outputs[0]
        else:
            predictions = outputs

        predictions = np.squeeze(predictions).astype(np.float32)
        if predictions.ndim == 1:
            predictions = np.expand_dims(predictions, axis=0)

        if predictions.shape[0] == 0 or predictions.shape[1] < 6:
            return []

        boxes = predictions[:, :4]
        objectness = _sigmoid(predictions[:, 4:5])
        class_scores = _sigmoid(predictions[:, 5:])
        scores = objectness * class_scores

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        mask = confidences >= self.confidence_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if scale <= 0:
            LOGGER.warning("Invalid scale computed for YOLOX preprocessing; skipping frame")
            return []

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, frame_width - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, frame_height - 1)

        widths_heights = np.column_stack(
            (
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
            )
        )

        indices = cv2.dnn.NMSBoxes(
            widths_heights.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            self.iou_threshold,
        )
        if len(indices) == 0:
            return []

        results: List[Detection] = []
        flat_indices = np.array(indices).reshape(-1)
        for idx in flat_indices:
            if not (0 <= class_ids[idx] < len(CLASS_LABELS)):
                continue
            x1 = int(round(boxes_xyxy[idx, 0]))
            y1 = int(round(boxes_xyxy[idx, 1]))
            x2 = int(round(boxes_xyxy[idx, 2]))
            y2 = int(round(boxes_xyxy[idx, 3]))
            x2 = max(x1 + 1, min(frame_width - 1, x2))
            y2 = max(y1 + 1, min(frame_height - 1, y2))

            results.append(
                Detection(
                    label=CLASS_LABELS[int(class_ids[idx])],
                    confidence=float(confidences[idx]),
                    box=(x1, y1, x2, y2),
                )
            )

        return results


class WebcamObjectDetector:
    """Run ONNX-based object detection on a webcam feed, log, and display results."""

    def __init__(
        self,
        model_dir: Path,
        log_path: Path,
        *,
        confidence_threshold: float = 0.35,
        log_interval_seconds: float = 1.0,
        iou_threshold: float = 0.6,
        model_variant: str = "yolox",
    ) -> None:
        self.model_dir = model_dir
        self.log_path = log_path
        self.confidence_threshold = confidence_threshold
        self.log_interval_seconds = log_interval_seconds
        self.iou_threshold = iou_threshold
        self.model_variant = model_variant.lower()

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._ensure_model_file(self._resolve_model_filename())
        self.detector = self._build_detector(model_path)

    def _ensure_model_file(self, filename: str) -> Path:
        destination = self.model_dir / filename
        if destination.exists():
            return destination

        try:
            source_url = MODEL_FILES[filename]
        except KeyError as exc:  # pragma: no cover - programming error
            raise ValueError(f"No download source configured for {filename}") from exc

        LOGGER.info("Downloading %s to %s", source_url, destination)
        try:
            urllib.request.urlretrieve(source_url, destination)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(
                f"Failed to download model file from {source_url}"
            ) from exc
        return destination

    def _resolve_model_filename(self) -> str:
        try:
            return MODEL_VARIANTS[self.model_variant]
        except KeyError as exc:
            supported = ", ".join(sorted(MODEL_VARIANTS))
            raise ValueError(
                f"Unsupported model variant '{self.model_variant}'. "
                f"Supported variants: {supported}"
            ) from exc

    def _build_detector(self, model_path: Path):
        if self.model_variant == "nanodet":
            return NanoDetModel(
                model_path=model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
            )
        if self.model_variant == "yolox":
            return YOLOXModel(
                model_path=model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
            )

        supported = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(
            f"Unsupported model variant '{self.model_variant}'. Supported variants: {supported}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self.detector.detect(frame)

    def run(self) -> None:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise RuntimeError("Unable to open the default webcam")

        try:
            detections_counter: Counter[str] = Counter()
            last_log_time = time.monotonic()
            window_name = "Object Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            LOGGER.info("Starting webcam object detection loop")

            while True:
                success, frame = capture.read()
                if not success:
                    LOGGER.warning("Failed to read frame from webcam; retrying")
                    time.sleep(0.1)
                    continue

                detections = self.detect(frame)
                for detection in detections:
                    detections_counter[detection.label] += 1

                annotated_frame = frame.copy()
                self._draw_annotations(annotated_frame, detections)

                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    LOGGER.info("Quit command received from video window; stopping loop")
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    LOGGER.info("Video window closed by user; stopping loop")
                    break

                now = time.monotonic()
                if now - last_log_time >= self.log_interval_seconds:
                    self._write_log(detections_counter)
                    detections_counter.clear()
                    last_log_time = now
        finally:
            LOGGER.info("Releasing webcam resource")
            capture.release()
            cv2.destroyAllWindows()
            if "detections_counter" in locals() and detections_counter:
                self._write_log(detections_counter)

    def _draw_annotations(
        self, frame: np.ndarray, detections: Iterable[Detection]
    ) -> None:
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{detection.label} {detection.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            text_x = max(x1, 0)
            text_y = max(y1 - 10, 10)

            top_left = (
                max(text_x - 2, 0),
                max(text_y - text_height - baseline - 2, 0),
            )
            bottom_right = (
                min(text_x + text_width + 2, frame.shape[1] - 1),
                min(text_y + baseline + 2, frame.shape[0] - 1),
            )

            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def _write_log(self, counts: Counter[str]) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        if counts:
            summary = ", ".join(
                f"{label}={count}" for label, count in sorted(counts.items())
            )
        else:
            summary = "no detections"

        log_line = f"{timestamp} :: {summary}\n"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(log_line)

        LOGGER.debug("Logged detections: %s", summary)

