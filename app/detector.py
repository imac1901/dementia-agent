import io
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from openai import OpenAI

import cv2
import numpy as np
from ultralytics import YOLO

from .class_names import COCO_CLASS_NAMES

# YOLO11 model - will auto-download if not present
# Options: 'yolo11n.pt' (fastest), 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt' (most accurate)
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25  # Lower threshold since YOLO is more accurate
IOU_THRESHOLD = 0.45  # IoU threshold for NMS (handled by YOLO)
DEFAULT_INTERVAL_SECONDS = 2.0
ALLOWED_CLASSES = {"cup", "cell phone", "remote"}
# Minimum delay between API calls in seconds (helps avoid rate limits)
MIN_API_DELAY_SECONDS = 3.0


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class CaptionRequestType(Enum):
    CONTEXT = "context"  # Describe what object is doing
    ROOM = "room"  # Identify which room


@dataclass
class CaptionRequest:
    """Represents a queued caption request."""
    request_type: CaptionRequestType
    frame: np.ndarray
    detection: Optional[Detection] = None
    callback: Optional[Callable[[str], None]] = None  # Called with result
    request_id: Optional[str] = None  # Unique ID for tracking results


class YOLOXDetector:
    def __init__(
        self,
        *,
        capture_index: int = 0,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        base_dir: Optional[Path] = None,
        log_filename: str = "detections.txt",
        model_name: str = MODEL_NAME,
    ) -> None:
        self.capture_index = capture_index
        self.interval_seconds = interval_seconds
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.model_name = model_name
        self.log_path = self.base_dir / "logs" / log_filename
        self.model: Optional[YOLO] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_logged: float = 0.0
        self._captioned_labels: set[str] = set()
        self.caption_client = OpenAI()
        
        # Queue and worker thread for throttled API calls
        self._caption_queue: queue.Queue[Optional[CaptionRequest]] = queue.Queue()
        self._caption_worker_thread: Optional[threading.Thread] = None
        self._last_api_call_time: float = 0.0
        self._pending_results: Dict[str, Tuple[threading.Event, Optional[str]]] = {}  # Store (event, result) by request ID
        self._request_counter: int = 0
        self._result_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Load YOLO model (will auto-download if not present)
        self.model = YOLO(self.model_name)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="YOLOWorker", daemon=True)
        self._thread.start()
        
        # Start caption worker thread if not already running
        if self._caption_worker_thread is None or not self._caption_worker_thread.is_alive():
            self._caption_worker_thread = threading.Thread(
                target=self._caption_worker_loop, 
                name="CaptionWorker", 
                daemon=True
            )
            self._caption_worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Signal caption worker to stop
        self._caption_queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        if self._caption_worker_thread and self._caption_worker_thread.is_alive():
            self._caption_worker_thread.join(timeout=5.0)
        self._caption_worker_thread = None
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


    def _run_loop(self) -> None:
        """
        Continuously capture frames, run YOLO inference, render the visualization,
        and log detections with context when behavior changes.
        """
        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._write_log("camera_unavailable\n")
            return

        cv2.namedWindow("YOLO Live", cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                # Store latest frame for context captioning
                self._last_frame = frame.copy()

                # Run object detection (already filtered to ALLOWED_CLASSES in _infer)
                detections = self._infer(frame)

                # Render detections to live window
                self._render_frame(frame.copy(), detections)

                # Stop gracefully if user closes window
                if cv2.getWindowProperty("YOLO Live", cv2.WND_PROP_VISIBLE) < 1:
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
        """Run YOLO inference on a frame and convert results to Detection objects."""
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized")

        # Run inference with YOLO
        results = self.model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections: List[Detection] = []
        
        # YOLO returns Results objects, get the first one (single image)
        if len(results) == 0:
            return detections

        result = results[0]
        
        # Check if we have any detections
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # Get boxes and class information
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names_dict = result.names  # YOLO's class name mapping (dict)

        # Convert YOLO results to Detection objects
        # Map YOLO class IDs to COCO class names (YOLO uses COCO classes)
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Get class name from YOLO (names is a dict mapping class_id to name)
            if isinstance(class_names_dict, dict):
                yolo_class_name = class_names_dict.get(int(cls_id), f"class_{cls_id}")
            else:
                # Fallback if it's a list instead
                yolo_class_name = class_names_dict[int(cls_id)] if int(cls_id) < len(class_names_dict) else f"class_{cls_id}"
            
            # Map YOLO class names to COCO class names if needed
            # YOLO uses the same COCO classes, so we can use the name directly
            # but ensure it matches our COCO_CLASS_NAMES format
            class_name = yolo_class_name
            
            # Handle any naming differences (e.g., "cell phone" vs "cellphone")
            if class_name == "cellphone":
                class_name = "cell phone"
            
            # Only include allowed classes
            if class_name in ALLOWED_CLASSES:
                detections.append(
                    Detection(
                        label=class_name,
                        confidence=float(conf),
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    )
                )

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

                should_caption = (
                    caption_needed
                    and det.label == "remote"
                    and det.label not in self._captioned_labels
                )

                if should_caption:
                    caption_text = self._describe_context(
                        frame=self._last_frame,
                        detection=det
                    )
                    room = self._describe_room(self._last_frame)
                    message_parts.append(
                        f"{det.label} ({det.confidence:.2f}): {caption_text}; room={room}"
                    )
                    self._captioned_labels.add(det.label)

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

    def _caption_worker_loop(self) -> None:
        """Worker thread that processes caption requests from the queue with throttling."""
        while not self._stop_event.is_set():
            try:
                # Get next request (with timeout to check stop event)
                request = self._caption_queue.get(timeout=1.0)
                
                # None is a sentinel value to stop the worker
                if request is None:
                    break
                
                # Enforce minimum delay between API calls
                now = time.time()
                time_since_last_call = now - self._last_api_call_time
                if time_since_last_call < MIN_API_DELAY_SECONDS:
                    sleep_time = MIN_API_DELAY_SECONDS - time_since_last_call
                    time.sleep(sleep_time)
                
                # Process the request
                result = None
                request_id = None
                
                try:
                    if request.request_type == CaptionRequestType.CONTEXT:
                        result = self._process_context_request(request)
                    elif request.request_type == CaptionRequestType.ROOM:
                        result = self._process_room_request(request)
                    
                    # Store result and notify waiting threads
                    with self._result_lock:
                        if request.request_id and request.request_id in self._pending_results:
                            event, _ = self._pending_results[request.request_id]
                            self._pending_results[request.request_id] = (event, result)
                            event.set()
                    
                    # Call callback if provided
                    if request.callback and result:
                        request.callback(result)
                        
                except Exception as e:
                    error_msg = f"caption service failed: {e}"
                    with self._result_lock:
                        if request.request_id and request.request_id in self._pending_results:
                            event, _ = self._pending_results[request.request_id]
                            self._pending_results[request.request_id] = (event, error_msg)
                            event.set()
                    if request.callback:
                        request.callback(error_msg)
                
                # Update last API call time
                self._last_api_call_time = time.time()
                
            except queue.Empty:
                # Timeout - continue loop to check stop event
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Caption worker error: {e}")
                continue

    def _process_context_request(self, request: CaptionRequest) -> str:
        """Process a context caption request (what object is doing)."""
        if request.detection is None:
            return "none"
        
        frame = request.frame
        detection = request.detection
        
        import base64
        import cv2
        
        # Encode the cropped region
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
        
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        prompt = f"You are viewing a camera frame. Describe what the {detection.label} is doing."
        
        response = self.caption_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ],
            max_tokens=50,
        )
        
        caption = response.choices[0].message.content.strip()
        return caption or "none"

    def _process_room_request(self, request: CaptionRequest) -> str:
        """Process a room identification request."""
        frame = request.frame
        
        import base64
        import cv2
        
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return "unknown"
        
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        prompt = (
            "Which room of a home is this image most likely taken in? "
            "Choose ONE word from: kitchen, living room, bedroom, bathroom, "
            "office, hallway, unknown. If unsure, say 'unknown'."
        )
        
        response = self.caption_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }
            ],
            max_tokens=5,
        )
        
        text = response.choices[0].message.content.strip().lower()
        return text.split()[0] if text else "unknown"

    def _describe_room(self, frame: np.ndarray, timeout: float = 10.0) -> str:
        """Classify which room the full frame is taken in using GPT-4o vision.
        
        Queues the request and waits for result with timeout.
        Returns 'unknown' if timeout or error occurs.
        """
        if frame is None:
            return "unknown"
        
        # Create request ID
        with self._result_lock:
            self._request_counter += 1
            request_id = f"room_{self._request_counter}"
            event = threading.Event()
            self._pending_results[request_id] = (event, None)
        
        # Create and queue request
        request = CaptionRequest(
            request_type=CaptionRequestType.ROOM,
            frame=frame.copy(),  # Copy to avoid frame being overwritten
            detection=None,
            request_id=request_id,
        )
        self._caption_queue.put(request)
        
        # Wait for result with timeout
        if event.wait(timeout=timeout):
            with self._result_lock:
                if request_id in self._pending_results:
                    _, result = self._pending_results[request_id]
                    del self._pending_results[request_id]
                    return result or "unknown"
        
        # Timeout or result not found
        with self._result_lock:
            if request_id in self._pending_results:
                del self._pending_results[request_id]
        return "unknown"



    def _describe_context(self, frame: np.ndarray, detection, timeout: float = 10.0) -> str:
        """Generate a short caption for a single detection using GPT-4o.
        
        Queues the request and waits for result with timeout.
        Returns 'none' if timeout or error occurs.
        """
        if frame is None or detection is None:
            return "none"
        
        # Create request ID
        with self._result_lock:
            self._request_counter += 1
            request_id = f"context_{self._request_counter}"
            event = threading.Event()
            self._pending_results[request_id] = (event, None)
        
        # Create and queue request
        request = CaptionRequest(
            request_type=CaptionRequestType.CONTEXT,
            frame=frame.copy(),  # Copy to avoid frame being overwritten
            detection=detection,
            request_id=request_id,
        )
        self._caption_queue.put(request)
        
        # Wait for result with timeout
        if event.wait(timeout=timeout):
            with self._result_lock:
                if request_id in self._pending_results:
                    _, result = self._pending_results[request_id]
                    del self._pending_results[request_id]
                    return result or "none"
        
        # Timeout or result not found
        with self._result_lock:
            if request_id in self._pending_results:
                del self._pending_results[request_id]
        return "none"




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

        cv2.imshow("YOLO Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._stop_event.set()

