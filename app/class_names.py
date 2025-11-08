"""
Class name definitions for object recognition and prompt generation.

The YOLOX model embedded in this app was trained on the COCO dataset, so the
model outputs class indices that must be decoded with the original COCO label
map. We expose both the canonical label list and a broad household vocabulary
that downstream features can use for reminders or search experiences.

- HOUSEHOLD_ITEM_NAMES: a wide vocabulary of everyday objects that a person
  living with dementia might misplace. This is useful for downstream prompting
  or search features.
"""

from typing import Sequence

COCO_CLASS_NAMES: Sequence[str] = (
      "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush" 
)

_HOUSEHOLD_COCO_NAMES = {
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

HOUSEHOLD_ITEM_NAMES: Sequence[str] = tuple(
    name for name in COCO_CLASS_NAMES if name in _HOUSEHOLD_COCO_NAMES
)

